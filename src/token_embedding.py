
import argparse
import os
import random
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp
import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import wandb
from config import Model
from config import TrainingConfig


files = ["part1.txt", "part2.txt"]

def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def load_pairs(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = []
    contexts = []
    ctx_counts = np.zeros(Model.vocab_size, dtype=np.int32)

    with open(path, "r", encoding="utf-8") as f:
        tokens = []
        for line in f:
            token = line.strip()
            if token:
                tokens.append(int(token))

    for idx, token in enumerate(tokens):
        ctx_counts[token] += 1
        if idx < 3 or idx >= len(tokens) - 3:
            continue
        else:
            centers.append(token)
            contexts.append([tokens[i] for i in range(idx - 3, idx + 4) if i != idx])

    return (
        np.array(centers),
        np.array(contexts),
        ctx_counts,
    )

def build_noise_distribution(ctx_counts: np.ndarray, power: float) -> np.ndarray:
	"""Build unigram^power distribution over context vocabulary.

	Returns a float64 numpy array that sums to 1.0.
	"""
	probs = ctx_counts.astype(np.float64)
	probs = np.power(np.maximum(probs, 1.0), power)
	total = probs.sum()
	if total <= 0:
		raise ValueError("Context counts sum to zero; cannot build noise distribution")
	probs /= total
	return probs


class SGNSModel(nn.Module):
	def __init__(self, vocab_size: int, dim: int, device: str = "cpu") -> None:
		import torch
		import torch.nn as nn
		
		super().__init__()

		self.vocab_size = vocab_size
		self.dim = dim
		self.device = torch.device(device)

		self.in_embed = nn.Embedding(vocab_size, dim)
		self.out_embed = nn.Embedding(vocab_size, dim)

		bound = 0.5 / dim
		nn.init.uniform_(self.in_embed.weight, a=-bound, b=bound)
		nn.init.zeros_(self.out_embed.weight)

		self.in_embed.to(self.device)
		self.out_embed.to(self.device)


	def sgns_loss(self, centers, pos_contexts, neg_contexts):
		import torch
		import torch.nn.functional as F

		B = centers.size(0)
		K = neg_contexts.size(1)

		v_c = self.in_embed(centers)  # [B, D]
		v_o = self.out_embed(pos_contexts)  # [B, D]
		pos_score = (v_c * v_o).sum(dim=1)  # [B]
		pos_loss = F.logsigmoid(pos_score).sum()

		v_n = self.out_embed(neg_contexts)  # [B, K, D]
		# [B, 1, D] * [B, K, D] -> [B, K]
		neg_score = (v_c.unsqueeze(1) * v_n).sum(dim=2)
		neg_loss = F.logsigmoid(-neg_score).sum()

		loss = -(pos_loss + neg_loss) / B
		return loss

def setup(rank, world_size):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	torch.cuda.set_device(rank)
	dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def save_embeddings(embeddings: np.ndarray, file_path: str) -> None:
    """Save embeddings to a file in NumPy format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, embeddings)
    print(f"Embeddings saved to {file_path}")

	
def train(rank, world_size, load = None) -> None:

	setup(rank, world_size)
	set_seed(TrainingConfig.seed)

	if len(files) != world_size:
		raise ValueError(f"len(files)={len(files)} must equal world_size={world_size} (one shard per rank)")
	centers_np, contexts_np, ctx_counts = load_pairs(files[rank])
	noise_probs_np = build_noise_distribution(ctx_counts, power=TrainingConfig.power)

	device = torch.device(f"cuda:{rank}")

	class PairDataset(Dataset):
		def __init__(self, c_arr: np.ndarray, o_arr: np.ndarray):
			self.c = c_arr
			self.o = o_arr

		def __len__(self):
			return self.c.shape[0]

		def __getitem__(self, idx: int):
			return self.c[idx], self.o[idx]

	dataset = PairDataset(centers_np, contexts_np)

	loader = DataLoader(dataset, batch_size=TrainingConfig.batch_size, shuffle=True,  num_workers=0)

	noise_probs = torch.from_numpy(noise_probs_np).to(device=device, dtype=torch.float32)

	model = SGNSModel(vocab_size=Model.vocab_size, dim=Model.embedding_dim, device=device)
	optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.learning_rate)
	if load is not None:
		checkpoint_model = torch.load(load[0])
		checkpoint_optimizer = torch.load(load[1])
		model.load_state_dict(checkpoint_model)
		optimizer.load_state_dict(checkpoint_optimizer)

	model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

	if rank == 0:
		wandb.init(project="token_embedding", name=f"run_{rank}")

	total_steps = 0
	for epoch in range(1, TrainingConfig.epochs + 1):
		running_loss = 0.0

		for step, batch in enumerate(loader, start=1):
	
			centers_b, pos_b = batch 
			centers_b = centers_b.to(device=device, dtype=torch.int)
			pos_b = pos_b.to(device=device, dtype=torch.int)

			K = Model.neg_sampling
			centers_b = centers_b.repeat_interleave(Model.context_size)
			neg_b = torch.multinomial(noise_probs, num_samples=centers_b.size(0) * K, replacement=True)
			neg_b = neg_b.view(centers_b.size(0), K)
			for i in range(neg_b.size(0)):
				for j in range(neg_b.size(1)):
					# resamling when needed
					if neg_b[i, j] in pos_b[i // Model.context_size]:
						while neg_b[i, j] in pos_b[i // Model.context_size]:
							neg_b[i, j] = torch.multinomial(noise_probs, num_samples=1).item()
			neg_b = neg_b.to(device=device, dtype=torch.int)
			pos_b = pos_b.view(-1)




			loss = model.module.sgns_loss(centers_b, pos_b, neg_b)

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			total_steps += 1




			if step % 100 == 0 and rank == 0:
				avg = running_loss / 100
				print(f"epoch {epoch} step {step} | avg_loss {avg:.4f}")
				running_loss = 0.0
				wandb.log({
					"epoch": epoch,
					"step": step,
					"avg_loss": avg,
					"total_steps": total_steps
				})
			if step % 1000 == 0 and rank == 0:
				torch.save(model.module.state_dict(), f"sgns_epoch{epoch}_step{step}.pt")
				torch.save(optimizer.state_dict(), f"sgns_epoch{epoch}_step{step}_optimizer.pt")
				old_step = step - 1000
				if old_step > 0:
					old_model = f"sgns_epoch{epoch}_step{old_step}.pt"
					old_opt = f"sgns_epoch{epoch}_step{old_step}_optimizer.pt"
					if os.path.exists(old_model):
						os.remove(old_model)
					if os.path.exists(old_opt):
						os.remove(old_opt)


	if rank == 0:
		in_emb = model.module.in_embed.weight.detach().cpu().numpy().astype(np.float32)
		out_emb = model.module.out_embed.weight.detach().cpu().numpy().astype(np.float32)
		save_embeddings(in_emb, "in_embeddings.npy")
		save_embeddings(out_emb, "out_embeddings.npy")

		wandb.save("in_embeddings.npy")
		wandb.save("out_embeddings.npy")
		wandb.finish()

	dist.barrier()
	cleanup()


if __name__ == "__main__":
	world_size = torch.cuda.device_count()
	if world_size == 0:
		raise RuntimeError("No CUDA devices available for distributed training.")
	mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)