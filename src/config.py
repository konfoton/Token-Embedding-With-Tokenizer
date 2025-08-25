from dataclasses import dataclass

@dataclass
class Model:
    vocab_size = 19744
    embedding_dim = 200
    context_size = 6
    neg_sampling = 5


@dataclass
class TrainingConfig:
    batch_size = 1024
    learning_rate = 0.001
    num_epochs = 10
    epochs = 2
    seed = 42
    power = 0.75