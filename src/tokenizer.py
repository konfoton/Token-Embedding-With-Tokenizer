def parse_merge_history(path: str) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f):
            line = raw.strip()
            parts = line.split()
            a, b = map(int, parts)
            result[lineno + 256] = (a, b)
    return result


def merge_pair(list_of_bytes, pair, idx):
    newid = []
    i = 0
    while i < len(list_of_bytes):
        if i < len(list_of_bytes) - 1 and (list_of_bytes[i], list_of_bytes[i + 1]) == pair:
            newid.append(idx)
            i += 2
        else:
            newid.append(list_of_bytes[i])
            i += 1
    return newid


def encoder(text, merges_history):
    list_of_tokens = list(text.encode('utf-8'))
    for idx, pair in merges_history.items():
        list_of_tokens = merge_pair(list_of_tokens, pair, idx)
    return list_of_tokens


def decoder(list_of_tokens, merges_history, vocab = None):
    if vocab is None:
        for idx, pair in merges_history.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] 
    decoded_text = b''.join([vocab[token] for token in list_of_tokens])
    return decoded_text.decode('utf-8', errors='ignore')