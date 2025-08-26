# Tokenizer
I trained my tokenizer using BPE (bytes UTF8 level) algorithm on app. 2GB of data (fineweb dataset). 
Tranining lasted app. 30 minutes and was perfomed on single A100 GPU. 
# Token Embedding
I trained token embedding on token lvl assocaited with my tokenizer using 
Skip-Gram wirh negative sampling method. Training is adjusted to work on multiple nodes 