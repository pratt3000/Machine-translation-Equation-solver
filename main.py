import sys
import torch
import pickle
import numpy as np
from typing import Tuple

from model import T5custom
from poly_lang import Lang
from utils import tensorFromSentence


MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps") # Pl avoid mps as Pytorch doesnt support all functionalities of my code yet. (torch.roll)
else:
    device = torch.device("cpu")
MAX_LENGTH = 32 # Maximum sentence length to consider, can be 29, but kept it 32 for <SOS>, <EOS>.
print("Device being used:", device)

model = T5custom(
        dim = 128, 
        max_seq_len = MAX_LENGTH, 
        enc_num_tokens = 36, # 36
        enc_depth = 3,
        enc_heads = 8,
        enc_dim_head = 64,
        enc_mlp_mult = 4,
        dec_num_tokens = 36, # 36 
        dec_depth = 3,
        dec_heads = 8,
        dec_dim_head = 64,
        dec_mlp_mult = 4,
        dropout = 0.2,
        tie_token_emb = True
    ).to(device)
model.load_state_dict(torch.load("weights/best_model_state_dict.pt", map_location=device))
model.eval()

with open("weights/poly_language.pkl", 'rb') as inp:
    poly_lang = pickle.load(inp)

def predict(factors: str):

    input_tensor, ele = tensorFromSentence(poly_lang, factors)
    input_ids = np.zeros((1, MAX_LENGTH), dtype=np.int32)
    input_ids[0, :len(input_tensor)] = input_tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    
    tgt_seq = torch.zeros((1, 1), dtype=torch.long).to(device)
    tgt_seq[0, 0] = poly_lang.word2index["<SOS>"] #1
    
    stop = False
    idx = 0
    while not stop:
        pred = model(input_ids, tgt_seq)
        cur_toks = pred.argmax(dim=-1)[:, -1] 
        tgt_seq = torch.cat((tgt_seq, cur_toks.unsqueeze(-1)), dim=-1)
        idx += 1
        if idx == MAX_LENGTH-1:
            stop = True
    pred = tgt_seq
    
    decoded_words = []
    for idx in pred[0]:
        if idx.item() == 2: # EOS token, stop decoding at this point
            break
        if idx.item() == 1: # SOS token
            continue
        decoded_words.append(poly_lang.index2word[idx.item()])

    if ele: # This is if we found an unknown variable name. Re-repacing it with the original variable name.
        decoded_words = [ele if word == "a" else word for word in decoded_words]

    factors = "".join(decoded_words)
    return factors
# --------- END OF IMPLEMENT THIS --------- #

def main(filepath: str):
    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")