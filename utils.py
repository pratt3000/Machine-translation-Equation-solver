import torch
import re
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def indexesFromSentence(lang, sentence):
    words = lang.sentence_to_words(sentence) # splits the equation into words using regex.

    # If a variable other than a, c, h, i, j, k, n, o, s, t, x, y, z, sin, cos, tan, *, +, -, (, ) is encountered.
    unknown_elements = list(set([element for element in words if element not in lang.word2index])) # Get unknown elements if any.

    if len(unknown_elements) == 0:
        ans = [lang.word2index.get(word, 33) for word in words]
        return ans, None
    if len(unknown_elements) == 1:
        if re.match(r'^[a-z]$', unknown_elements[0]): # If its only 1 unknown element and its from a-z, then replace it with "a"'s token, 33.
            ans = [lang.word2index.get(word, 33) for word in words] 
            return ans, unknown_elements[0]
    
    print("There are multiple unseen elements in your input pl check the guidelines in the pdf regarding valid inputs.")
    print("Now you'll probably get a garbage answer.")
    
    ans = [lang.word2index.get(word, 3) for word in words] # <UNK> token here (3).
    return ans, None


def tensorFromSentence(lang, sentence, target_tensor=False):
    indexes, ele = indexesFromSentence(lang, sentence)
    #add start and end to tgt tensors only because we will need these when passing input to decoder. (first input needs to be start, 1 to the right)
    if target_tensor:
        indexes = [lang.word2index["<SOS>"]] + indexes + [lang.word2index["<EOS>"]]
    return torch.tensor(indexes, dtype=torch.long), ele


def get_dataloader(batch_size, input_lang, output_lang, pairs, device, MAX_LENGTH=32):

    # zeros because we will be padding the sentences to make them of equal length. padding tok is set to 0.
    input_ids = np.zeros((len(pairs), MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((len(pairs), MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids, _ = tensorFromSentence(input_lang, inp)
        tgt_ids, _ = tensorFromSentence(output_lang, tgt, target_tensor=True)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    data = TensorDataset(torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device))

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader