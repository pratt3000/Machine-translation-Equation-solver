import re
from typing import Tuple

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3} # These are the reserved tokens.
        self.word2count = {}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = 4  # Count the above 4

    def addSentence(self, sentence):
        # Construct the language bit by bit.
        sentence_list = self.sentence_to_words(sentence)
        for word in sentence_list:
            self.addWord(word)

    def addWord(self, word):    

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def sentence_to_words(self, sentence):
        return re.findall(r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+", sentence.strip().lower())

# Get train file data
def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions

def readLangs(file_path = "train.txt"):

    factors, expansions = load_file(file_path)
    pairs = [[a, b] for a, b in zip(factors, expansions)]

    # Reverse pairs, make Lang instances
    lang1 = "factors"
    lang2 = "expansions"
    
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(file_path):
    input_lang, output_lang, pairs = readLangs(file_path)
    
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
