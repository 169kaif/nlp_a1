import numpy as np
from transformers import BertTokenizer

class BigramLM:
    def get_tokens(self, seq):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq))

    def get_seq(self, ids):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer.convert_ids_to_tokens(ids)
    
    def create_matrix(self, tokens):
        max_tkn = max(tokens) + 1
        coeff_mat = np.zeros((max_tkn, max_tkn), dtype=np.int32)
        for t1, t2 in zip(tokens, tokens[1:]):
            coeff_mat[t1, t2] += 1
        return coeff_mat
    
    def __init__(self, data):
        self.token_data = self.get_tokens(data)
        self.coeff_mat = self.create_matrix(self.token_data)
    
    def genrate(self, seq_ln, strt = 1045):
        idx = strt
        i = 0
        while True:
            print(self.get_seq(idx))
            if(i == seq_ln):
                break
            p  = self.coeff_mat[idx]
            p = p/p.sum()
            idx = np.random.choice(len(p), p=p)
            i += 1

        
with open('corpus.txt', 'r') as fil:
    inp_data = fil.read()
bgram = BigramLM(inp_data)
bgram.genrate(10)