import numpy as np
import pickle
from utils import emotion_scores


import warnings
warnings.filterwarnings("ignore")

class BigramLM:
    def get_tokens(self, seq):
        toks = seq.split()
        return [self.tok_id[tok] for tok in toks]
    
    def get_seq(self, ids):
        tokid = [self.id_tok[id] for id in ids]
        sent = "".join(tokid)
        return sent
    
    def create_mat(self, tokens):
        max_tkn = max(tokens) + 1
        coeff_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        for t1, t2 in zip(tokens, tokens[1:]):
            coeff_mat[t1, t2] += 1
        return coeff_mat/coeff_mat.sum(1, keepdims=True)
    
    def laplace_mat(self, tokens):
        max_tkn = max(tokens) + 1
        coeff_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        for t1, t2 in zip(tokens, tokens[1:]):
            coeff_mat[t1, t2] += 1
        coeff_mat += 1
        return coeff_mat/coeff_mat.sum(1, keepdims=True)
    
    def kneser_ney_mat(self, tokens, d = 0.75):
        max_tkn = max(tokens) + 1
        coeff_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        for t1, t2 in zip(tokens, tokens[1:]):
            coeff_mat[t1, t2] += 1
        frst_trm = np.maximum(coeff_mat - d, 0)/coeff_mat.sum(1, keepdims=True)
        lmbd_trm = (d/coeff_mat.sum(axis = 1,keepdims=True)) * np.count_nonzero(coeff_mat, axis=1, keepdims=True)
        cont = np.count_nonzero(coeff_mat, axis=0, keepdims=True)
        pcnt_trm = cont/ cont.sum()
        return frst_trm + lmbd_trm * pcnt_trm
    
    def genrate(self, size = -1):
        idx = self.get_tokens("<s>")[0]
        generated_sequence = ""
        i = 0
        while (self.get_seq([idx]) != "<e>"):
            if(i == size):
                generated_sequence += "..."
                break
            elif(i != 0):
                generated_sequence += self.get_seq([idx]) + " "
            p = self.coeff_mat[idx]
            idx = int(np.random.choice(len(p), p=p))
            i += 1
        return generated_sequence
    
    def __init__(self, file, smoothing = 0):

        self.tokens = set()
        self.data = ""

        with open(file, 'r') as fil:
            data = fil.read()
        lin_data = data.split("\n")
        for sntc in lin_data:
            fsntc = f'<s> {sntc} <e> '
            self.data += fsntc
            self.tokens.update(fsntc.split())
        
        self.tok_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_tok = {index: token for token, index in self.tok_id.items()}
        self.tokenized = self.get_tokens(self.data)

        if(smoothing == 1):
            self.coeff_mat = self.laplace_mat(self.tokenized)
        elif(smoothing == 2):
            self.coeff_mat = self.kneser_ney_mat(self.tokenized)
        else:
            self.coeff_mat = self.create_mat(self.tokenized)

    def save(self, filename = "bigram_mdl.pkl"):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"Object saved to '{filename}' successfully.")
        except Exception as e:
            print(f"Error occurred while saving object: {e}")

    def get_emotion_score(self, size = -1):
        sntnc = self.genrate(size = size)
        emt_scr = emotion_scores(sntnc)
        emote = max(emt_scr, key=lambda x: x['score'])
        return sntnc, emote["label"], emote["score"]





def load(filename = "bigram_mdl.pkl"):
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            print(f"Object retrieved from '{filename}' successfully.")
            return obj
        except Exception as e:
            print(f"Error occurred while retrieving object: {e}")




        

bgram = BigramLM('corpus.txt',2)
sent, label, score = bgram.get_emotion_score(size = 15)
print(sent)
print(label, " - ", score)




bgram.save()


# bgram = load()
# bgram.genrate(10)