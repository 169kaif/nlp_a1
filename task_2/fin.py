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
        sent = " ".join(tokid)
        return sent
    
    def create_mat(self):
        return self.coeff_mat/self.coeff_mat.sum(1, keepdims=True)
    
    def laplace_mat(self):
        prob_mt = self.coeff_mat + 1
        return prob_mt/prob_mt.sum(1, keepdims=True)
    
    def kneser_ney_mat(self, d = 0.75):
        prob_mt = self.coeff_mat
        frst_trm = np.maximum(prob_mt - d, 0)/prob_mt.sum(1, keepdims=True)
        lmbd_trm = (d/prob_mt.sum(axis = 1,keepdims=True)) * np.count_nonzero(prob_mt, axis=1, keepdims=True)
        cont = np.count_nonzero(prob_mt, axis=0, keepdims=True)
        pcnt_trm = cont/ cont.sum()
        return frst_trm + lmbd_trm * pcnt_trm
    
    def genrate(self, emotion, size = -1):
        idx = self.get_tokens("<s>")[0]
        generated_sequence = ""
        i = 0
        if emotion == 'sad':
            mat = self.sad_mat
        elif emotion == 'joy':
            mat = self.joy_mat
        elif emotion == 'love':
            mat = self.lov_mat
        elif emotion == 'angry':
            mat = self.ang_mat
        elif emotion == 'fear':
            mat = self.fea_mat
        elif emotion == 'surprise':
            mat = self.sup_mat
        else:
            mat = np.zeros_like(self.coeff_mat)
        
        curr_mat = self. prob_mat + mat
        curr_mat /= curr_mat.sum(1, keepdims=True)

        while (self.get_seq([idx]) != "<e>"):
            if(i == size):
                generated_sequence += "..."
                break
            elif(i != 0):
                generated_sequence += self.get_seq([idx]) + " "
            p = curr_mat[idx]
            idx = int(np.random.choice(len(p), p=p))
            i += 1
        return generated_sequence
    
    def gen_matrix(self, tokens):
        max_tkn = max(tokens) + 1
        coeff_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        sad_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        joy_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        lov_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        ang_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        fea_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        sup_mat = np.zeros((max_tkn, max_tkn), dtype=np.float64)
        for t1, t2 in zip(tokens, tokens[1:]):
            coeff_mat[t1, t2] += 1
            if(coeff_mat[t1, t2] == 1):
                emote = emotion_scores(self.get_seq([t1,t2]))
                sad_mat[t1, t2] = emote[0]['score']
                joy_mat[t1, t2] = emote[1]['score']
                lov_mat[t1, t2] = emote[2]['score']
                ang_mat[t1, t2] = emote[3]['score']
                fea_mat[t1, t2] = emote[4]['score']
                sup_mat[t1, t2] = emote[5]['score']
        return coeff_mat, sad_mat, joy_mat, lov_mat, ang_mat, fea_mat, sup_mat

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

        self.coeff_mat, self.sad_mat, self.joy_mat, self.lov_mat, self.ang_mat, self.fea_mat, self.sup_mat = self.gen_matrix(self.tokenized)

        if(smoothing == 1):
            self.prob_mat = self.laplace_mat()
        elif(smoothing == 2):
            self.prob_mat = self.kneser_ney_mat()
        else:
            self.prob_mat = self.create_mat()

    def save(self, filename = "bigram_mdl.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def get_emotion_score(self, size = -1, emotion = None):
        sntnc = self.genrate(emotion, size = size)
        emt_scr = emotion_scores(sntnc)
        emote = max(emt_scr, key=lambda x: x['score'])
        return sntnc, emote["label"], emote["score"]





def load(filename = "bigram_mdl.pkl"):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj



bgram = load()
        

# bgram = BigramLM('corpus.txt')
for i in range(50):
    sent, label, score = bgram.get_emotion_score(size = 50, emotion="joy")
    print(label, " - ", score)

# bgram.save()