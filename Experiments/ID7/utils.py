# coding=utf-8
import re
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import pickle
import numpy as np

class ConllEntry:
    def __init__(self, id, form, lemma, pos, xpos, feats, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.xpos = xpos
        self.pos = pos
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None
        self.pred_pos = None
        self.predicted_sequence = None

        self.idChars = []
        self.chars = []
        self.idFeats = []
        self.idWord = []
        self.decoder_gold_input = []
        self.decoder_gold_output = []

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pred_pos, self.xpos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    # Character vocabulary
    c2i = {"UNK": 0, "ROOT_WORD": 1}
    t2i = {"UNK": 0,"<s>": 1}
    tokens = []
    for line in open(conll_path, 'r'):
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            tokens = []
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5],
                                   int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])

                for char in entry.norm:
                    if char not in c2i:
                        c2i[char] = len(c2i)

                for feat in entry.feats.split("|"):
                    if feat not in t2i:
                        t2i[feat] = len(t2i)
                tokens.append(entry)
    return c2i, t2i


def read_conll(fh, c2i, t2i):
    # Character vocabulary
    tokens = []
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1: yield tokens
            tokens = []
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5],
                                   int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])
                chars_of_word = []
                for char in entry.norm:
                    if char in c2i:
                        chars_of_word.append(c2i[char])
                    else:
                        chars_of_word.append(c2i["UNK"])
                entry.idChars = chars_of_word
                
                feats_of_word = []
                for feat in entry.feats.split("|"):
                    if feat in t2i:
                        feats_of_word.append(t2i[feat])
                    else:
                        feats_of_word.append(t2i["UNK"])
                entry.idFeats = feats_of_word
                
                
                entry.decoder_gold_input =  [t2i["<s>"]] + entry.idFeats + [t2i["<s>"]] 
                entry.decoder_gold_output = entry.idFeats 
                tokens.append(entry)

    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


def lower(text):
    text = text.replace("Ü", "ü")
    text = text.replace("Ğ", "ğ")
    text = text.replace("İ", "i")
    text = text.replace("Ş", "ş")
    text = text.replace("Ç", "ç")
    text = text.replace("Ö", "ö")
    return text.lower()

def normalize(word):
    w = lower(word)
    w = re.sub(r"``", '"', w)
    w = re.sub(r"''", '"', w)
    return w



def load_embeddings_file(file_name, lower=False, type=None):
    if type == None:
        file_type = file_name.rsplit(".",1)[1] if '.' in file_name else None
        if file_type == "p":
            type = "pickle"
        elif file_type == "bin":
            type = "word2vec"
        elif file_type == "vec":
            type = "fasttext"
        else:
            type = "word2vec"

    if type == "word2vec":
        model = KeyedVectors.load_word2vec_format(file_name, binary=True, unicode_errors="ignore")
        words = model.index2entity
    elif type == "fasttext":
        model = FastText.load_fasttext_format(file_name)
        words = [w for w in model.wv.vocab]
    elif type == "pickle":
        with open(file_name,'rb') as fp:
            model = pickle.load(fp)
        words = model.keys()

    if lower:
        vectors = {word.lower(): model[word] for word in words}
    else:
        vectors = {word: model[word] for word in words}

    if "UNK" not in vectors:
        unk = np.mean([vectors[word] for word in vectors.keys()], axis=0)
        vectors["UNK"] = unk

    return vectors, len(vectors["UNK"])