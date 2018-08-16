# coding=utf-8
from collections import Counter
import os, re, codecs, string


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
        self.idFeats = []
        self.decoder_gold_input = []
        self.decoder_gold_output = []

        self.comb = None
        self.context_enc = None
        self.word_enc = None


    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pred_pos, self.xpos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    # Character vocabulary
    c2i = {"_UNK": 0, "<s>": 1, "</s>":2, "<st>":3}
    features = set()
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

                chars_of_word = []
                for char in tok[1]:
                    if char not in c2i:
                        c2i[char] = len(c2i)
                    chars_of_word.append(c2i[char])
                entry.idChars = chars_of_word

                feats_of_word = []
                for feat in tok[5].split("|"):
                    if feat not in c2i:
                        c2i[feat] = len(c2i)
                        features.add(c2i[feat])
                    feats_of_word.append(c2i[feat])
                entry.idFeats = feats_of_word
                tokens.append(entry)

    return c2i, features


def read_conll(fh, c2i):
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
                for char in tok[1]:
                    if char in c2i:
                        chars_of_word.append(c2i[char])
                    else:
                        chars_of_word.append(c2i["_UNK"])
                entry.idChars = chars_of_word

                feats_of_word = []
                for feat in tok[5].split("|"):
                    if feat in c2i:
                        feats_of_word.append(c2i[feat])
                    else:
                        feats_of_word.append(c2i["_UNK"])
                entry.idFeats = feats_of_word

                decoder_input = [c2i["<s>"]]
                for c in entry.lemma:
                    if c in c2i:
                        decoder_input.append(c2i[c])
                    else:
                        decoder_input.append(c2i["_UNK"])

                decoder_input.extend(entry.idFeats)

                entry.decoder_gold_input = decoder_input
                entry.decoder_gold_output = decoder_input[1:] + [c2i["</s>"]]
                tokens.append(entry)

    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


def normalize(word):
    w = word.lower()
    w = re.sub(r"``", '"', w)
    w = re.sub(r"''", '"', w)
    return w