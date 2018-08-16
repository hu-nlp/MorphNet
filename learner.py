# coding=utf-8
import dynet as dy
import random
import time

import utils
from mnnl import RNNSequencePredictor
from utils import read_conll

# TODO :
"""
1 - Glorot init
2 - 
"""


class Learner:
    def __init__(self, c2i, features, options):
        self.model = dy.ParameterCollection()
        random.seed(1)
        self.trainer = dy.AdamTrainer(self.model)

        self.dropput_rate = options.dropout_rate
        self.ldims = options.lstm_dims
        self.cdims = options.cembedding_dims

        self.c2i = c2i
        self.i2c = {self.c2i[k]:k for k in self.c2i}
        self.features = features

        self.W_d = self.model.add_parameters((self.ldims, 2 * self.ldims))
        self.W_db = self.model.add_parameters(self.ldims)

        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims))

        self.word_encoder = RNNSequencePredictor(dy.VanillaLSTMBuilder(1, self.cdims, self.ldims, self.model))
        self.context_encoder = [dy.VanillaLSTMBuilder(1, self.ldims, self.ldims, self.model),
                                dy.VanillaLSTMBuilder(1, self.ldims, self.ldims, self.model)]
        self.output_encoder = dy.VanillaLSTMBuilder(1, self.cdims, self.ldims, self.model)

        self.decoder = dy.VanillaLSTMBuilder(2, self.cdims, self.ldims, self.model)

        self.W_s = self.model.add_parameters((len(self.c2i), self.ldims))
        self.W_sb = self.model.add_parameters((len(self.c2i)))

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.populate(filename)

    def convert2chars(self, lst):
        s = []
        for i in lst:
            s.append(self.i2c[i])
        return s

    def predict(self, conll_path):
        start = time.time()
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, self.c2i)):
                if iSentence % 500 == 0:
                    print "Prediction : Processing sentence number: %d" % iSentence, ", Time: %.2f" % (time.time() - start)
                    start = time.time()

                dy.renew_cg()
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                # I- Word encoding
                for entry in conll_sentence:
                    c_embeddings = []
                    for c in entry.idChars:
                        c_embedding = self.clookup[c]
                        c_embeddings.append(c_embedding)

                    e_i = self.word_encoder.predict_sequence(c_embeddings)[-1]
                    entry.word_enc = e_i
                    entry.context_lstms = [entry.word_enc, entry.word_enc]

                # II- Context encoding
                blstm_forward = self.context_encoder[0].initial_state()
                blstm_backward = self.context_encoder[1].initial_state()
                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    blstm_forward = blstm_forward.add_input(entry.word_enc)
                    blstm_backward = blstm_backward.add_input(rentry.word_enc)

                    entry.context_lstms[1] = blstm_forward.output()
                    rentry.context_lstms[0] = blstm_backward.output()

                for entry in conll_sentence:
                    entry.context_enc = dy.concatenate(entry.context_lstms)

                # Init for Context encoding
                for entry in conll_sentence:
                    entry.context_enc = dy.rectify(self.W_d.expr() * entry.context_enc + self.W_db.expr())

                output_state = self.output_encoder.initial_state()
                output_state = output_state.add_input(self.clookup[self.c2i["<st>"]])
                for entry in conll_sentence:
                    # III- Output encoding
                    entry.comb = entry.word_enc + output_state.output()

                    # IV- Decoder
                    decoder_state = self.decoder.initial_state().set_s([entry.context_enc,entry.context_enc,entry.comb, entry.comb])
                    entry.predicted_sequence = []
                    predicted = self.c2i["<s>"]
                    counter = 0
                    stop = False
                    while not stop:
                        counter += 1
                        decoder_state = decoder_state.add_input(self.clookup[predicted])
                        probs = self.softmax(decoder_state.output())
                        predicted = probs.npvalue().argmax()
                        if predicted == self.c2i["</s>"]:
                            entry.predicted_sequence.append(predicted)
                            stop = True
                        elif counter > 50:
                            stop = True
                        else:
                            entry.predicted_sequence.append(predicted)

                    for seq_i in entry.predicted_sequence:
                        if seq_i in self.features:
                            tag_embedding = self.clookup[seq_i]
                            output_state = output_state.add_input(tag_embedding)

                yield conll_sentence

    def softmax(self, rnn_output):
        output_w = dy.parameter(self.W_s)
        output_b = dy.parameter(self.W_sb)
        probs = dy.softmax(output_w * rnn_output + output_b)
        return probs

    def train(self, conll_path):
        total = 0.0
        start = time.time()
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, self.c2i))
            random.shuffle(shuffledData)

            for iSentence, sentence in enumerate(shuffledData):
                dy.renew_cg()

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                # I- Word encoding
                for entry in conll_sentence:
                    c_embeddings = []
                    for c in entry.idChars:
                        c_embedding = self.clookup[c]
                        c_embeddings.append(c_embedding)

                    e_i = self.word_encoder.predict_sequence(c_embeddings)[-1]
                    entry.word_enc = e_i
                    entry.context_lstms = [entry.word_enc, entry.word_enc]

                # II- Context encoding
                blstm_forward = self.context_encoder[0].initial_state()
                blstm_backward = self.context_encoder[1].initial_state()
                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    blstm_forward = blstm_forward.add_input(entry.word_enc)
                    blstm_backward = blstm_backward.add_input(rentry.word_enc)

                    entry.context_lstms[1] = blstm_forward.output()
                    rentry.context_lstms[0] = blstm_backward.output()

                for entry in conll_sentence:
                    entry.context_enc = dy.concatenate(entry.context_lstms)

                # IV- Decoder
                # Init for Context encoding
                for entry in conll_sentence:
                    entry.context_enc = dy.rectify(self.W_d.expr() * entry.context_enc + self.W_db.expr())

                probs = []
                losses = []
                output_state = self.output_encoder.initial_state()
                output_state = output_state.add_input(self.clookup[self.c2i["<st>"]])
                for entry in conll_sentence:
                    # III- Output encoding
                    entry.comb = entry.word_enc + output_state.output()

                    # IV- Decoder
                    # TODO
                    decoder_state = self.decoder.initial_state().set_s([entry.context_enc, entry.context_enc,entry.comb, entry.comb])

                    for gold in entry.decoder_gold_input:
                        decoder_state = decoder_state.add_input(self.clookup[gold])
                        p = self.softmax(decoder_state.output())
                        probs.append(p)

                    for gold in entry.idFeats:
                        tag_embedding = self.clookup[gold]
                        output_state = output_state.add_input(tag_embedding)

                    losses += [-dy.log(dy.pick(p, o)) for p, o in zip(probs, entry.decoder_gold_output)]

                total_losses = dy.esum(losses)
                cur_loss = total_losses.scalar_value()
                total += cur_loss
                total_losses.backward()
                self.trainer.update()
                if iSentence != 0 and iSentence % 500 == 0:
                    print "Processing sentence number: %d" % iSentence, ", Time: %.2f" % (time.time() - start) , " Loss:" + str(total / (iSentence + 1))
                    start = time.time()
