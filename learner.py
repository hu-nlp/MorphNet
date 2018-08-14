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
    def __init__(self, c2i, options):
        self.model = dy.ParameterCollection()
        random.seed(1)
        self.trainer = dy.AdamTrainer(self.model)

        self.dropput_rate = options.dropout_rate
        self.ldims = options.lstm_dims
        self.cdims = options.cembedding_dims

        self.c2i = c2i

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

    def predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, self.c2i, )):
                dy.renew_cg()

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                # I- Word encoding
                for entry in conll_sentence:
                    c_embeddings = []
                    for c in entry.idChars:
                        # TODO : try different formulas like alpha/(alpha + #(w))
                        dropFlag = False  # random.random() < self.dropput_rate
                        c_embedding = self.clookup[c if not dropFlag else 0]
                        c_embeddings.append(c_embedding)

                    e_i = self.word_encoder.predict_sequence(c_embeddings)[-1]
                    entry.word_enc = dy.dropout(e_i, self.dropput_rate)
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
                """
                # III- Output encoding
                c_embeddings = []
                for entry in conll_sentence:
                    for f in entry.idFeats:
                        # TODO : try different formulas like alpha/(alpha + #(w))
                        dropFlag = False  # random.random() < self.dropput_rate
                        c_embedding = self.clookup[f if not dropFlag else 0]
                        c_embeddings.append(c_embedding)
                t_i = self.output_encoder.predict_sequence(c_embeddings)

                for entry, vec in zip(conll_sentence, t_i):
                    entry.output_enc = dy.dropout(vec, self.dropput_rate)
                """
                # IV- Decoder
                # Init for Context encoding
                for entry in conll_sentence:
                    entry.context_enc = dy.rectify(self.W_d.expr() * entry.context_enc + self.W_db.expr())
                """
                # Init for Word and Output encoding
                for entry in conll_sentence:
                    entry.comb = entry.word_enc + entry.output_enc
                """

                # predicted_sequences = []
                output_state = self.output_encoder.initial_state()
                for entry in conll_sentence:
                    if output_state.output():
                        entry.comb = entry.word_enc + output_state.output()
                    else:
                        entry.comb = entry.word_enc

                    decoder_state = self.decoder.initial_state().set_s(
                        [entry.context_enc, dy.tanh(entry.context_enc),
                         entry.comb, dy.tanh(entry.comb)])
                    predicted_sequence = []
                    predicted_char = self.c2i["<s>"]
                    counter = 0
                    while True:
                        counter += 1
                        decoder_state.add_input(self.clookup[predicted_char])
                        probs = self._get_probs(decoder_state.output())
                        predicted_char = probs.npvalue().argmax()
                        if predicted_char != self.c2i["</s>"] and counter < 50:
                            predicted_sequence.append(predicted_char)
                        else:
                            break
                    for seq_i in predicted_sequence:
                        tag_embedding = self.clookup[seq_i]
                        decoder_state.add_input(tag_embedding)
                    entry.predicted_sequence = predicted_sequence
                    # predicted_sequences.append(predicted_sequence)

                yield conll_sentence

    def _get_probs(self, rnn_output):
        output_w = dy.parameter(self.W_s)
        output_b = dy.parameter(self.W_sb)

        probs = dy.softmax(output_w * rnn_output + output_b)
        return probs

    def train(self, conll_path):
        total = 0.0
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, self.c2i))
            random.shuffle(shuffledData)

            for iSentence, sentence in enumerate(shuffledData):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                # I- Word encoding
                for entry in conll_sentence:
                    c_embeddings = []
                    for c in entry.idChars:
                        # TODO : try different formulas like alpha/(alpha + #(w))
                        dropFlag = False  # random.random() < self.dropput_rate
                        c_embedding = self.clookup[c if not dropFlag else 0]
                        c_embeddings.append(c_embedding)

                    e_i = self.word_encoder.predict_sequence(c_embeddings)[-1]
                    entry.word_enc = dy.dropout(e_i, self.dropput_rate)
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

                # III- Output encoding
                c_embeddings = []
                for entry in conll_sentence:
                    for f in entry.idFeats:
                        # TODO : try different formulas like alpha/(alpha + #(w))
                        dropFlag = False  # random.random() < self.dropput_rate
                        c_embedding = self.clookup[f if not dropFlag else 0]
                        c_embeddings.append(c_embedding)
                output_encoder = RNNSequencePredictor(self.output_encoder)
                t_i = output_encoder.predict_sequence(c_embeddings)

                for entry, vec in zip(conll_sentence, t_i):
                    entry.output_enc = dy.dropout(vec, self.dropput_rate)

                # IV- Decoder
                # Init for Context encoding
                for entry in conll_sentence:
                    entry.context_enc = dy.rectify(self.W_d.expr() * entry.context_enc + self.W_db.expr())

                # Init for Word and Output encoding
                for entry in conll_sentence:
                    entry.comb = entry.word_enc + entry.output_enc

                probs = []
                losses = []

                for entry in conll_sentence:
                    decoder_state = self.decoder.initial_state().set_s([entry.context_enc, dy.tanh(entry.context_enc),
                                                                        entry.comb, dy.tanh(entry.comb)])

                    for g_c in entry.decoder_input:
                        decoder_state = decoder_state.add_input(self.clookup[g_c])
                        p = self._get_probs(decoder_state.output())
                        probs.append(p)

                    losses += [-dy.log(dy.pick(p, o)) for p, o in zip(probs, entry.decoder_input)]

                total_losses = dy.esum(losses)
                cur_loss = total_losses.scalar_value()
                total += cur_loss
                total_losses.backward()
                self.trainer.update()
                if iSentence != 0 and iSentence % 500 == 0:
                    print("Sentence:" + str(iSentence) + " Loss:" + str(total / (iSentence + 1)))
                dy.renew_cg()
