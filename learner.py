# coding=utf-8
import dynet as dy
import random
import time

import utils
from utils import read_conll, load_embeddings_file

# TODO :
"""
1. Word Encoding  - OK
2. Input word encoding
3. Att. word encoding
4. Word Context
5. Tag Context 
6. Check out the evaluation criteria
"""


class Learner:
    def __init__(self, c2i, w2i, features, options):
        self.model = dy.ParameterCollection()
        random.seed(1)
        self.trainer = dy.AdamTrainer(self.model, )
        self.dropput_rate = options.dropout_rate

        self.ldims = options.enc_lstm_dims
        self.cdims = options.cembedding_dims
        self.wdims = options.wembedding_dims


        self.word_enc = True
        self.word_enc_bilstm = True
        self.external_embbeddins = True

        self.c2i = c2i
        self.w2i = w2i
        self.features = features

        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims), init=dy.GlorotInitializer())
        self.wlookup = self.model.add_lookup_parameters((len(w2i), self.wdims), init=dy.GlorotInitializer())

        if self.external_embbeddins:
            self.embedding_out_dim = self.wdims
            ext_embeddings, ext_emb_dim = load_embeddings_file(options.external_embedding, lower=True, type=options.external_embedding_type)
            self.ext_embeddings = ext_embeddings
            assert (ext_emb_dim == self.wdims)
            print("Initializing word embeddings by pre-trained vectors")
            count = 0
            for word in self.w2i:
                if word.lower() in ext_embeddings:
                    count += 1
                    self.wlookup.init_row(self.w2i[word], ext_embeddings[word.lower()])
            print("Vocab size: %d; #words having pretrained vectors: %d" % (len(self.w2i), count))
        else:
            self.embedding_out_dim = 0

        if self.word_enc:
            self.word_enc_out_dim = self.ldims
            if self.word_enc_bilstm:
                self.word_encoder = [dy.VanillaLSTMBuilder(1, self.cdims, self.ldims/2, self.model),
                                     dy.VanillaLSTMBuilder(1, self.cdims, self.ldims/2, self.model)]
            else:
                self.word_encoder = dy.VanillaLSTMBuilder(1, self.cdims, self.ldims, self.model)
        else:
            self.word_enc_out_dim = 0

        self.decoder = dy.VanillaLSTMBuilder(2, self.cdims, self.ldims, self.model)

        self.W_s = self.model.add_parameters((len(self.c2i), self.ldims), init=dy.GlorotInitializer())
        self.W_sb = self.model.add_parameters(len(self.c2i), init=dy.GlorotInitializer())

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.populate(filename)

    def softmax(self, rnn_output):
        output_w = dy.parameter(self.W_s)
        output_b = dy.parameter(self.W_sb)
        probs = dy.softmax(output_w * rnn_output + output_b)
        return probs

    def encode_word(self, entry):
        encoding = None
        if self.word_enc_bilstm:
            blstm_forward = self.word_encoder[0].initial_state()
            blstm_backward = self.word_encoder[1].initial_state()
            for c, rc in zip(entry.idChars, reversed(entry.idChars)):
                blstm_forward = blstm_forward.add_input(self.clookup[c])
                blstm_backward = blstm_backward.add_input(self.clookup[rc])
                encoding = dy.concatenate([blstm_forward.output(), blstm_backward.output()])
        else:
            char_lstm = self.word_encoder.initial_state()
            for c in entry.idChars:
                c_embedding = self.clookup[c]
                char_lstm = char_lstm.add_input(c_embedding)
            encoding = char_lstm.output()
        return encoding

    def predict(self, conll_path):
        start = time.time()
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, self.c2i, self.w2i)):
                dy.renew_cg()

                if iSentence != 0 and iSentence % 500 == 0:
                    print "Prediction : Processing sentence number: %d" % iSentence, ", Time: %.2f" % (time.time() - start)
                    start = time.time()

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                # I- Word encoding
                for entry in conll_sentence:
                    entry.word_enc = self.encode_word(entry)

                for entry in conll_sentence:
                    # IV- Decoder
                    cell_state1 = dy.zeros(self.ldims)
                    cell_state2 = dy.zeros(self.ldims)
                    # hidden_state = dy.rectify(self.W_red.expr() * self.wlookup[entry.idWord] + self.W_redb.expr())

                    decoder_state = self.decoder.initial_state().set_s([cell_state1, cell_state2, self.wlookup[entry.idWord], entry.word_enc])

                    entry.predicted_sequence = []
                    predicted = self.c2i["<s>"]
                    counter = 0
                    stop = False
                    while not stop:
                        counter += 1
                        #if self.word_enc or self.external_embbeddins:
                            #comb = dy.concatenate([self.clookup[predicted],entry.word_enc])
                        #else:
                            #comb = dy.concatenate([self.clookup[predicted]])
                        comb = dy.concatenate([self.clookup[predicted]])
                        decoder_state = decoder_state.add_input(comb)
                        probs = self.softmax(decoder_state.output())
                        predicted = probs.npvalue().argmax()
                        if predicted == self.c2i["</s>"]:
                            entry.predicted_sequence.append(predicted)
                            stop = True
                        elif counter > 50:
                            stop = True
                        else:
                            entry.predicted_sequence.append(predicted)
                yield conll_sentence

    def train(self, conll_path):
        total = 0.0
        start = time.time()
        with open(conll_path, 'r') as conllFP:
            shuffled_data = list(read_conll(conllFP, self.c2i, self.w2i))
            random.shuffle(shuffled_data)

            for iSentence, sentence in enumerate(shuffled_data):
                dy.renew_cg()

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                # I- Word encoding
                for entry in conll_sentence:
                    entry.word_enc = self.encode_word(entry)

                probs = []
                losses = []
                for entry in conll_sentence:
                    # IV- Decoder
                    cell_state1 = dy.zeros(self.ldims)
                    cell_state2 = dy.zeros(self.ldims)
                    #hidden_state = dy.rectify(self.W_red.expr() * self.wlookup[entry.idWord] + self.W_redb.expr())

                    decoder_state = self.decoder.initial_state().set_s([cell_state1, cell_state2, self.wlookup[entry.idWord], entry.word_enc])
                    for gold in entry.decoder_gold_input:
                        #if self.word_enc or self.external_embbeddins:
                        #    comb = dy.concatenate([self.clookup[gold], entry.word_enc])
                        #else:
                        #    comb = dy.concatenate([self.clookup[gold]])
                        comb = dy.concatenate([self.clookup[gold]])
                        decoder_state = decoder_state.add_input(comb)
                        p = self.softmax(decoder_state.output())
                        probs.append(p)

                    losses += [-dy.log(dy.pick(p, o)) for p, o in zip(probs, entry.decoder_gold_output)]

                total_losses = dy.esum(losses)
                cur_loss = total_losses.scalar_value()
                total += cur_loss
                total_losses.backward()
                self.trainer.update()

                if iSentence != 0 and iSentence % 500 == 0:
                    print "Processing sentence number: %d" % iSentence, ", Time: %.2f" % (time.time() - start) , " Loss:" + str(total / (iSentence + 1))
                    start = time.time()
