# coding=utf-8
from optparse import OptionParser
import pickle, utils, learner, os, os.path, time

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Path to annotated CONLL train file", metavar="FILE",
                      default="N/A")
    parser.add_option("--dev", dest="conll_dev", help="Path to annotated CONLL dev file", metavar="FILE", default="N/A")
    parser.add_option("--test", dest="conll_test", help="Path to CONLL test file", metavar="FILE", default="N/A")
    parser.add_option("--output", dest="conll_test_output", help="File name for predicted output", metavar="FILE",
                      default="N/A")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="model.params")
    parser.add_option("--prevectors", dest="external_embedding", help="Pre-trained vector embeddings", metavar="FILE")
    parser.add_option("--prevectype", dest="external_embedding_type", help="Pre-trained vector embeddings type", default=None)
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=64)
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=300)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    # parser.add_option("--lr", type="float", dest="learning_rate", default=0.0001)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--elstmdims", type="int", dest="enc_lstm_dims", default=512)
    parser.add_option("--dlstmdims", type="int", dest="dec_lstm_dims", default=256)
    parser.add_option("--droputrate", type="float", dest="dropout_rate", default=0.3)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
    parser.add_option("--model-type", type="int", dest="model_type", default=0) # 0 none -1  simple char rnn - 2 simple char bilstm - 3 simple prevec



    (options, args) = parser.parse_args()

    print("Training file: " + options.conll_train)
    if options.conll_dev != "N/A":
        print("Development file: " + options.conll_dev)

    highestScore = 0.0
    eId = 0

    print 'Extracting vocabulary'
    c2i, w2i, features = utils.vocab(options.conll_train)

    parser = learner.Learner(c2i, w2i, features, options)

    highestScore = 0.0
    eId = 0
    for epoch in xrange(options.epochs):
        print '\n-----------------\nStarting epoch', epoch + 1

        if epoch % 10 == 0:
            if epoch == 0:
                parser.trainer.restart(learning_rate=0.001)
            elif epoch == 10:
                parser.trainer.restart(learning_rate=0.0005)
            else:
                parser.trainer.restart(learning_rate=0.00025)

        parser.train(options.conll_train)

        if options.conll_dev == "N/A":
            parser.save(os.path.join(options.output, os.path.basename(options.model)))

        else:
            devPredSents = parser.predict(options.conll_dev)

            count = 0
            correct = 0
            for idSent, devSent in enumerate(devPredSents):
                conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]
                for entry in conll_devSent:
                    if entry.id <= 0:
                        continue
                    correct_out = 0.0

                    for gold in entry.decoder_gold_output:
                        if gold in entry.predicted_sequence:
                            correct_out += 1
                    correct += correct_out/len(entry.decoder_gold_output)
                    count += 1
            print "---\nAccuracy:\t%.2f" % (float(correct) * 100 / count)

            score = float(correct) * 100 / count
            if score >= highestScore:
                # parser.save(os.path.join(options.output, os.path.basename(options.model)))
                highestScore = score
                eId = epoch + 1

            print "Highest: %.2f at epoch %d" % (highestScore, eId)
