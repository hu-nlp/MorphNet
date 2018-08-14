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
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=64)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    # parser.add_option("--lr", type="float", dest="learning_rate", default=None)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=512)
    parser.add_option("--droputrate", type="float", dest="dropout_rate", default=0.3)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)

    (options, args) = parser.parse_args()

    print("Training file: " + options.conll_train)
    if options.conll_dev != "N/A":
        print("Development file: " + options.conll_dev)

    highestScore = 0.0
    eId = 0

    if os.path.isfile(os.path.join(options.output, options.params)) and \
            os.path.isfile(os.path.join(options.output, os.path.basename(options.model))):

        print 'Found a previous saved model => Loading this model'
        with open(os.path.join(options.output, options.params), 'r') as paramsfp:
            c2i, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = None
        parser = learner.Learner(c2i, stored_opt)
        parser.load(os.path.join(options.output, os.path.basename(options.model)))
        parser.trainer.restart()
        if options.conll_dev != "N/A":
            devPredSents = parser.predict(options.conll_dev)

            count = 0

            for idSent, devSent in enumerate(devPredSents):
                conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]

                for entry in conll_devSent:
                    if entry.id <= 0:
                        continue
                    count += 1

    else:
        print 'Extracting vocabulary'
        c2i = utils.vocab(options.conll_train)

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((c2i, options), paramsfp)
        parser = learner.Learner(c2i, options)

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
            parser.Save(os.path.join(options.output, os.path.basename(options.model)))

        else:
            devPredSents = parser.predict(options.conll_dev)

            count = 0
            correct = 0
            for idSent, devSent in enumerate(devPredSents):
                conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]

                for entry in conll_devSent:
                    if entry.id <= 0:
                        continue
                    if len(entry.predicted_sequence) == len(entry.decoder_input):
                        all_equal = True
                        for g,p in zip(entry.decoder_input, entry.predicted_sequence):
                            if g != p:
                                all_equal = False
                        if all_equal:
                            correct += 1
                    count += 1
            print "---\nAccuracy:\t%.2f" % (float(correct) * 100 / count)
            score = float(correct) * 100 / count
            if score >= highestScore:
                parser.save(os.path.join(options.output, os.path.basename(options.model)))
                highestScore = score
                eId = epoch + 1

            print "Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId)


