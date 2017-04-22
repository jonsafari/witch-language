#!/usr/bin/env python3
# By Jon Dehdari, 2017
# License: GPLv.3 (see www.fsf.org)

""" Language identification for 380 languages. """

from __future__ import print_function
import os
import sys
import re
import math
import random
import argparse
import collections
try:
    import cPickle as pickle
except:
    import pickle
from nltk.corpus import udhr2
import numpy as np
import mxnet as mx
import logging
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s %(message)s')

class Model(dict):
    """ Simple model containing ngrams and smoothed statistical probs.
    Also allows for easy serialization.
    """
    def __init__(self, vocab_size):
        dict.__init__(self)
        self.tests = {}
        self.stats = {}
        self.char2int = {'☃':0} # <unk> == '☃' (snowman)
        self.int2char = {0:'☃'}
        self.lang2int = {}
        self.int2lang = {}
        self.deleted_langs = []

        # Build model, using MXNet's Python Symbolic API
        self.net = mx.sym.Variable('data')
        self.net = mx.sym.Embedding(self.net, name='embedding', input_dim=vocab_size, output_dim=20)
        #self.net = mx.sym.RNN(self.net, name='rnn', mode='lstm', bidirectional=True, state_size=20, num_layers=2)
        self.net = mx.sym.FullyConnected(self.net, name='ff1', num_hidden=50)
        self.net = mx.sym.SoftmaxOutput(self.net, name='softmax')

    def __repr__(self):
        return "%i tests; %i stats; %i deleted langs" % (len(self.tests), len(self.stats), len(self.deleted_langs))


def parse_lang_codes(iso_codes_filename):
    """ Bijective mapping between ISO 639-3 language codes and their (macro-)language name.
    Eg. iso_codes = {'eng':'English', ... }; iso_codes_rev = {'English': 'eng', ...}
    """
    iso_codes = {}
    iso_codes_rev = {}
    with open(iso_codes_filename) as code_file:
        for line in code_file:
            iso_code, lang = line.rstrip().split('\t')
            iso_codes[iso_code] = lang
            iso_codes_rev[lang] = iso_code

    return (iso_codes, iso_codes_rev)


def parse_lang_stats(lang_stats_filename):
    """ Builds dict between ISO 639-3 language codes and their approximate
    number of speakers, based on Wikipedia.
    """
    iso_codes = {}
    with open(lang_stats_filename) as stats_file:
        for line in stats_file:
            iso_code, population = line.rstrip().split('\t')
            iso_codes[iso_code] = int(population)
    return iso_codes

def train(cmd_args, corpus_files, model):
    """ Trains statistical model. """

    xs, ys = generate_train_set(cmd_args, model, corpus_files)
    #xs = [[x] for x in xs] # wrap each item in another list, which is what mxnet wants
    print("head(xs)=", xs[:10])
    print("head(ys)=", ys[:10])
    data_iter = mx.io.NDArrayIter(np.array(xs), label=np.array(ys), shuffle=True, batch_size=cmd_args.batch)
    print("Converted training text to NDArrayIter; %i items" % len(xs))
    print(" data=%s;\nlabel=%s" % (data_iter.provide_data, data_iter.provide_label))
    mxmodule = mx.mod.Module(symbol=model.net, context=mx.cpu(0), data_names=['data'])
    print("net shape:", model.net.infer_shape())
    mxmodule.fit(data_iter, num_epoch=cmd_args.epochs,
              #eval_metric='acc',
              #initializer=mx.init.Xavier(factor_type='in'),
              #optimizer_params={'learning_rate':1.0, 'momentum':0.9}
              )
    return mxmodule

def generate_train_set(cmd_args, model, corpus_files):
    """ Returns (ngrams, lang_labels) """
    ngrams = []
    lang_labels = []

    for lang in corpus_files:
        text = udhr2.raw(lang)
        # Replace multiple whitespaces (including ' ', '\n', '\t') with just one ' '
        text = re.sub(r'\s+', ' ', text)

        # Skip empty files, like nku.txt
        if len(text) < 17000:
            #print("skipping pathological file", lang)
            model.deleted_langs.append(lang)
            continue

        # Replace letters with integers
        int_text = text2ints(model, text)

        if cmd_args.cross_valid:
            # Remove the first 100 characters to go to the test set
            model.tests[lang] = int_text[:cmd_args.test_len]
            int_text = int_text[cmd_args.test_len:]

        # Train model
        lang_ngrams = get_instances(cmd_args, int_text)
        ngrams += lang_ngrams
        lang_labels += [model.lang2int[lang]] * len(lang_ngrams)

    return (ngrams, lang_labels)

def text2ints(model, text):
    ints = [0] * len(text)
    for char_pos in range(len(text)):
        char = text[char_pos]
        if char not in model.char2int:
            model.char2int[char] = len(model.char2int) + 1
        ints[char_pos] = model.char2int[char]
    print("vocab size=", len(model.char2int))
    return ints
            

def get_instances(cmd_args, text):
    ngrams = []

    for char in range(len(text) - cmd_args.n_order + 1):
        substring = text[char:char + cmd_args.n_order]
        # This could be stored more efficiently as a dict, but more model assumptions would be needed
        ngrams.append(substring)
    return ngrams


def get_test_probs(cmd_args, ngrams_test, corpus_files, model, mxmodule):
    """ Get sum of probabilities for ngrams of test data. """
    # Initialize probs
    sumprobs = {}
    for lang in corpus_files:
        sumprobs[lang] = 0.0

    # Count ngrams
    ngrams_test_count = collections.Counter(ngrams_test)

    for ngram in ngrams_test:
        #ngram_int = np.array([model.char2int[char] for char in ngram])
        ngram_int = np.array([model.char2int[char] for char in ngram])
        ngram_int_mx = mx.io.NDArrayIter(ngram_int)
        print("ngram=", ngram)
        print("ngram_int=", ngram_int)
        print("ngram_int_mx=", ngram_int_mx)
        print("ngrams_test=", ngrams_test)
        print("ngrams_test_count[ngram]=", ngrams_test_count[ngram])
        for lang in corpus_files:
            #sumprobs[lang] += ngrams_test[ngram] * probability.LaplaceProbDist.logprob(model.smoothed[lang], ngram)
            #print("mxpredict=", mxmodule.score(ngram_int_mx, mx.metric.CrossEntropy()))
            #sumprobs[lang] += ngrams_test_count[ngram] * mxmodule.predict(eval_data=ngram_int)
            ...

    # The population prior is mostly useful for really small test snippets
    if not cmd_args.no_prior:
        for lang in corpus_files:
            # Strip trailing .txt, and check if it's in the population statistics dict
            lang_prefix = lang[:-4]
            if lang_prefix in model.stats:
                # Normalize population counts by approximate total number of people on earth
                sumprobs[lang] += math.log(model.stats[lang_prefix] / 8e9)
            else:
                # If language isn't in the language population statistics,
                # assume median value of all langs, which is about 500K
                sumprobs[lang] += math.log(500000 / 8e9)

    return sumprobs


def format_lang_guesses(sorted_probs, max_guesses, iso_codes):
    """ Pretty prints hypotheses. """
    for prob, lang in sorted_probs[0:max_guesses]:
        # Strip ".txt" from eng.txt, for example
        if lang.endswith(".txt"):
            lang = lang[:-4]

        # Try to resolve ISO code to language name
        iso_name = ""
        try:
            iso_name += "%s (%s)" % (iso_codes[lang], lang)
        except:
            iso_name += "%s" % lang

        print("%s: %g" % (iso_name, prob))


def test_input(cmd_args, user_data, corpus_files, model, mxmodule):
    """ Use command-line argument as test data. """
    # First deal with unknown characters
    # Take the set difference of chars in test_set and chars in model.char2int
    # Then regex-replace these unknown chars with dummy 0 char
    input_counts = collections.Counter(user_data)
    unk_chars = input_counts.keys() - model.char2int.keys()
    for char in unk_chars:
        # replace a given unknown character with fun snowman
        user_data = re.sub(char, '☃', user_data)
    print("input_counts.keys=", input_counts.keys())
    print("model.char2int.keys=", model.char2int.keys())
    print("unk_chars=", unk_chars)
    print("user_data=", user_data)


    ngrams_test = get_instances(cmd_args, user_data)

    probs = get_test_probs(cmd_args, ngrams_test, corpus_files, model, mxmodule)

    probssort = [(value, key) for key, value in probs.items()]
    probssort.sort()
    probssort.reverse()

    # Produce probabilities, if requested
    if not cmd_args.no_probs:
        # Get normalizer
        Z = sum([2**score for score, _ in probssort])
        probssort = [(((2**score)/Z), val) for (score, val) in probssort]

    return probssort


def test_all(cmd_args, corpus_files, model):
    """ Cross-validate data. """
    correct = 0
    probs = {}
    #print("tests:", model.tests)
    for testlang in corpus_files:
        ngrams_test = get_instances(model.tests[testlang], cmd_args.n_order)
        probs = get_test_probs(cmd_args, ngrams_test, corpus_files, model)

        # This sorts the languages by probability
        probssort = [(value, key) for key, value in probs.items()]
        probssort.sort()
        probssort.reverse()

        # Print results
        if testlang == probssort[0][1]:
            print(testlang, " :-)")
            correct += 1
        else:
            print(testlang, "best guess: ", probssort[0:2])

    print("\nCorrect: At least", str(correct) + "/" + str(len(corpus_files)), "= ", end='')
    print(str((100.0*correct) / len(corpus_files))[0:6] + "%")


def create_model_filename(cmd_args):
    """ Constructs the filename of the pickled model. """
    filename = "witch-lang"
    filename += "_n-%i" % cmd_args.n_order
    filename += "_cv-%i" % cmd_args.cross_valid
    filename += "_tl-%i" % cmd_args.test_len
    filename += "_py-%i" % sys.version_info.major
    filename += ".pkl"
    return filename

#def main():
""" Identifies language from STDIN. """
parser = argparse.ArgumentParser(
    description='Easy massively multilingual language identification')
parser.add_argument('-n', '--n_order', type=int, default=2,
                    help='Specify n-gram order (default: %(default)i)')
parser.add_argument('--cross_valid', action="store_true",
                    help='Test all languages with cross-validation')
parser.add_argument('--epochs', type=int, default=7,
                    help='Specify number of training epochs (default: %(default)i)')
parser.add_argument('--batch', type=int, default=32,
                    help='Specify batch size (default: %(default)i)')
parser.add_argument('--no_probs', action="store_true",
                    help="Don't output probabilities")
parser.add_argument('--test_len', type=int, default=200,
                    help='Specify cross-validation test length (default: %(default)i)')
parser.add_argument('--no_prior', action="store_true",
                    help='Disable language population prior')
parser.add_argument('--top', type=int, default=10,
                    help='Show top number of guesses (default: %(default)i)')
cmd_args = parser.parse_args()

random.seed(598383715)
basepath = os.path.dirname(os.path.realpath(__file__))
iso_codes_filename = os.path.join(basepath, 'lang_codes_iso-639-3.tsv')
lang_stats_filename = os.path.join(basepath, 'lang_stats_wp.tsv')
corpus_files = udhr2.fileids()

# Build model
model = Model(vocab_size=400)

# Populate lang_id <-> int dictionary
for corpus_file in corpus_files:
    if corpus_file not in model.lang2int:
        model.lang2int[corpus_file] = len(model.lang2int) + 1
        model.int2lang[len(model.lang2int)] = corpus_file

iso_codes, _ = parse_lang_codes(iso_codes_filename)
model_filename = create_model_filename(cmd_args)



try:
    model = pickle.load(open(model_filename, "rb"))
    print("Loading model: %s" % model_filename, file=sys.stderr)
except:
    print("Existing model not found.  Training...", file=sys.stderr)
    model.stats = parse_lang_stats(lang_stats_filename)
    mxmodule = train(cmd_args, corpus_files, model)
    pickle.dump(model, open(model_filename, "wb"))
    #mxmodel.save_params(model_filename)

# Remove langs having empty or tiny files
for lang in model.deleted_langs:
    corpus_files.remove(lang)
print("Using %i languages" % len(corpus_files), file=sys.stderr)

if cmd_args.cross_valid:
    test_all(cmd_args, corpus_files, model, mxmodule)
else:
    user_data = sys.stdin.read().rstrip()
    probssort = test_input(cmd_args, user_data, corpus_files, model, mxmodule)
    print("\n    Top %i Guesses:" % cmd_args.top, file=sys.stderr)
    format_lang_guesses(probssort, cmd_args.top, iso_codes)


#if __name__ == '__main__':
#    main()
