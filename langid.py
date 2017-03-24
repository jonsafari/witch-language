#!/usr/bin/env python3
# Mostly written in 2008; updated in 2017
# By Jon Dehdari.
# License: GPLv.3 (see www.fsf.org)
# TODO: reorganize train(); look at diff bet. probs after dict refactoring; rewrite with lstm

""" Simple language identification for 380 languages. """

from __future__ import print_function
import os
import sys
import re
import math
import random
import argparse
try:
    import cPickle as pickle
except:
    import pickle
from nltk.corpus import udhr2
from nltk import probability

class Model(dict):
    """ Simple model containing ngrams and smoothed statistical probs.
    Also allows for easy serialization.
    """
    def __init__(self):
        dict.__init__(self)
        self.ngrams = {}
        self.smoothed = {}
        self.tests = {}
        self.stats = {}
        self.deleted_langs = []

    def __repr__(self):
        return "%i ngrams; %i smoothed; %i tests; %i stats; %i deleted langs" % (len(self.ngrams), len(self.smoothed), len(self.tests), len(self.stats), len(self.deleted_langs))


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


def char_freqs(text, n_order, prune=False):
    """ Build ngrams for input text. """
    ngrams = {}
    for char in range(len(text) - n_order + 1):
        substring = text[char:char+n_order]
        if substring:
            if substring in ngrams:
                ngrams[substring] += 1
            else:
                ngrams[substring] = 1
    return ngrams


def train(cmd_args, corpus_files, model):
    """ Trains statistical model. """
    for lang in corpus_files:

        text = udhr2.raw(lang)
        #print("lang:", lang, "; length:", len(text))
        # Replace multiple whitespaces (including ' ', '\n', '\t') with just one ' '
        text = re.sub(r'\s+', ' ', text)

        # Skip empty files, like nku.txt
        if len(text) < 1000:
            #print("skipping pathological file", lang)
            model.deleted_langs.append(lang)
            continue

        model.ngrams[lang] = {}
        model.smoothed[lang] = []

        if cmd_args.cross_valid:
            # Remove the first 100 characters to go to the test set
            model.tests[lang] = text[:cmd_args.test_len]
            text = text[cmd_args.test_len:]

        # Build ngrams for each language in training
        model.ngrams[lang] = char_freqs(text, cmd_args.n_order)

        model.smoothed[lang] = probability.LaplaceProbDist(probability.FreqDist(model.ngrams[lang]))


def get_test_probs(cmd_args, ngrams_test, corpus_files, model):
    """ Get sum of probabilities for ngrams of test data. """
    # Initialize probs
    sumprobs = {}
    for lang in corpus_files:
        sumprobs[lang] = 0.0

    for ngram in ngrams_test:
        for lang in corpus_files:
            sumprobs[lang] += ngrams_test[ngram] * probability.LaplaceProbDist.logprob(model.smoothed[lang], ngram)

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


def test_input(cmd_args, user_data, corpus_files, model):
    """ Use command-line argument as test data. """
    ngrams_test = char_freqs(user_data, cmd_args.n_order)

    probs = get_test_probs(cmd_args, ngrams_test, corpus_files, model)

    probssort = [(value, key) for key, value in probs.items()]
    probssort.sort()
    probssort.reverse()
    return probssort


def test_all(cmd_args, corpus_files, model):
    """ Cross-validate data. """
    correct = 0
    probs = {}
    #print("tests:", model.tests)
    for testlang in corpus_files:
        ngrams_test = char_freqs(model.tests[testlang], cmd_args.n_order)
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

def main():
    """ Identifies language from STDIN. """
    parser = argparse.ArgumentParser(
        description='Easy massively multilingual language identification')
    parser.add_argument('-n', '--n_order', type=int, default=3,
                        help='Specify n-gram order (default: %(default)i)')
    parser.add_argument('--cross_valid', action="store_true",
                        help='Test all languages with cross-validation')
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
    
    
    iso_codes, _ = parse_lang_codes(iso_codes_filename)
    model_filename = create_model_filename(cmd_args)
    
    model = Model()
    try:
        model = pickle.load(open(model_filename, "rb"))
        print("Loading model: %s" % model_filename, file=sys.stderr)
    except:
        print("Existing model not found.  Training...", file=sys.stderr)
        model.stats = parse_lang_stats(lang_stats_filename)
        train(cmd_args, corpus_files, model)
        pickle.dump(model, open(model_filename, "wb"))
    
    # Remove langs having empty or tiny files
    for lang in model.deleted_langs:
        corpus_files.remove(lang)
    print("Using %i languages" % len(corpus_files), file=sys.stderr)
    
    if cmd_args.cross_valid:
        test_all(cmd_args, corpus_files, model)
    else:
        user_data = sys.stdin.read()
        probssort = test_input(cmd_args, user_data, corpus_files, model)
        print("\n    Top %i Guesses:" % cmd_args.top, file=sys.stderr)
        format_lang_guesses(probssort, cmd_args.top, iso_codes)

if __name__ == '__main__':
    main()
