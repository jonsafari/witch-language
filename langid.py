#!/usr/bin/env python3
# Mostly written in 2008; updated in 2017
# By Jon Dehdari.
# License: GPLv.3 (see www.fsf.org)
# TODO: reorganize train() and test(); restore cross-validation functionality; save model file; rewrite with lstm

""" Simple language identification for 380 languages. """

from __future__ import print_function
import sys
import random
import argparse
from nltk.corpus import udhr2
from nltk import probability

ngrams = {}
tests = {}

class Model(dict):
    """ Simple model containing ngrams and smoothed statistical probs.
    Also allows for easy serialization.
    """
    def __init__(self):
        dict.__init__(self)
        #self.ngrams = {}
        self.smoothed = {}
        #self.tests = {}



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


def ngramize(text, n_order):
    """ Build ngrams for input text. """
    ngrams = []
    for char in range(len(text) - n_order + 1):
        if text[char:char+n_order]:
            ngrams.append(text[char:char+n_order])
    #print(ngrams[0:9], "...")
    return ngrams


def train(cmd_args, corpus_files, model):
    """ Trains statistical model. """
    del_list = []
    for lang in corpus_files:

        text = udhr2.raw(lang)
        #print("lang:", lang, "; length:", len(text))

        # Skip empty files, like nku.txt
        if len(text) < 1000:
            #print("skipping pathological file", lang)
            del_list.append(lang)
            continue

        ngrams[lang] = []
        model.smoothed[lang] = []

        # Build ngrams for each language in training
        ngrams[lang] = ngramize(text, cmd_args.n_order)

        if cmd_args.testall:
            # Randomly remove 10% from the set of ngrams for each language, for testing
            randstart = random.randint(0, len(ngrams[lang]) - len(ngrams[lang]) // 20)
            tests[lang] = []
            for _ in range(0, len(ngrams[lang]) // 20, cmd_args.n_order):
                tests[lang] += [ngrams[lang].pop(randstart)]
            #print(tests[lang])


        # Build model based on ngrams for each language in training
        model.smoothed[lang] = probability.LaplaceProbDist(probability.FreqDist(ngrams[lang]))
        #model.smoothed[lang] = probability.LidstoneProbDist(probability.FreqDist(ngrams[lang]),0.50)
        #model.smoothed[lang] = probability.ELEProbDist(probability.FreqDist(ngrams[lang]))

        #model.smoothed[lang] = probability.MLEProbDist(probability.FreqDist(ngrams[lang]))
        #model.smoothed[lang] = probability.WittenBellProbDist(probability.FreqDist(ngrams[lang]))
        #model.smoothed[lang] = probability.UniformProbDist(probability.FreqDist(ngrams[lang]))


    # Remove langs having empty or tiny files
    for lang in del_list:
        corpus_files.remove(lang)
    print("Trained on %i languages" % len(corpus_files), file=sys.stderr)


def get_test_probs(ngrams_test, corpus_files, model):
    """ Get sum of probabilities for ngrams of test data. """
    sumprobs = {}
    for i in ngrams_test:
        for lang in corpus_files:
            try:
                sumprobs[lang] += probability.LaplaceProbDist.logprob(model.smoothed[lang], i)
                #sumprobs[lang] += probability.LidstoneProbDist.logprob(model.smoothed[lang], i)
            except:
                sumprobs[lang] = 0
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


def test(cmd_args, user_data, corpus_files, iso_codes, model):
    """ Use command-line argument as test data, if given.  Otherwise use testing sections. """
    if user_data:
        ngrams_test = ngramize(user_data, cmd_args.n_order)

        probs = get_test_probs(ngrams_test, corpus_files, model)

        probssort = [(value, key) for key, value in probs.items()]
        probssort.sort()
        probssort.reverse()

        max_guesses = cmd_args.top
        print("\n    Top %i Guesses:" % max_guesses, file=sys.stderr)
        format_lang_guesses(probssort, max_guesses, iso_codes)


    else:
        correct = 0
        probs = {}
        print("tests:", tests)
        for testlang in corpus_files:
            print('testlang is', testlang)
            ngrams_test = tests[testlang]
            probs = get_test_probs(ngrams_test, corpus_files)

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

        print("\nCorrect: At least", str(correct) + "/" + str(len(corpus_files)), "=")
        print(str((100.0*correct) / len(corpus_files))[0:6] + "%")


def main():
    """ Identifies language from STDIN. """
    #nltk.download('udhr2')
    iso_codes_filename = 'lang_codes_iso-639-3.tsv'
    corpus_files = udhr2.fileids()

    parser = argparse.ArgumentParser(
        description='Massively multilingual lightweight language identification')
    parser.add_argument('--n_order', type=int, default=2,
                        help='Specify n-gram order (default: %(default)i)')
    parser.add_argument('--testall', type=bool, default=False,
                        help='Test all languages with cross-validation')
    parser.add_argument('--top', type=int, default=10,
                        help='Show top number of guesses (default: %(default)i)')
    cmd_args = parser.parse_args()

    if cmd_args.testall:
        user_data = None
    else:
        user_data = sys.stdin.read()

    random.seed(598383715)

    iso_codes, _ = parse_lang_codes(iso_codes_filename)

    print("Training...", file=sys.stderr)
    model = Model()
    train(cmd_args, corpus_files, model)

    test(cmd_args, user_data, corpus_files, iso_codes, model)

if __name__ == '__main__':
    main()
