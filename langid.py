#!/usr/bin/env python3
# Mostly written in 2008 for a class; updated a little in 2017
# By Jon Dehdari.
# Usage: python langid.py [n ["input string"]]
#        where 'n' represents n-gram order (eg. bigram, trigram, etc.)  Default: 2
#        If no input string is given, it uses cross-validation to test itself
# License: GPLv.3 (see www.fsf.org)
# TODO: reorganize; use command-line arg parser; resolve ISO ID's to names & rewrite guess output; save model file; lint; pep8; rewrite with lstm

from __future__ import print_function
import sys
import random
from nltk.corpus import udhr2
from nltk import probability
import nltk

ngrams  = {}
laplace = {}
tests   = {}


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


def ngramize(input, n_order):
    """ Build ngrams for input text. """
    ngrams = []
    for char in range(len(input) - n_order + 1):
        if input[char:char+n_order]:
            ngrams.append(input[char:char+n_order])
    #print(ngrams[0:9], "...")
    return ngrams


def train(n_order, corpus_files):
    del_list = []
    for lang in corpus_files:
        
        text = udhr2.raw(lang)
        #print("lang:", lang, "; length:", len(text))
    
        # Skip empty files, like nku.txt
        if len(text) < 500:
            print("skipping pathological file", lang)
            del_list.append(lang)
            continue
    
        ngrams[lang] = []
        laplace[lang] = []
        
    ### Build ngrams for each language in training
        ngrams[lang] = ngramize(text,n_order)
    
    ### Randomly remove 10% from the set of ngrams for each language, for testing
        randstart = random.randint(0, len(ngrams[lang]) - len(ngrams[lang])//20 )
        tests[lang] = []
        for i in range(0,len(ngrams[lang])//20, n_order):
            tests[lang] += [ngrams[lang].pop(randstart)]
        #print(tests[lang])
    
    
    ### Build model based on ngrams for each language in training
        laplace[lang] = probability.LaplaceProbDist(probability.FreqDist(ngrams[lang]))
        #laplace[lang] = probability.LidstoneProbDist(probability.FreqDist(ngrams[lang]),0.50)
        #laplace[lang] = probability.ELEProbDist(probability.FreqDist(ngrams[lang]))
    
        #laplace[lang] = probability.MLEProbDist(probability.FreqDist(ngrams[lang]))
        #laplace[lang] = probability.WittenBellProbDist(probability.FreqDist(ngrams[lang]))
        #laplace[lang] = probability.UniformProbDist(probability.FreqDist(ngrams[lang]))
    
    
    # Remove langs having empty or tiny files
    for lang in del_list:
        corpus_files.remove(lang)


def get_test_probs(input_ngrams, corpus_files):
    """ Get sum of probabilities for ngrams of test data. """
    sumprobs = {}
    for i in ngrams["test"]:
        for lang in corpus_files:
            try:
                sumprobs[lang] += probability.LaplaceProbDist.logprob(laplace[lang],i)
                #sumprobs[lang] += probability.LidstoneProbDist.logprob(laplace[lang],i)
            except:
                sumprobs[lang] = 0
    return sumprobs


def format_lang_guesses(sorted_probs, max_guesses, iso_codes):
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


def test(n_order, user_data, corpus_files, iso_codes):
    """ Use command-line argument as test data, if given.  Otherwise use testing sections. """
    if user_data:
        ngrams["test"] = ngramize(user_data,n_order)
    
        probs = get_test_probs(ngrams["test"], corpus_files)
    
        probssort = [ (value, key) for key, value in probs.items() ]
        probssort.sort()
        probssort.reverse()
    
        format_lang_guesses(probssort, 3, iso_codes)
    
    
    else:
        correct = 0
        probs = {}
        print("tests:", tests)
        for testlang in corpus_files:
            print('testlang is', testlang)
            ngrams["test"] = tests[testlang]
            probs = get_test_probs(ngrams["test"])
    
    
            """
            for lang in corpus_files:
                try:
                    if maxprob < probs[lang]:
                        maxlang = lang
                        maxprob = probs[lang]
                except:
                    maxlang = lang
                    maxprob = probs[lang]
            
                print(lang, probs[lang])
            
            print("Best guess: ", maxlang, maxprob)
            
            """
    
            ### This sorts the languages by probability
            probssort = [ (value, key) for key, value in probs.items() ]
            probssort.sort()
            probssort.reverse()
    
    
            ### Print results
        if testlang == probssort[0][1]:
            print(testlang, " :-)")
            correct += 1
        else:
                print(testlang, "best guess: ", probssort[0:2])
    
        print("\nCorrect: At least", str(correct) + "/" + str(len(corpus_files)), "=")
        ### Python's current handling of rounding floating point numbers is just plain lame.
        ### Sure, blame it on the compiler or hardware.
        ### Somehow other languages just get it right.
        print(str( (100.0*correct) / len(corpus_files) )[0:6] + "%")


def main():
    nltk.download('udhr2')
    iso_codes_filename = 'lang_codes_iso-639-3.tsv'
    corpus_files = udhr2.fileids()
        
    ### If no command-line arg is given, just use n = 2
    try:
        n_order = int(sys.argv[1])
    except:
        n_order = 2
    
    try:
        user_data = sys.argv[2]
    except:
        user_data = None
    
    random.seed(598383715)
    
    iso_codes, iso_codes_rev = parse_lang_codes(iso_codes_filename)
    
    print("\nTraining...\n", file=sys.stderr)
    train(n_order, corpus_files)

    print("Testing...\n", file=sys.stderr)
    test(n_order, user_data, corpus_files, iso_codes)

if __name__ == '__main__':
    main()
