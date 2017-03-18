#!/usr/bin/env python3
# Mostly written in 2008 for a class; updated a little in 2017
# By Jon Dehdari.
# Usage: python langid.py [n ["input string"]]
#        where 'n' represents n-gram order (eg. bigram, trigram, etc.)  Default: 2
#        If no input string is given, it uses cross-validation to test itself
# License: GPLv.3 (see www.fsf.org)
# TODO: use command-line arg parser; lint; pep8; rewrite with lstm

from __future__ import print_function
import sys, random, decimal
from nltk.corpus import udhr2
from nltk import probability
import nltk
nltk.download('udhr2')
    
### If no command-line arg is given, just use n = 2
try:
    n = int(sys.argv[1])
except:
    n = 2

try:
    user_data = sys.argv[2]
except:
    user_data = None

random.seed(598383715)

### See http://www.unhchr.ch/udhr/navigate/alpha.htm
corpus_files = udhr2.fileids()
#corpus_files = [f[:-1] for f in open("usablefiles.txt").readlines()]


### Build ngrams for input text
def ngramize(input, n):
    ngrams = []
    for char in range(len(input) - n + 1):
        if input[char:char+n]:
            ngrams.append(input[char:char+n])
    #print(ngrams[0:9], "...")
    return ngrams

ngrams  = {}
laplace = {}
tests   = {}

sys.stderr.write("\nTraining...\n")

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
    ngrams[lang] = ngramize(text,n)

### Randomly remove 10% from the set of ngrams for each language, for testing
    randstart = random.randint(0, len(ngrams[lang]) - len(ngrams[lang])//20 )
    tests[lang] = []
    for i in range(0,len(ngrams[lang])//20, n):
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


### Test

### Get sum of probabilities for ngrams of test data
def getprobs(input_ngrams):
    sumprobs = {}
    for i in ngrams["test"]:
        for lang in corpus_files:
            try:
                sumprobs[lang] += probability.LaplaceProbDist.logprob(laplace[lang],i)
                #sumprobs[lang] += probability.LidstoneProbDist.logprob(laplace[lang],i)
            except:
                sumprobs[lang] = 0
    return sumprobs


sys.stderr.write("Testing...\n")

### Use command-line argument as test data, if given.  Otherwise use testing sections
if user_data:
    ngrams["test"] = ngramize(user_data,n)

    probs = getprobs(ngrams["test"])

    probssort = [ (value, key) for key, value in probs.items() ]
    probssort.sort()
    probssort.reverse()

    print(probssort[0:3])


else:
    correct = 0
    probs = {}
    print("tests:", tests)
    for testlang in corpus_files:
        print('testlang is', testlang)
        ngrams["test"] = tests[testlang]
        probs = getprobs(ngrams["test"])


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
