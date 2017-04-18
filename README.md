# Witch-language
Massively multilingual, easy language identification.
It currently works on 380 languages.

## Prerequisites
Install NLTK:
```Bash
pip3 install --user nltk
```

Download the UDHR2 dataset:
```Bash
echo "import nltk; nltk.download('udhr2')" | python3
```

## Usage
```Bash
echo "Fufú kele madya ya bilûmbu nyonso na Afelika ya Kati." | python3 langid.py

    Top 10 Guesses:
Koongo (kng): -408.319
Kituba (Democratic Republic of Congo) (ktu): -408.895
Lozi (loz): -419.485
Kaonde (kqn): -421.326
Luba-Lulua (lua): -421.911
Nyamwezi (nym): -426.761
Swahili (individual language) (swh): -426.967
Kimbundu (kmb): -427.247
Lingala (lin): -427.414
Sukuma (suk): -428.922

```

You can add the `--help` command-line flag to see more options.
Using Python3 gives much better performance than Python2 for this task.
