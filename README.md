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
Koongo (kng): 0.519163
Kituba (Democratic Republic of Congo) (ktu): 0.479616
Lozi (loz): 0.000642021
Kaonde (kqn): 0.00050906
Nyamwezi (nym): 3.45846e-05
Luba-Lulua (lua): 2.30121e-05
Lingala (lin): 5.39564e-06
Bemba (Zambia) (bem): 2.98133e-06
Swahili (individual language) (swh): 2.55979e-06
Sukuma (suk): 6.94787e-07
```

You can add the `--help` command-line flag to see more options.
Using Python3 gives much better performance than Python2 for this task.
