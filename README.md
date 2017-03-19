# Witch-language
Massively multilingual, lightweight language identification.

## Prerequisites
Install NLTK:
```Bash
pip install --user nltk
```

Download the UDHR2 dataset within Python:
```Python
import nltk
nltk.download('udhr2')
```

## Usage
```Bash
echo "Fufú kele madya ya bilûmbu nyonso na Afelika ya Kati." | python3 langid.py
```

You can add the `--help` command-line flag to see more options.
