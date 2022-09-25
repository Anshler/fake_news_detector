import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
stop_word = open('news/vietnamese/viet_stop_word.txt', 'r', encoding="utf-8").read().split('\n')
stop_word.extend(string.punctuation)
stop_word.extend(['enb','ccc'])
print(stop_word)