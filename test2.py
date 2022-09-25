import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

data4 =pd.read_csv('news\\More_news\\train.csv')
data4['text'] = data4['title']+' '+data4['text']
data4['label']=data4['label'].astype(int)
data4['label']= data4['label'].replace([1,0],[0,1])
data4.drop(['title','author','id'], axis=1, inplace=True)
print(data4.head())

