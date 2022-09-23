# This is a sample Python script.
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from joblib import dump, load

#data

real=pd.read_csv('news\\True.csv')
real['label']= 1
fake=pd.read_csv('news\\Fake.csv')
fake['label']= 0

data=pd.concat([fake, real], axis=0)
data = data.sample(frac=1).reset_index(drop=True)
data['text']= data['title']+' '+data['text']
data.drop(['title','subject','date'], axis=1, inplace=True)


#train-test split
x_train, x_test, y_train, y_test =train_test_split(data['text'],data['label'], test_size=0.2, random_state=7)
#tạo TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#transform train với test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)
#lưu vectorizer
#dump(tfidf_vectorizer,'tfidf.joblib')

#tạo model
model = PassiveAggressiveClassifier(max_iter=100)
model.fit(tfidf_train,y_train)

#dự đoán
y_pred=model.predict(tfidf_test)
print(y_pred)
print(classification_report(y_test,y_pred))
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#lưu model
#dump(model,'model.joblib')
