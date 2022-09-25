import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

data5_real=pd.read_csv('news\\cnn\\clean_true_data.csv', encoding = "ISO-8859-1")
data5_real['label']= 1
data5_fake=pd.read_csv('news\\cnn\\clean_fake_data.csv', encoding = "ISO-8859-1")
data5_fake['label']=0

def removeURL(text):
    text=text.replace('» 100percentfedUp.com','')
    text=text.replace('- ABC News', '')
    text=text.replace('- CNNPolitics.com', '')
    text=text.replace('- CNN.com', '')
    return text

data5_real = data5_real[['text','title','label']]
data5_fake = data5_fake[['text','title','label']]

data5=pd.concat([data5_fake, data5_real], axis=0)
data5 = data5.sample(frac=1).reset_index(drop=True)
data5['title']=data5['title'].apply(lambda x: removeURL(x))
data5['text']= data5['title']+' '+data5['text']
data5.drop(['title'], axis=1, inplace=True)
data5['text'] = data5['text'].astype(str)
data5['label'] = data5['label'].astype(int)
print(data5.head())


x_train, x_test, y_train, y_test =train_test_split(data5['text'].apply(lambda x : x.lower()),data5['label'], test_size=0.2, random_state=7)
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
#print(y_pred)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred, labels=[0,1]))
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#lưu model
#dump(model,'model.joblib')