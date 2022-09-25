import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

#data 1

real=pd.read_csv('news\\True.csv')
real['label']= 1
fake=pd.read_csv('news\\Fake.csv')
fake['label']= 0

data=pd.concat([fake, real], axis=0)
data = data.sample(frac=1).reset_index(drop=True)
data['text']= data['title']+' '+data['text']
data.drop(['title','subject','date'], axis=1, inplace=True)

#data 2 (bigger)
data2_train = pd.read_csv('news\\Fake or Real News Dataset\\train.csv')
data2_test = pd.read_csv('news\\Fake or Real News Dataset\\test.csv')
data2=pd.concat([data2_train, data2_test], axis=0)

data2['text']=data2['text;label'].apply(lambda x: x.split(';')[0])
data2['label']=data2['text;label'].apply(lambda x: x.split(';')[1]).astype(int)
data2.drop('text;label', axis=1, inplace=True)

data2=pd.concat([data, data2], axis=0)
data2 = data2.sample(frac=1).reset_index(drop=True)

#data 3 (even bigger)
data3_train = pd.read_csv('news\\Fake News Detection Dataset\\train.csv')
data3_test = pd.read_csv('news\\Fake News Detection Dataset\\test.csv')
data3=pd.concat([data3_train, data3_test], axis=0)

data3['text']=data3['text;label'].apply(lambda x: x.split(';')[0])
data3['label']=data3['text;label'].apply(lambda x: x.split(';')[1]).astype(int)
data3.drop('text;label', axis=1, inplace=True)

data3=pd.concat([data3, data2], axis=0)
data3 = data3.sample(frac=1).reset_index(drop=True)

#data 4 (biggest)
data4 =pd.read_csv('news\\More_news\\train.csv')
data4['text'] = data4['title']+' '+data4['text']
data4['text'] = data4['text'].astype(str)
data4['label']=data4['label'].astype(int)
data4['label']= data4['label'].replace([1,0],[0,1])
data4.drop(['title','author','id'], axis=1, inplace=True)

data4=pd.concat([data3, data4], axis=0)
data4 = data4.sample(frac=1).reset_index(drop=True)

#data 5
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

data5=pd.concat([data4, data5], axis=0)
data5 = data5.sample(frac=1).reset_index(drop=True)

#train-test split
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
print(data4.head())
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred, labels=[0,1]))
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#lưu model
#dump(model,'model.joblib')
