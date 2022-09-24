import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import load

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

#data 3 (biggest)
data3_train = pd.read_csv('news\\Fake News Detection Dataset\\train.csv')
data3_test = pd.read_csv('news\\Fake News Detection Dataset\\test.csv')
data3=pd.concat([data3_train, data3_test], axis=0)

data3['text']=data3['text;label'].apply(lambda x: x.split(';')[0])
data3['label']=data3['text;label'].apply(lambda x: x.split(';')[1]).astype(int)
data3.drop('text;label', axis=1, inplace=True)

data3=pd.concat([data3, data2], axis=0)
data3 = data3.sample(frac=1).reset_index(drop=True)

#train-test split
x_train, x_test, y_train, y_test =train_test_split(data3['text'],data3['label'], test_size=0.2, random_state=7)
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
