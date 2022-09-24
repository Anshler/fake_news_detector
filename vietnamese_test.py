import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score
from underthesea import word_tokenize
from joblib import dump

data_v = pd.concat([pd.read_csv('news\\vietnamese\\CSV\\vn_news_226_tlfr.csv'), pd.read_csv('news\\vietnamese\\CSV\\vn_news_223_tdlfr.csv').drop(['domain'], axis=1, inplace=True)])
data_v['label']= data_v['label'].replace([1,0],[0,1])
data_v = data_v.sample(frac=1).reset_index(drop=True)

stop_word = open('news/vietnamese/viet_stop_word.txt', 'r', encoding="utf-8").read().split('\n').extend(string.punctuation)
def viet_tokenizer(text):
    text = word_tokenize(str(text))
    return text

x_train, x_test, y_train, y_test =train_test_split(data_v['text'],data_v['label'], test_size=0.2, random_state=7)
#tạo TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(tokenizer=viet_tokenizer, stop_words=stop_word, max_df=0.7)
#transform train với test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)
#lưu vectorizer
#dump(tfidf_vectorizer,'tfidf_viet.joblib')

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
#dump(model, 'model_viet.joblib')