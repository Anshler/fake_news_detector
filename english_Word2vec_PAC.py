import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import nltk
import re
import string
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
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

    #data 2 (big)
data2_train = pd.read_csv('news\\Fake or Real News Dataset\\train.csv')
data2_test = pd.read_csv('news\\Fake or Real News Dataset\\test.csv')
data2=pd.concat([data2_train, data2_test], axis=0)

data2['text']=data2['text;label'].apply(lambda x: x.split(';')[0])
data2['label']=data2['text;label'].apply(lambda x: x.split(';')[1]).astype(int)
data2.drop('text;label', axis=1, inplace=True)

data2=pd.concat([data, data2], axis=0)
data2 = data2.sample(frac=1).reset_index(drop=True)

    #data 3 (bigger)
data3_train = pd.read_csv('news\\Fake News Detection Dataset\\train.csv')
data3_test = pd.read_csv('news\\Fake News Detection Dataset\\test.csv')
data3=pd.concat([data3_train, data3_test], axis=0)

data3['text']=data3['text;label'].apply(lambda x: x.split(';')[0])
data3['label']=data3['text;label'].apply(lambda x: x.split(';')[1]).astype(int)
data3.drop('text;label', axis=1, inplace=True)

data3=pd.concat([data3, data2], axis=0)
data3 = data3.sample(frac=1).reset_index(drop=True)

    #data 4 (even bigger)
data4 =pd.read_csv('news\\More_news\\train.csv')
data4['text'] = data4['title']+' '+data4['text']
data4['text'] = data4['text'].astype(str)
data4['label']=data4['label'].astype(int)
data4['label']= data4['label'].replace([1,0],[0,1])
data4.drop(['title','author','id'], axis=1, inplace=True)

data4=pd.concat([data3, data4], axis=0)
data4 = data4.sample(frac=1).reset_index(drop=True)

    #data 5 (biggest)
data5_real=pd.read_csv('news\\cnn\\clean_true_data.csv', encoding = "ISO-8859-1")
data5_real['label']= 1
data5_fake=pd.read_csv('news\\cnn\\clean_fake_data.csv', encoding = "ISO-8859-1")
data5_fake['label']=0

def removeURL(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.replace('» 100percentfedUp.com', '')
    text = text.replace('- ABC News', '')
    text = text.replace('- CNNPolitics.com', '')
    text = text.replace('- CNN.com', '')
    text = text.replace('[^A-Za-z0-9\s]', '')
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
data5 = data5.drop_duplicates().sample(frac=1).reset_index(drop=True)
print(data5.shape[0])

    #Vectorizing corpus
stop_words = set(nltk.corpus.stopwords.words("english"))
regex_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

global index
index = 0
def pre_process(text): # xử lý text, tokenize, bỏ stop word
    tokens = regex_tokenizer.tokenize(text)
    filtered_words = [w.lower().strip() for w in tokens if w.lower().strip() not in stop_words and w.isalpha()]
    global index
    print(index)
    index+=1
    return filtered_words

x = data5['text'][:15000].apply(lambda x: pre_process(x))
y= data5['label'][:15000].values
#w2v_model = Word2Vec(sentences=x, vector_size=1, window=5, min_count=1)
    #save model
#w2v_model.save('models/w2v.model')
    #load model
w2v_model = Word2Vec.load('models/w2v.model')

    #Tạo vocabulary, mỗi văn bản sẽ biến thành 1 vector kích thước 250k
    #Những từ trong văn bản đó sẽ có trọng số W2V, ko có thì sẽ = 0
def write_vocab(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    vocab_size = len(tokenizer.word_index) + 1 # +1 vì index của vocab bắt đầu từ 1
    vocab = tokenizer.word_index
    f = open("news/vocabulary/vocabulary.txt", "a", encoding='utf-8')
    for key, value in vocab.items():
        f.write('%s:%s\n' % (key, value))
    f.write('%d\n' % vocab_size)
    f.close()
    print("size of vocab: ", vocab_size)
    print("finish writing vocab")

def load_vocab():
    f = open('news/vocabulary/vocabulary.txt', 'r', encoding='utf-8')
    vocab ={}
    for line in f:
        try:
            (key, val) = line.split(':')
            vocab[key] = int(val)
        except:
            vocab_size = int(line)
    print("size of vocab: ", vocab_size)
    print("finish reading vocab")
    return vocab_size,vocab

#write_vocab(x)
vocab_size, vocab = load_vocab()

index = 0
def post_process(text): # biến từng token thành giá trị Word2vec
    vector = np.zeros((50000,), dtype=float)
    for word, i in vocab.items():
        try:
            if word in text:
                vector[i] = float(w2v_model.wv[word])
        except:
            pass
        if i == 49999:
            break
    global index
    print(index)
    index+=1
    return vector

x = list(x.apply(lambda x: post_process(x)))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

x_train, x_test, y_train, y_test = train_test_split(x,y)
model = PassiveAggressiveClassifier(max_iter=100)
model.fit(x_train, y_train)

    #save model
#dump(model,'models/model_w2v.joblib')
    #prediction
y_pred=model.predict(x_test)
score=accuracy_score(y_pred,y_test)
print(f'Accuracy: {round(score*100,2)}%')


