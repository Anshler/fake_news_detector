import pandas as pd
import numpy as np
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

stop_word = open('news/vietnamese/viet_stop_word.txt', 'r', encoding="utf-8").read().split('\n')
stop_word.extend(string.punctuation)
stop_word.extend(['…','”','–','“','. ảnh'])

def viet_tokenizer(text):
    text = word_tokenize(str(text))
    text = [word for word in text if word.lower() not in stop_word]
    return text

x_train, x_test, y_train, y_test =train_test_split(data_v['text'].apply(lambda x: x.lower()),data_v['label'], test_size=0.2, random_state=7)
#tạo TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(tokenizer=viet_tokenizer, max_df=0.7)
#transform train với test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)
#lưu vectorizer
#dump(tfidf_vectorizer,'tfidf_viet.joblib')

#tạo model
model = AggressivPassieClassifier(max_iter=100)
model.fit(tfidf_train,y_train)

#dự đoán
y_pred=model.predict(tfidf_test)
print(y_pred)
print(classification_report(y_test,y_pred))
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


#lưu model
#dump(model, 'model_viet.joblib')

# Sort non-zero weights
weights_nonzero = model.coef_[model.coef_!=0]
feature_sorter_nonzero = np.argsort(weights_nonzero)
weights_nonzero_sorted =weights_nonzero[feature_sorter_nonzero]
# Sort features by their associated weights
tokens = tfidf_vectorizer.get_feature_names_out()
tokens_nonzero = np.array(tokens)[model.coef_[0]!=0]
tokens_nonzero_sorted = np.array(tokens_nonzero)[feature_sorter_nonzero]

num_tokens = 20
fake_indicator_tokens = tokens_nonzero_sorted[:num_tokens]
real_indicator_tokens = np.flip(tokens_nonzero_sorted[-num_tokens:])

fake_indicator = pd.DataFrame({
    'Token': fake_indicator_tokens,
    'Weight': weights_nonzero_sorted[:num_tokens]
})

real_indicator = pd.DataFrame({
    'Token': real_indicator_tokens,
    'Weight': np.flip(weights_nonzero_sorted[-num_tokens:])
})

print('The top {} tokens likely to appear in fake news were the following: \n'.format(num_tokens))
print(fake_indicator)

print('\n\n...and the top {} tokens likely to appear in real news were the following: \n'.format(num_tokens))
print(real_indicator)

fake_contain_fake = data_v[data_v['label']==0].text.loc[[np.any([token in body for token in fake_indicator.Token])
                                for body in data_v[data_v['label']==0].text.str.lower()]]
real_contain_real = data_v[data_v['label']==1].text.loc[[np.any([token in body for token in real_indicator.Token])
                                for body in data_v[data_v['label']==1].text.str.lower()]]

print('Articles that contained any of the matching indicator tokens:\n')

print('FAKE: {} out of {} ({:.2f}%)'
      .format(len(fake_contain_fake), len(data_v[data_v['label']==0]), len(fake_contain_fake)/len(data_v[data_v['label']==0]) * 100))
print(fake_contain_fake)

print('\nREAL: {} out of {} ({:.2f}%)'
      .format(len(real_contain_real), len(data_v[data_v['label']==1]), len(real_contain_real)/len(data_v[data_v['label']==1]) * 100))
print(real_contain_real)
