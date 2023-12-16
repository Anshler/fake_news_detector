# DETECT FAKE NEWS IN ENGLISH AND VIETNAMESE

Due to the strong development of the internet in recent years, news all over the world are transmitted faster and faster through news websites, social media, blogs, etc. However, this also make it easier for Fake news to spread, which, sometimes, can lead to the outbreak of mass hysteria and negative impact on society. 

In this paper, we have we researched and built a machine learning model that can distinguish fake news from real news. Firstly, we introduce two datasets of English and Vietnamese articles. Secondly, we use different algorithms to train our model to predict on each dataset. Finally, we compare the results and draw conclusions. Specifically, we discovered that the English test using TF-IDF Vectorization with Passive-Aggressive Classifier produced the highest accuracy of ```96.12%```.

_Read our report_ [here](Fake%20news%20detection.pdf)

## Dataset
This experiment is split into English and Vietnamese. 

For the English language, we use the Kaggle dataset and other datasets from large news websites. The Kaggle dataset from 2017 contains 45000 newspapers evenly divided between real and fake, the real news come from Reuters website whereas fake news are from websites deemed unreliable by Wikipedia. Other datasets come from different sources such as CNN, BBC, FOX, etc. But similar in date and distribution of real and fake. Together with the Kaggle dataset, this collection of data contains nearly 90000 newspapers.

For the Vietnamese test we use the Vietnamese Fake News Dataset (VFND). This contains over 200 news, collected in the period from 2017 to 2019 and uses several, cross-referencing sources, classified by the community.

All data are included in this repo.

## Tutorial

For inference, run [main.py](main.py)

All other files are for training. To retrain the models, uncomment the following lines:

- In [main.py](main.py)
```python
model_viet = load('models/model_viet.joblib')
model=load('models/model.joblib')
tfidf_vectorizer_viet = load('models/tfidf_viet.joblib')
tfidf_vectorizer = load('models/tfidf.joblib')
```

- In [english_test.py](english_test.py)
```python
  # lưu vectorizer
dump(tfidf_vectorizer,'models/tfidf.joblib')
dump(count_vectorizer, 'models/count.joblib')

  # lưu model
dump(model,'models/model.joblib')
dump(model,'models/modelc.joblib')
```

- In [vietnamese_test.py](vietnamese_test.py)
```python
  #lưu vectorizer
dump(tfidf_vectorizer,'tfidf_viet.joblib')

  #lưu model
dump(model, 'models/model_viet.joblib')
```

- In [english_Word2vec_PAC.py](english_Word2vec_PAC.py)
```python
w2v_model = Word2Vec(sentences=x, vector_size=1, window=5, min_count=1)
  #save model
w2v_model.save('models/w2v.model')

  #save model
dump(model,'models/model_w2v.joblib')
```

## Explanation

[english_test.py](english_test.py): use [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) and [TFIDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). Train with [PasiveAgressiveClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)

[vietnamese_test.py](vietnamese_test.py): similar to english_test.py, but use [Underthesea](https://underthesea.readthedocs.io/en/latest/readme.html) tokenizer(for Vietnamese) before vectorization

[english_Word2vec_PAC.py](english_Word2vec_PAC.ipynb): similar to english_test.py, but use [Word2Vec](https://arxiv.org/abs/1301.3781)
