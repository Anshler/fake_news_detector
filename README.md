# DETECT FAKE NEWS IN ENGLISH AND VIETNAMESE

Due to the strong development of the internet in recent years, news all over the world are transmitted faster and faster through news websites, social media, blogs, etc. However, this also make it easier for Fake news to spread, which, sometimes, can lead to the outbreak of mass hysteria and negative impact on society. 

In this paper, we have we researched and built a machine learning model that can distinguish fake news from real news. Firstly, we introduce two datasets of English and Vietnamese articles. Secondly, we use different algorithms to train our model to predict on each dataset. Finally, we compare the results and draw conclusions. Specifically, we discovered that the English test using TF-IDF Vectorization with Passive-Aggressive Classifier produced the highest accuracy.

_Read our report_ [here](Fake%20news%20detection.pdf)

## Dataset
This experiment is split into English and Vietnamese. 

For the English language, we use the Kaggle dataset and other datasets from large news websites. The Kaggle dataset from 2017 contains 45000 newspapers evenly divided between real and fake, the real news come from Reuters website whereas fake news are from websites deemed unreliable by Wikipedia. Other datasets come from different sources such as CNN, BBC, FOX, etc. But similar in date and distribution of real and fake. Together with the Kaggle dataset, this collection of data contains nearly 90000 newspapers.

For the Vietnamese test we use the Vietnamese Fake News Dataset (VFND). This contains over 200 news, collected in the period from 2017 to 2019 and uses several, cross-referencing sources, classified by the community.

All data are included in this repo.

## Tutorial

For inference, run [main.py](main.py)

All other files are for training. To retrain the models, uncomment the lo```load``` and ```dump``` function

```python
# in main.py
model_viet = load('models/model_viet.joblib')
model=load('models/model.joblib')
tfidf_vectorizer_viet = load('models/tfidf_viet.joblib')
tfidf_vectorizer = load('models/tfidf.joblib')
```

```python
# in english_test.py

dump(tfidf_vectorizer,'models/tfidf.joblib')
dump(count_vectorizer, 'models/count.joblib')

#dump(model,'models/model.joblib')
#dump(model,'models/modelc.joblib')
```

## Explanation

[english_test.py](english_test.py): sử dụng [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) và [TFIDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). Train bằng [PasiveAgressiveClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)

[vietnamese_test.py](vietnamese_test.py): giống english_test.py, sử dụng [Underthesea](https://underthesea.readthedocs.io/en/latest/readme.html) tokenizer(cho tiếng Việt) trước khi vectorize

[english_Word2vec_PAC.py](english_Word2vec_PAC.ipynb): Sử dụng Word2Vec. Train bằng PasiveAgressiveClassifier

test.py để chơi thôi, muốn xóa cũng đc
