# DETECT FAKE NEWS IN ENGLISH AND VIETNAMESE

## Tutorial

Chỉ chạy [main.py](main.py)

Các file python khác để train model. Model train sẵn rồi ko cần chạy làm j. Nếu muốn retrain thì xóa # trước hàm 'load' và 'dump'

```python
    #main.py
#model_viet = load('models/model_viet.joblib')
#model=load('models/model.joblib')
#tfidf_vectorizer_viet = load('models/tfidf_viet.joblib')
#tfidf_vectorizer = load('models/tfidf.joblib')
```

```python
    #english_test.py
    #lưu vectorizer
#dump(tfidf_vectorizer,'models/tfidf.joblib')
#dump(count_vectorizer, 'models/count.joblib')

    #lưu model
#dump(model,'models/model.joblib')
#dump(model,'models/modelc.joblib')
```

## Explanation

[english_test.py](english_test.py): sử dụng [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) và [TFIDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). Train bằng [PasiveAgressiveClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)

[vietnamese_test.py](vietnamese_test.py): giống english_test.py, sử dụng [Underthesea](https://underthesea.readthedocs.io/en/latest/readme.html) tokenizer(cho tiếng Việt) trước khi vectorize

[english_Word2vec_PAC.py](english_Word2vec_PAC.ipynb): Sử dụng Word2Vec. Train bằng PasiveAgressiveClassifier

test.py để chơi thôi, muốn xóa cũng đc
