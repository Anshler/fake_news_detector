from joblib import dump, load
from newsplease import NewsPlease
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

model=load('model.joblib')
tfidf_vectorizer = load('tfidf.joblib')

def get_news_to_list(url_list):
    news_list = []
    for url in url_list:
        try:
            news = NewsPlease.from_url(url)
            news_list.append(news)
        except:
            continue
    return news_list

url_list=[]
url_list.append('https://edition.cnn.com/2022/09/24/middleeast/mahsa-amini-death-iran-internet-un-investigation-intl-hnk/index.html')
url_list.append('https://edition.cnn.com/travel/article/skytrax-world-airline-awards-2022/index.html')
url_list.append('https://edition.cnn.com/2022/09/23/australia/ancient-rock-art-gas-project-climate-threat-intl-hnk-dst-scn/index.html')
#url_list.append('https://vietnamnews.vn/economy/1337858/raising-awareness-on-blockchain-potential-crucial-experts.html')
#url_list.append('https://vietnamnews.vn/economy/1337781/us-businesses-interested-in-cooperating-with-vie-t-nam-in-energy-transition-digital-transformation.html')
#url_list.append('https://vietnamnews.vn/economy/1337775/untapped-opportunities-for-vietnamese-pepper-exports-in-the-uk.html')
#url_list.append('https://vietnamnews.vn/society/1337522/farmers-rise-up-in-difficulty-to-get-rich.html')
#url_list.append('https://vietnamnews.vn/travel/1337914/vie-t-nam-among-10-best-destinations-for-germans-to-escape-winter-news-site.html')
url_list.append('https://edition.cnn.com/2022/09/24/tech/iran-internet-blackout/index.html')
url_list.append('https://edition.cnn.com/2022/09/23/investing/dow-stock-market-today/index.html')
url_list.append('https://edition.cnn.com/2022/09/24/economy/home-heat-winter/index.html')
#url_list.append('https://www.wsmv.com/2022/09/23/macon-deputy-rams-car-speeding-through-homecoming-parade-route/')
#url_list.append('https://www.cbsnews.com/chicago/news/how-did-plainfield-get-max-from-stranger-things-halloween-decoration-float/')
#url_list.append('https://bleacherreport.com/articles/10050165-kenyas-eliud-kipchoge-sets-marathon-world-record-at-37-years-old-in-berlin?utm_source=cnn.com&utm_medium=referral&utm_campaign=editorial')
url_list.append('https://edition.cnn.com/2022/09/25/europe/italy-election-results-intl/index.html')
url_list.append('https://edition.cnn.com/2022/09/25/entertainment/rihanna-nfl-superbowl-halftime-show-spt/index.html')
url_list.append('https://edition.cnn.com/travel/article/fehmarnbelt-longest-immersed-tunnel-cmd/index.html')
url_list.append('https://edition.cnn.com/2022/09/22/business/boeing-737-sec/index.html')
url_list.append('https://edition.cnn.com/style/article/photographer-hal-flesh-love/index.html')
url_list.append('https://edition.cnn.com/2022/09/23/tennis/roger-federer-final-match-career-spt-intl/index.html')
url_list.append('https://edition.cnn.com/2022/09/23/opinions/mahsa-amini-iran-protests-hair-women-nemat/index.html')
url_list.append('https://edition.cnn.com/2022/09/25/investing/stocks-week-ahead/index.html')
url_list.append('https://edition.cnn.com/2022/09/24/business/goldman-sachs-huge-lawsuit-sexual-harassment/index.html')
url_list.append('https://edition.cnn.com/2022/09/25/tech/ai-film-salt/index.html')
news=get_news_to_list(url_list)
data=[]
for a in news:
    if a.maintext is not None and a.title is not None:
        data.append(str(a.title).lower()+' '+str(a.maintext).lower())

y_pred = model.predict(tfidf_vectorizer.transform(data))
print(y_pred)
#print(data[0])
