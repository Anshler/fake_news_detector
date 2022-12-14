from newsplease import NewsPlease
import PySimpleGUI as sg
import string
from underthesea import word_tokenize
from joblib import load
from langdetect import detect

def get_news_to_list(url_list):
    news_list = []
    for url in url_list:
        try:
            news = NewsPlease.from_url(url)
            if news.title is not None and news.maintext is not None:
                news_list.append(news)
        except:
            continue
    return news_list

stop_word = open('news/vietnamese/viet_stop_word.txt', 'r', encoding="utf-8").read().split('\n')
stop_word.extend(string.punctuation)
stop_word.extend(['…','”','–','“','. ảnh'])

def viet_tokenizer(text):
    text = word_tokenize(str(text))
    text = [word for word in text if word.lower() not in stop_word]
    return text

model_viet = load('models/model_viet.joblib')
model=load('models/model.joblib')
tfidf_vectorizer_viet = load('models/tfidf_viet.joblib')
tfidf_vectorizer = load('models/tfidf.joblib')

sg.Window._move_all_windows = True

def main():
    # -------------------- Thanh công cụ và interface --------------------

    # ----- Tạo thanh công cụ (tên program và ➖ ⏹ ❎) -----

    background_layout = [[sg.Col([[sg.Text('FAKE NEWS DETECTOR', font=('',13,'bold'), grab=True,
                            background_color= 'maroon', text_color= 'old lace')]]), #texbox tên chương trình, 'old lace' là màu trắng
                            sg.Col([[sg.Text('➖ ⏹ ❎', text_color= 'black',pad=(7,0), #textbox dấu ❎
                                enable_events=True,  key='Exit')]],
                                    element_justification='r', key='-C-', grab=True)],
                            [sg.Image('background.png',enable_events=True, key='Priority')]] # ảnh nền

    # ----- Window thanh công cụ -----

    window_background = sg.Window('Background', background_layout, keep_on_top=True, no_titlebar=True, finalize=True, margins=(0, 0),
                                  element_padding=(0, 0), size =(600,510))
    window_background['-C-'].expand(True)  #căn ❎ lề bên phải
    # ----- Interface -----

    upper = [[sg.Text('News URLs', background_color='old lace', text_color='black', font=('',12,''), expand_x= True, pad= (4,0))],
                [sg.Multiline(enable_events=True, size=(80, 10), background_color= 'old lace',
                              sbar_background_color='old lace', sbar_arrow_color='black', key='-URLLIST INP-', pad= (4,0))]]

    mid = [[sg.Col([[sg.Col([[sg.Button('▶ DETECT ◀', button_color='old lace on maroon', font=('',13,'bold'), key='-START PROCESS-', pad=(0,0))]],
                background_color='black')]],
                   #outline cho button, #theo dạng 3 box trắng, trong đen, trong đỏ -> đen ở giữa tạo thành đường viền
                    background_color='old lace', expand_x= True, element_justification='c', key='-X-', pad=(10,10))]]

    low = [[sg.Text('Results', background_color='old lace', text_color='black', font=('',12,''),expand_x= True, pad= (4,0))],
                 [sg.Multiline(key='-OUTPUT-', size=(80, 10), background_color= 'old lace',
                               sbar_background_color='old lace', sbar_arrow_color='black',
                               reroute_cprint=True, disabled=True, autoscroll=True, pad= (4,0))]]

    # ----- Full layout -----

    layout = [
        [sg.Column(upper)],
        [mid],
        [sg.Column(low)]
    ]

    # ----- Window interface -----

    window = sg.Window('Foreground', layout, finalize=True, keep_on_top=True, grab_anywhere=True,
                           transparent_color=sg.theme_background_color(), no_titlebar=True)
    window['-X-'].expand(True)  #căn button DETECT lề giữa

    # --------------------------------- Chạy Event Loop ---------------------------------

    while True:
        frame, event, values = sg.read_all_windows()

        if event is None or event == 'Exit': #bấm ❎ là thoát
            break
        if event == 'Priority': #background ko đc đè lên khung chữ
            window.BringToFront()
        if event == '-START PROCESS-': #nhập vào tạo list url
            url_list_str = values['-URLLIST INP-']
            url_list = url_list_str.split('\n')
            news = get_news_to_list(url_list) #đọc url lấy nội dung
            pred_list = []
            for a in news:
                lang=detect(a.title)
                if lang == 'vi':
                    data = str(a.title.lower()) + ' ' + str(a.maintext.lower())
                    pred = model_viet.predict(tfidf_vectorizer_viet.transform([data]))
                    pred_list.extend(pred)
                else:
                    data = str(a.title.lower()) + ' ' + str(a.maintext.lower())
                    pred = model.predict(tfidf_vectorizer.transform([data]))
                    pred_list.extend(pred)
            pred_word=''
            for i in range(len(pred_list)):
                if pred_list[i] == 1:
                    pred_word+= str(news[i].title) +'\n-Real -' + str(detect(news[i].title)) + '\n\n'
                else:
                    pred_word+= str(news[i].title) +'\n-Fake -' + str(detect(news[i].title)) + '\n\n'

            window['-OUTPUT-'].update(pred_word)

    window.close()
    window_background.close()

    # --------------------------------- Close & Exit ---------------------------------

if __name__ == '__main__':
    main()