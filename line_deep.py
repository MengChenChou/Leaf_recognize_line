# --coding:utf-8--
from flask import Flask, request, abort
import tensorflow as tf
import numpy as np
import time
import os
import json
# import model_tmp_write_in

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    ImageMessage)

#圖片預處理,將圖片整理成模型可以接受的格式
def read_image(img_path):
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    except Exception as e:
        print(img_path,e)

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img/255


app = Flask(__name__)

#YOUR_CHANNEL_ACCESS_TOKEN
line_bot_api = LineBotApi('Sz3O/LkIS4af4vxkGW2m6CqRDh10ro/GnnY1Qn2lwVxZnvlbUIffsHGM7LkAj2Wx5j2q237kVad4M70ist32QlXIy79W8ONzmtLU6VuMxwRV4S/MlmbAwJsLBeOHN3fr5gJgeIDVREZBVOiJr2bIwAdB04t89/1O/w1cDnyilFU=')
#YOUR_CHANNEL_SECRET
handler = WebhookHandler('d18768f7d01f0e0c50ce25a2d89c0dfb')

@app.route("/callback", methods=['POST'])
def callback():

    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


#處理文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    #不處理官方發來的訊息
    if event.source.user_id != "Udeadbeefdeadbeefdeadbeefdeadbeef":

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=event.message.text) #event.message.text得到傳來的文字
        )

#處理影像訊息
@handler.add(MessageEvent, message=ImageMessage)
def handle_message(event):
    #不處理官方發來的訊息
    if event.source.user_id != "Udeadbeefdeadbeefdeadbeefdeadbeef":
        #使用影像id拿到影像內容
        message_content = line_bot_api.get_message_content(event.message.id) #event.message.id得到傳來的影像id

        #將影像內容存成jpg檔
        tempfile_path = os.path.join("./tmp", event.message.id+".jpg")
        with open(tempfile_path, 'wb') as fd:
            for chunk in message_content.iter_content():
                fd.write(chunk)

        print('開始推論...')
        start = time.process_time()
        #貼上訓練時的分類字典
        #labels = {'員工A': 0, '員工B': 1, '員工C': 2, '員工D': 3, '員工E': 4, '員工F': 5}
        labels = {'紅葉子': 0, '綠葉子': 1, '褐色葉子': 2, '黃綠葉子': 3}
        labels = {str(v): k for k, v in labels.items()}

        #載入模型
        model = tf.keras.models.load_model('mode_iv3LeafFinetune.h5')  # 辨識樹葉
        #model = tf.keras.models.load_model('myiv3LeafFinetune_employee.h5')
        # mdoel = m1


        img = read_image(tempfile_path)

        #推論
        pred = model.predict(img)[0]

        # 推論出機率最高的分類, 取得所在位置
        index = np.argmax(pred)
        print('印出測試圖片推論的分類與機率', tempfile_path, labels[str(index)], pred[index])

        end = time.process_time()

        print('推論圖片花費時間:', round(end - start), '秒')

        #產生推論報告文字

        response_text = "這張圖有{0}%是{1}，分析時間為{2}秒{3}".format(
            round(pred[index] * 100, 1),labels[str(index)], round(end - start), chr(0x1F436))


        #將報告文字推送到Line聊天室
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response_text)
        )

if __name__ == "__main__":
    #允許所有IP存取服務, 打開port 5000
    app.run(debug=True, port='5000', host='0.0.0.0')

