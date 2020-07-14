# --coding:utf-8--

from keras.preprocessing import image
import glob
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import os
from keras.models import load_model


def read_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(299, 299))
    except Exception as e:
        print(img_path,e)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img/255

def draw_save(img_path, label, out='tmp/'):
    img = Image.open(img_path)
    os.makedirs(os.path.join(out,label),exist_ok=True)
    if img is None:return None
    # 在圖片上加入文字
    draw = ImageDraw.Draw(img)
    # 使用中文字形
    font = ImageFont.truetype("TW-Kai-98_1.ttf",160)
    #fill文字顏色 黃色
    draw.text((10,10), label, fill='#FFFF00', font=font)
    img.save(os.path.join(out,label,"test.jpg"))


labels = {'紅葉子': 0, '綠葉子': 1, '褐色葉子': 2, '黃綠葉子': 3}
labels= {str(v): k for k,v in labels.items()}
print(labels)
#隨意選一個照片
files = glob.glob("leaf/test/紅葉子/*.jpg")

testID = 1
print(files[testID])


model = load_model('mode_iv3LeafFinetune.h5') #辨識樹葉
img = read_image(files[testID])

pred = model.predict(img)[0]
print(pred)

#推論出機率最高的分類, 取得所在位置
index = np.argmax(pred)

print(files[testID], labels[str(index)], pred[index])
draw_save(files[testID], labels[str(index)], out='tmp/')