import time
from datetime import datetime
from pathlib import Path
import os
import cv2

os.chdir("../")
root = os.getcwd()

classifier = cv2.CascadeClassifier(root + '/lbpcascade_animeface.xml')

output_dir = root + '/out'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

p = Path(root + "/in")
list = list(p.glob("*"))

for file in list:
    print(file)
    # 顔の検出
    image = cv2.imread(str(file))
    # グレースケールで処理を高速化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_image)

    for i, (x, y, w, h) in enumerate(faces):
        now = time.time()
        utc = datetime.utcfromtimestamp(now)
        # 一人ずつ顔を切り抜く
        face_image = image[y:y+h, x:x+w]
        output_path = os.path.join(output_dir, '{0}-{1}.jpg'.format(datetime.now().timestamp(), i))
        cv2.imwrite(output_path, face_image)

    for x,y,w,h in faces:
        # 四角を描く
        cv2.rectangle(image, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=3)

    cv2.imwrite(output_dir + '/' + str(file), image)
