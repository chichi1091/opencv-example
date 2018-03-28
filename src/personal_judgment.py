import cv2, os
import numpy as np
from PIL import Image

os.chdir("../")
root = os.getcwd()

# 学習画像
train_path = root + '/out'
# テスト画像
test_path = root + '/test_data'

# Haar-like特徴分類器
cascadePath = root + "/lbpcascade_animeface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# LBPH
recognizer = cv2.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    # 画像を格納する配列
    images = []
    # ラベルを格納する配列
    labels = []
    # ファイル名を格納する配列
    files = []
    for f in os.listdir(path):
        # 画像のパス
        image_path = os.path.join(path, f)
        # グレースケールで画像を読み込む
        image_pil = Image.open(image_path).convert('L')
        # NumPyの配列に格納
        image = np.array(image_pil, 'uint8')
        # Haar-like特徴分類器で顔を検知 (パラメータは適当)
        faces = faceCascade.detectMultiScale(image,1.1,9,0)

        # 検出した顔画像の処理
        for (x, y, w, h) in faces:
            # 顔を 200x200 サイズにリサイズ
            roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
            # 画像を配列に格納
            images.append(roi)
            # ファイル名からラベルを取得 "0_xxxxx.jpg みたいなファイル名を想定している"
            labels.append(int(f[0]))
            # ファイル名を配列に格納
            files.append(f)

    return images, labels, files

# トレーニング画像を取得
images, labels, files = get_images_and_labels(train_path)
# トレーニング実施
recognizer.train(images, np.array(labels))
# テスト画像を取得
test_images, test_labels, test_files = get_images_and_labels(test_path)

i = 0
while i < len(test_labels):
    # テスト画像に対して予測実施
    label, confidence = recognizer.predict(test_images[i])
    # 予測結果をコンソール出力
    print("Test Image: {}, Predicted Label: {}, Confidence: {}".format(test_files[i], label, confidence))
    # テスト画像を表示
    cv2.imshow("test image", test_images[i])
    cv2.waitKey(1000)

    i += 1


cv2.destroyAllWindows()
