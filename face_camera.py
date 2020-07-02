# import list
import os
import numpy as np
import cv2
import random


# some setting
IMG_SIZE = 64


def create_dir(*args):
    """

    :param args: the dir you want to store data
    :return:
    """
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)


def get_padding_size(shape):
    """
    get size to make image to be a square rect

    :param shape:
    :return:
    """
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest]*4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()


def deal_with_image(img, h=64, w=64):
    """

    :param img:
    :param h:
    :param w:
    :return:
    """
    top, bottom, left, right = get_padding_size(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img


def relight(img_src, alpha=1, bias=0):
    """

    :param imgsrc:
    :param alpha:
    :param bias:
    :return:
    """
    img_src = img_src.astype(float)
    img_src = img_src*alpha + bias
    img_src[img_src < 0] = 0
    img_src[img_src > 255] = 255
    img_src = img_src.astype(np.uint8)
    return img_src


def get_face_from_camera(out_dir):
    """

    :param out_dir:
    :return:
    """
    create_dir(out_dir)

    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier(r'/Users/neonrocks/Desktop/face_recognition/opencv/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

    n = 1
    img_num = 100

    while 1:
        if n <= img_num:
            print("It's processing %s image...." % n)
            success, img = camera.read()

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)

            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y + f_h, f_x:f_x + f_w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                # could deal with face to train
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                cv2.imwrite(os.path.join(out_dir, str(n) + '.jpg'), face)

                cv2.putText(img, 'Collecting', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 显示采集状态
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n += 1

            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff

            if key == 27:
                break
        else:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    name = input('please input your name: ')
    get_face_from_camera(os.path.join('./image/train_faces', name))

