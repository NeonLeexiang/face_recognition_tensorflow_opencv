import os
import logging as log
import numpy as np
import tensorflow as tf
import cv2
import face_cnn as myconv


def createdir(*args):
    """
    创建目录
    :param args: 目录名字
    :return:
    """
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)


IMG_SIZE = 64


def get_padding_size(shape):
    """
    获得使图像成为方形矩形的大小
    :param shape: 维度，例如shape=28，则图片被整理成28x28
    :return:
    """
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest] * 4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()


def deal_with_image(img, h=64, w=64):
    """
    处理图片
    :param img: 图像
    :param h: 图片高度
    :param w: 图片宽度
    :return:
    """
    # img = cv2.imread(img_path)
    top, bottom, left, right = get_padding_size(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img


def relight(img_src, alpha=1, bias=0):
    """
    改变图片亮度
    线性处理，根据一个公式  y = x * alpha + b
    :param img_src: 图片
    :param alpha: 公式中的alpha值
    :param bias: 公式中的b值
    :return:
    """
    '''relight'''
    img_src = img_src.astype(float)
    img_src = img_src * alpha + bias
    # 经过线性变换以后还要保证图片的像素值在0-255之间，
    # 如果小于0则置0，如果大于255则置为255
    img_src[img_src < 0] = 0
    img_src[img_src > 255] = 255
    img_src = img_src.astype(np.uint8)
    return img_src


def get_face(img_path, out_dir):
    """
    从路径中得到人脸
    :param img_path: 图片路径
    :param out_dir: 输出路径
    :return:
    """
    ''' get face from path file'''
    filename = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    
    # haarcascade_frontalface_default.xml是opencv中的一个特征检测器
    # 文件在opencv中的位置为：./opencv/sources/data/haarcascades
    
    haar = cv2.CascadeClassifier(r'/Users/neonrocks/Desktop/face_recognition/opencv/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)
    
    n = 0

    for f_x, f_y, f_w, f_h in faces:
        n += 1
        face = img[f_y:f_y + f_h, f_x:f_x + f_w]
        # 可能现在不需要调整大小
        # face = cv2.resize(face, (64, 64))
        face = deal_with_image(face, IMG_SIZE, IMG_SIZE)
        for inx, (alpha, bias) in enumerate([[1, 1], [1, 50], [0.5, 0]]):
            face_temp = relight(face, alpha, bias)
            cv2.imwrite(os.path.join(out_dir, '%s_%d_%d.jpg' % (filename, n, inx)), face_temp)


def get_files_in_path(file_dir):
    """
    从文件目录中取出所有的文件
    :param file_dir: 文件目录
    :return:
    """
    for (path, dir_names, filenames) in os.walk(file_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):   # 判断文件是否是以.jpg结尾的
                yield os.path.join(path, filename)
        for dir_item in dir_names:
            get_files_in_path(os.path.join(path, dir_item))


def generate_face(pair_dirs):
    """
    生成人脸
    :param pair_dirs: 图像的输入路径和输出路径
    :return:
    """
    for input_dir, output_dir in pair_dirs:
        for name in os.listdir(input_dir):
            input_name, output_name = os.path.join(input_dir, name), os.path.join(output_dir, name)
            if os.path.isdir(input_name):
                createdir(output_name)
                for file_item in get_files_in_path(input_name):
                    get_face(file_item, output_name)


def read_image(pair_path_label):
    """
    读取数据集下的所有文件，数据集中的每个数据都是一个文件夹，以姓名命名
    :param pair_path_label: 一个数据集的路径
    :return:返回数据集的数据(图片)以及相对应的标签(姓名)
    """
    imgs = []
    labels = []
    for filepath, label in pair_path_label:
        for file_item in get_files_in_path(filepath):
            img = cv2.imread(file_item)
            imgs.append(img)
            labels.append(label)
    return np.array(imgs), np.array(labels)


def one_hot(num_list):
    """
    得到一个矩阵序号
    把姓名转换成数字，例如有zonas，wangzongchao两个姓名，则分别转换成0和1
    :param num_list:姓名的列表
    :return:
    """
    ''' get one hot return host matrix is len * max+1 demensions'''
    b = np.zeros([len(num_list), max(num_list) + 1])
    b[np.arange(len(num_list)), num_list] = 1
    return b.tolist()


def get_file_and_label(file_dir):
    """
    将人脸从子目录内读出来，根据不同的人名，分配不同的one_hot值，这里是按照遍历的顺序分配序号，然后训练，完成之后会保存checkpoint
    图像识别之前将像素值转换为0到1的范围
    需要多次训练的话，把checkpoint下面的上次训练结果删除，代码有个判断，有上一次的训练结果，就不会再训练了

    :param file_dir:
    :return:
    """
    ''' get path and host paire and class index to name'''
    # dict_dir = dict([[name, os.path.join(file_dir, name)] for name in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, name))])
    dict_dir = dict()
    for name in os.listdir(file_dir):
        if os.path.isdir(os.path.join(file_dir, name)):
            dict_dir[name] = os.path.join(file_dir, name)
    # for (path, dir_names, _) in os.walk(file_dir) for dirname in dir_names])

    dir_name_list, dir_path_list = dict_dir.keys(), dict_dir.values()
    index_list = list(range(len(dir_name_list)))

    return list(zip(dir_path_list, one_hot(index_list))), dict(zip(index_list, dir_name_list))


def main(_):
    save_path = './checkpoint/face.ckpt'
    is_need_train = False   # 根据是否存在./checkpoint/face.ckpt.meta文件来判断是否需要训练
    if os.path.exists(save_path + '.meta') is False:
        is_need_train = True
    if is_need_train:
        # first generate all face
        log.debug('generate_face')
        generate_face([['./image/train_images', './image/train_faces']])
        path_label_pair, index_to_name = get_file_and_label('./image/train_faces')

        print("*******************************************")
        print(path_label_pair)
        print("*******************************************")

        train_x, train_y = read_image(path_label_pair)
        train_x = train_x.astype(np.float32) / 255.0
        log.debug('len of train_x : %s', train_x.shape)
        myconv.train(train_x, train_y, save_path)
        log.debug('training is over, please run again')
    else:
        test_from_camera(save_path)
        # print(np.column_stack((out, argmax)))


def test_from_camera(chkpoint):
    """
    识别图像
    从训练的结果中恢复训练识别的参数，然后用于新的识别判断
    打开摄像头，采集到图片之后，进行人脸检测，检测出来之后，进行人脸识别，根据结果对应到人员名字，显示在图片中人脸的上面
    :param chkpoint:
    :return:
    """
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier(r'/Users/neonrocks/Desktop/face_recognition/opencv/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    path_label_pair, index_to_name = get_file_and_label('./image/train_faces')
    output = myconv.cnn_layer(len(path_label_pair))
    # predict = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
    predict = output

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, chkpoint)

        n = 1
        while 1:
            if (n <= 20000):
                print('It`s processing %s image.' % n)
                # 读帧
                success, img = camera.read()

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = haar.detectMultiScale(gray_img, 1.3, 5)
                for f_x, f_y, f_w, f_h in faces:
                    face = img[f_y:f_y + f_h, f_x:f_x + f_w]
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    # could deal with face to train
                    test_x = np.array([face])
                    test_x = test_x.astype(np.float32) / 255.0

                    res = sess.run([predict, tf.argmax(output, 1)],
                                   feed_dict={myconv.x_data: test_x,
                                              myconv.keep_prob_5: 1.0, myconv.keep_prob_75: 1.0})
                    print(res)

                    cv2.putText(img, index_to_name[res[1][0]], (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 显示名字
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
    main(0)
    # one_hot([1, 3, 9])
    # print(get_file_and_label('./image/train_images'))
    # generate_face([['./image/train_images', './image/train_faces']])

