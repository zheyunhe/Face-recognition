import csv
import cv2
import dlib
import numpy as np
import pandas as pd


class Face_Recognizer:
    # 初始化进行某些参数的定义
    def __init__(self):
        # dlib 正向人脸检测器
        self.detector = dlib.get_frontal_face_detector()

        # dlib 人脸 landmark 特征点检测器
        self.predictor = dlib.shape_predictor("tools/shape_predictor_68_face_landmarks.dat")

        # dlib Resnet 人脸识别模型，提取 128D 的特征矢量
        self.face_feature_extractor = dlib.face_recognition_model_v1("tools/dlib_face_recognition_resnet_model_v1.dat")

        # 获取摄像头
        self.cap = cv2.VideoCapture(0)

        # OpenCV 人脸分类器
        self.face_cascade = cv2.CascadeClassifier("tools/haarcascade_frontalface_default.xml")

        # 使用者人脸特征
        self.user_face_feature = []

    """
    参数：两个含有人脸特征向量的列表
    作用：计算两个人脸特征向量间的欧氏距离
    """

    def euclidean_distance(self, feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        outcome = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return outcome

    """
    参数：使用者图片，使用者姓名
    作用：录入使用者信息
    """

    def face_input(self, frame, name):
        """
        获取使用者人脸位置
        face其实是一个list，每有一个人脸被检测出，位置就会以list形式存入face这个list中，例：face=[ [人脸1位置],[人脸2位置] ]
        这里因为只有一个人脸，所以face[0]即可
        """
        face = self.face_cascade.detectMultiScale(frame)
        x, y, w, h = face[0]
        # 画出人脸位置
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 人脸图片裁剪
        user_face = frame[y:y + h, x:x + w]
        # 人脸图片保存
        cv2.imwrite("information/user_face/user_face.jpg", user_face)
        # 获取人脸特征向量
        face = self.detector(user_face, 0)
        shape = self.predictor(user_face, face[0])
        user_face_feature = self.face_feature_extractor.compute_face_descriptor(user_face, shape)
        """
        信息保存——人脸特征向量、姓名
        user_face_feature的数据类型是'_dlib_pybind11.vector'
        转换为list后，数据可以保存为csv文件中的记录并可提取使用，不转换为list则提取出的就是一堆字符而不是数字
        """
        information = list(user_face_feature)
        information.append(name)
        # 以追加的方式写入csv文件
        with open("information/features.csv", "a+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(information)
            csvfile.close()

    '''
    参数：图片
    作用：进行人脸识别
    '''

    def face_recognize(self, frame):
        # 获取csv文件中的人脸特征向量
        csv_reader = pd.read_csv("information/features.csv", header=None)
        face_feature_known = []
        for i in range(0, 129):
            # 如果单元格是空的按0计算，否则append单元格中的数据
            if csv_reader.iloc[0][i] == '':
                face_feature_known.append('0')
            else:
                face_feature_known.append(csv_reader.iloc[0][i])

        # 记录下每次的欧氏距离
        euclidean_distance_temp = []

        # 放大一倍进行人脸检测
        faces = self.detector(frame, 1)
        # 如果有人脸，则存储这一帧的人脸特征
        if len(faces) != 0:
            for i in range(len(faces)):
                # 获取此人的人脸特征向量
                shape = self.predictor(frame, faces[i])
                face_feature = self.face_feature_extractor.compute_face_descriptor(frame, shape)
                euclidean_distance_temp.append(self.euclidean_distance(list(face_feature), face_feature_known[0:128]))

                # 寻找出最小的欧式距离匹配
                similar_person = euclidean_distance_temp.index(min(euclidean_distance_temp))
                if min(euclidean_distance_temp) < 0.4:
                    # 读取csv文件
                    csv_reader = pd.read_csv("information/features.csv", header=None)
                    print(csv_reader.iloc[0][128])
                else:
                    print("不认识")


if __name__ == "__main__":
    face_recognizer = Face_Recognizer()
    while True:
        ret, frame = face_recognizer.cap.read()
        face_recognizer.face_recognize(frame)
        cv2.imshow("test", frame)
        cv2.waitKey(1)
