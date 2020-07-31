# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 20:29
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : data_cfg.py
# @Software: PyCharm
# @Brief   : 生成测试、验证、训练的图片和标签

import os
from shutil import copyfile

from PIL import Image, ImageDraw
from xml.dom.minidom import parse
import numpy as np

FILE_ROOT = f"E:\AI_Project\AI_Learning\Dataset" + "\\"

IMAGE_SET_ROOT = FILE_ROOT + f"Safety_Helmet_Person_dataset\ImageSets\Main"  # 图片区分文件的路径
IMAGE_PATH = FILE_ROOT + f"Safety_Helmet_Person_dataset\JPEGImages"  # 图片的位置
ANNOTATIONS_PATH = FILE_ROOT + f"Safety_Helmet_Person_dataset\Annotations"  # 数据集标签文件的位置
LABELS_ROOT = FILE_ROOT + f"Safety_Helmet_Person_dataset\Labels"  # 进行归一化之后的标签位置

DEST_IMAGES_PATH = f"Safety_Helmet_Train_dataset\score\images"  # 区分训练集、测试集、验证集的图片目标路径
DEST_LABELS_PATH = f"Safety_Helmet_Train_dataset\score\labels"  # 区分训练集、测试集、验证集的标签文件目标路径


def cord_converter(size, box):
    """
    将标注的 xml 文件标注转换为 darknet 形的坐标
    :param size: 图片的尺寸： [w,h]
    :param box: anchor box 的坐标 [左上角x,左上角y,右下角x,右下角y,]
    :return: 转换后的 [x,y,w,h]
    """

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def save_file(img_jpg_file_name, size, img_box):
    save_file_name = LABELS_ROOT + '\\' + img_jpg_file_name + '.txt'
    print(save_file_name)
    file_path = open(save_file_name, "a+")
    for box in img_box:
        if box[0] == "head":
            cls_num = 0
        elif box[0] == "safetyhat":
            cls_num = 1
        elif box[0] == "person":
            cls_num = 2
        else:
            continue
        new_box = cord_converter(size, box[1:])
        file_path.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")

    file_path.flush()
    file_path.close()


def test_dataset_box_feature(file_name, point_array):
    """
    使用样本数据测试数据集的建议框
    :param image_name: 图片文件名
    :param point_array: 全部的点 [建议框sx1,sy1,sx2,sy2]
    :return: None
    """
    im = Image.open(rf"{IMAGE_PATH}\{file_name}")
    imDraw = ImageDraw.Draw(im)
    for box in point_array:
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        imDraw.rectangle((x1, y1, x2, y2), outline='red')

    im.show()


def get_xml_data(file_path, img_xml_file):
    img_path = file_path + '\\' + img_xml_file + '.xml'
    print(img_path)

    dom = parse(img_path)
    root = dom.documentElement
    img_name = root.getElementsByTagName("filename")[0].childNodes[0].data
    img_size = root.getElementsByTagName("size")[0]
    objects = root.getElementsByTagName("object")
    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    print("img_name:", img_name)
    print("image_info:(w,h,c)", img_w, img_h, img_c)
    img_box = []
    for box in objects:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
        y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
        x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
        y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
        print("box:(c,xmin,ymin,xmax,ymax)", cls_name, x1, y1, x2, y2)
        img_jpg_file_name = img_xml_file + '.jpg'
        img_box.append([cls_name, x1, y1, x2, y2])
    # print(img_box)

    # test_dataset_box_feature(img_jpg_file_name, img_box)
    save_file(img_xml_file, [img_w, img_h], img_box)


def copy_data(img_set_source, img_labels_root, imgs_source, type):
    file_name = img_set_source + '\\' + type + ".txt"
    file = open(file_name)
    for line in file.readlines():
        img_name = line.strip('\n')
        img_sor_file = imgs_source + '\\' + img_name + '.jpg'
        label_sor_file = img_labels_root + '\\' + img_name + '.txt'

        print(img_sor_file)
        print(label_sor_file)
        # im = Image.open(rf"{img_sor_file}")
        # im.show()

        # 复制图片
        DICT_DIR = FILE_ROOT + DEST_IMAGES_PATH + '\\' + type
        img_dict_file = DICT_DIR + '\\' + img_name + '.jpg'
        copyfile(img_sor_file, img_dict_file)

        if type is not "test":
            # 复制 label
            DICT_DIR = FILE_ROOT + DEST_LABELS_PATH + '\\' + type
            img_dict_file = DICT_DIR + '\\' + img_name + '.txt'
            copyfile(label_sor_file, img_dict_file)


if __name__ == '__main__':
    # 生成标签
    root = ANNOTATIONS_PATH
    files = os.listdir(root)
    for file in files:
        print("file name: ", file)
        file_xml = file.split(".")
        get_xml_data(root, file_xml[0])

    # 将文件进行 train 和 val 的区分
    # img_set_root = IMAGE_SET_ROOT
    # imgs_root = IMAGE_PATH
    # img_labels_root = LABELS_ROOT
    # copy_data(img_set_root, img_labels_root, imgs_root, "train")
    # copy_data(img_set_root, img_labels_root, imgs_root, "val")
    # copy_data(img_set_root, img_labels_root, imgs_root, "test")
