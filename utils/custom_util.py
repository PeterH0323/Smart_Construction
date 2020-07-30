# -*- coding: utf-8 -*-
# @Time    : 2020/7/30 22:20
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : custom_util.py
# @Software: PyCharm
# @Brief   : 检测危险区域里面的人
import json
import os
from pathlib import Path

import cv2

AREA_DANGEROUS_FILE_ROOT = f"area_dangerous\\area_labels\\"  # 危险区域的标注文件的路径


def draw_area_dangerous(img, img_name):
    """
    画危险区域的框
    :param img_name: 检测的图片标号，用这个来对应图片的危险区域信息
    :param img: 图片数据
    :return: None
    """
    area_file_path = os.getcwd() + "\\" + AREA_DANGEROUS_FILE_ROOT
    json_file_name = area_file_path + img_name.split('.')[0] + ".json"

    if not Path(json_file_name).exists():
        print(f"json file {json_file_name} not exists !! ")
        return

    with open(json_file_name, 'r') as f:
        json_info = json.load(f)
        area_info = json_info['outputs']['object'][0]['bndbox']

        # 危险区域
        area_x1 = area_info['xmin']
        area_y1 = area_info['ymin']
        area_x2 = area_info['xmax']
        area_y2 = area_info['ymax']
        cv2.rectangle(img, (area_x1, area_y1), (area_x2, area_y2), (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)


def person_in_area_dangerous(xyxy, img_name):
    """
    检测人体是否在危险区域内
    :param xyxy: 人体框的坐标
    :param img_name: 检测的图片标号，用这个来对应图片的危险区域信息
    :return: 1 -> 在危险区域内，0 -> 不在危险区域内
    """
    area_file_path = os.getcwd() + "\\" + AREA_DANGEROUS_FILE_ROOT
    json_file_name = area_file_path + img_name.split('.')[0] + ".json"

    if not Path(json_file_name).exists():
        print(f"json file {json_file_name} not exists !! ")
        return

    with open(json_file_name, 'r') as f:
        json_info = json.load(f)
        # print("=============")
        area_info = json_info['outputs']['object'][0]['bndbox']
        # print(area_info)

        # 物体框的位置
        object_x1 = int(xyxy[0])
        object_y1 = int(xyxy[1])
        object_x2 = int(xyxy[2])
        object_y2 = int(xyxy[3])
        object_w = object_x2 - object_x1
        object_h = object_y2 - object_y1
        object_cx = object_x1 + (object_w / 2)
        object_cy = object_y1 + (object_h / 2)

        # 危险区域
        area_x1 = area_info['xmin']
        area_y1 = area_info['ymin']
        area_x2 = area_info['xmax']
        area_y2 = area_info['ymax']

        # 判断是否在框内
        if (object_cx > area_x1) and (object_cx < area_x2):
            if (object_cy > area_y1) and (object_cy < area_y2):
                return 1

    return 0


if __name__ == '__main__':
    pass
