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
import numpy as np

import cv2

AREA_DANGEROUS_FILE_ROOT = f"area_dangerous\\area_labels\\"  # 危险区域的标注文件的路径


def load_poly_area_data(img_name):
    """
    加载对用图片多边形点数据
    :param img_name: 图片名称
    :return: 多边形的坐标 [[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]] 二维数组
    """
    area_file_path = os.getcwd() + "\\" + AREA_DANGEROUS_FILE_ROOT
    json_file_name = area_file_path + img_name.split('.')[0] + ".json"

    if not Path(json_file_name).exists():
        print(f"json file {json_file_name} not exists !! ")
        return []

    with open(json_file_name, 'r') as f:
        json_info = json.load(f)

        area_poly = []
        for area_info in json_info['outputs']['object']:
            if 'polygon' not in area_info:
                return []

            pts_len = len(area_info['polygon'])
            if pts_len % 2 is not 0:  # 多边形坐标点必定是2的倍数
                return []

            xy_index_max = pts_len // 2
            for i in range(0, xy_index_max):  # "x1": 402,"y1": 234,"x2": 497,"y2": 182,.....
                str_index = str(i + 1)
                x_index = 'x' + str_index
                y_index = 'y' + str_index
                one_poly = [area_info['polygon'][x_index], area_info['polygon'][y_index]]
                area_poly.append(one_poly)

        return area_poly


def draw_poly_area_dangerous(img, img_name):
    """
    画多边形危险区域的框
    :param img_name: 检测的图片标号，用这个来对应图片的危险区域信息
    :param img: 图片数据
    :return: None
    """

    area_poly = np.array(load_poly_area_data(img_name), np.int32)
    cv2.polylines(img, [area_poly], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)


def is_ray_intersects_segment(poi, s_poi, e_poi):
    """
    正确计算射线与每条边是否相交。并且规定线段与射线重叠或者射线经过线段下端点属于不相交。首先排除掉不相交的情况,
    备注：输入都是 [x,y]->[lng,lat] 格式数组
    :param poi:判断点
    :param s_poi:边起点
    :param e_poi:边终点
    :return: True -> 合格点
    """
    if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
        return False
    if s_poi[0] < poi[0] and e_poi[1] < poi[1]:  # 线段在射线左边
        return False

    xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])  # 求交
    if xseg < poi[0]:  # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后


def is_poi_in_poly(poi, poly):
    # 输入：点，多边形三维数组
    # poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组

    # 可以先判断点是否在外包矩形内
    # if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    # 但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc = 0  # 交点个数
    for epoly in poly:  # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
        for i in range(len(epoly) - 1):  # [0,len-1]
            s_poi = epoly[i]
            e_poi = epoly[i + 1]
            if is_ray_intersects_segment(poi, s_poi, e_poi):
                sinsc += 1  # 有交点就加1

    return True if sinsc % 2 == 1 else False


def person_in_poly_area_dangerous(xyxy, img_name):
    """
    检测人体是否在多边形危险区域内
    :param xyxy: 人体框的坐标
    :param img_name: 检测的图片标号，用这个来对应图片的危险区域信息
    :return: True -> 在危险区域内，False -> 不在危险区域内
    """
    area_poly = load_poly_area_data(img_name)
    # print(area_poly)
    if not area_poly:  # 为空
        return False

    # 求物体框的中点
    object_x1 = int(xyxy[0])
    object_y1 = int(xyxy[1])
    object_x2 = int(xyxy[2])
    object_y2 = int(xyxy[3])
    object_w = object_x2 - object_x1
    object_h = object_y2 - object_y1
    object_cx = object_x1 + (object_w / 2)
    object_cy = object_y1 + (object_h / 2)

    return is_poi_in_poly([object_cx, object_cy], [area_poly])


if __name__ == '__main__':
    pass
