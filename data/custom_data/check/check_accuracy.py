# -*- coding: utf-8 -*-
# @Time    : 2020/8/2 22:04
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : check_accuracy.py
# @Software: PyCharm
# @Brief   : 进行数据准确度检查


"""
需要执行 python detect.py --save-txt --weights ./weights/pro_helmet_head_person.pt --source xxx
加入 --save-txt 是生成数据 txt 以便判断
"""
import os

TEST_DATA_LABELS_PATH = f"E:\AI_Project\AI_Learning\Dataset\Safety_Helmet_Train_dataset\score\labels\\test\\"
INFER_DATA_LABELS_PATH = f"/inference/output\\"


def calculate():
    """
    计算 输出的结果 和 测试集标签 的准确率
    :return:
    """
    file_path = INFER_DATA_LABELS_PATH
    files = os.listdir(file_path)
    file_num = 0
    accuracy_sum = 0
    for file in files:
        if not file.endswith(".txt"):
            continue

        file_num += 1  # 计算 txt文件数量

        txt_file_path = file_path + '\\' + file
        with open(txt_file_path) as fi:
            infer_cls_num = len(fi.readlines())  #
            print(f"{file} infer_cls_num = {infer_cls_num}")

        label_file_path = TEST_DATA_LABELS_PATH + file
        with open(label_file_path) as fl:
            label_cls_num = len(fl.readlines())  #
            print(f"{file} label_cls_num = {label_cls_num}")

        accuracy_temp = infer_cls_num / label_cls_num
        # print(accuracy_temp)

        if accuracy_temp >= 1:
            accuracy_temp = 1
        accuracy_sum += accuracy_temp
        # print(accuracy_sum)
    accuracy_rate = (accuracy_sum / file_num) * 100
    print(f"total test file = {file_num}, accuracy rate = {accuracy_rate}%")


if __name__ == '__main__':
    calculate()
