# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 19:02
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : detect_visual.py
# @Software: PyCharm
# @Brief   :
from copy import deepcopy
from PyQt5.QtGui import QImage, QPixmap

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *


class YOLOPredict(object):
    def __init__(self, weights, out_file_path):
        """
        YOLO 模型初始化
        :param weights: 权重路径
        :param out_file_path: 推理结果存放路径
        """

        '''模型参数'''
        self.agnostic_nms = False
        self.augment = False
        self.classes = None
        self.conf_thres = 0.4
        self.device = ''
        self.img_size = 640
        self.iou_thres = 0.5
        self.output = out_file_path
        self.save_txt = False
        self.update = False
        self.view_img = False
        self.weights = weights  # 权重文件路径，修改这里

        # 加载模型
        self.model, self.half, self.names, self.colors, self.device = self.load_model()

        self.predict_info = ""

    def load_model(self):
        """
        加载模型
        :return: 模型
        """
        imgsz = self.img_size
        weights = self.weights
        device = self.device
        # Initialize
        device = torch_utils.select_device(device)

        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        return model, half, names, colors, device

    @staticmethod
    def show_real_time_image(image_label, img):
        """
        image_label 显示实时推理图片
        :param image_label: 本次需要显示的 label 句柄
        :param img: cv2 图片
        :return:
        """
        image_label_width = image_label.width()
        resize_factor = image_label_width / img.shape[1]

        img = cv2.resize(img, (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor)),
                         interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
        image = QImage(img_rgb[:],
                       img_rgb.shape[1],
                       img_rgb.shape[0],
                       img_rgb.shape[1] * 3,
                       QImage.Format_RGB888)
        img_show = QPixmap(image)
        image_label.setPixmap(img_show)

    def detect(self, source, save_img=False, qt_input=None, qt_output=None):
        """
        进行推理操作
        :param source: 推理素材
        :param save_img: 保存图片 flag
        :param qt_input: QT 输入窗口
        :param qt_output: QT 输出窗口
        :return:
        """
        out = self.output
        view_img = self.view_img
        save_txt = self.save_txt
        imgsz = self.img_size
        augment = self.augment
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        cclasses = self.classes
        agnostic_nms = self.agnostic_nms
        update = self.update

        # if os.path.exists(out):
        #     shutil.rmtree(out)  # delete output folder
        os.makedirs(out, exist_ok=True)  # make new output folder
        show_count = 0
        t0 = time.time()
        # Set Data loader
        vid_path, vid_writer = None, None
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz, visualize_flag=True)

        for path, img, im0s, vid_cap, info_str in dataset:

            # im0s 为当前推理的图片
            origin_image = deepcopy(im0s)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = self.model(img, augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=cclasses, agnostic=agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))  # 打印每张图片的推理信息

                # 保存推理信息
                self.predict_info = info_str + '%sDone. (%.3fs)' % (s, t2 - t1)
                # QT 显示
                if qt_input is not None and qt_output is not None and dataset.mode == 'video':
                    video_count, vid_total = info_str.split(" ")[2][1:-1].split("/")  # 得出当前总帧数
                    fps = ((t2 - t1) / 1) * 100
                    fps_threshold = 25  # FPS 阈值
                    show_flag = True
                    if fps > fps_threshold:  # 如果 FPS > 阀值，则跳帧处理
                        fps_interval = 15  # 实时显示的帧率
                        show_unit = math.ceil(fps / fps_interval)  # 取出多少帧显示一帧，向上取整
                        if int(video_count) % show_unit != 0:  # 跳帧显示
                            show_flag = False
                        else:
                            show_count += 1

                    if show_flag:
                        # 推理前的图片 origin_image, 推理后的图片 im0
                        self.show_real_time_image(qt_input, origin_image)
                        self.show_real_time_image(qt_output, im0)

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            print('Results saved to %s' % str(out))
            if platform == 'darwin' and not update:  # MacOS
                os.system('open ' + save_path)

        print('Done. (%.3fs)' % (time.time() - t0))
        self.predict_info = 'Done. (%.3fs)' % (time.time() - t0)

        return save_path


if __name__ == '__main__':
    print("This is not for run, may be you want to run 'detect.py' or 'visual_interface.py', pls check your file name. Thx ! ")
#     parameter_agnostic_nms = False
#     parameter_augment = False
#     parameter_classes = None
#     parameter_conf_thres = 0.4
#     parameter_device = ''
#     parameter_img_size = 640
#     parameter_iou_thres = 0.5
#     parameter_output = 'inference/output'
#     parameter_save_txt = False
#     parameter_source = './area_dangerous'
#     parameter_update = False
#     parameter_view_img = False
#     parameter_weights = ['./weights/helmet_head_person_m.pt']
#     predict = YOLOPredict(parameter_device, parameter_weights, parameter_img_size)

#     # with torch.no_grad():
#     predict.detect(parameter_output, parameter_source, parameter_view_img, parameter_save_txt,
#                    parameter_img_size, parameter_augment, parameter_conf_thres, parameter_iou_thres,
#                    parameter_classes, parameter_agnostic_nms, parameter_update)

    # detect(parameter_output, parameter_source, parameter_weights, parameter_view_img, parameter_save_txt,
    #        parameter_img_size, parameter_augment, parameter_conf_thres, parameter_iou_thres, parameter_classes,
    #        parameter_agnostic_nms, parameter_device, parameter_update)
