# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 19:02
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : detect_visual.py
# @Software: PyCharm
# @Brief   :


import sys
import time
from pathlib import Path
from copy import deepcopy

from PyQt5.QtGui import QImage, QPixmap
import cv2
import torch
import torch.backends.cudnn as cudnn


FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync


class YOLOPredict(object):
    def __init__(self, weights, out_file_path):
        """
        YOLO 模型初始化
        :param weights: 权重路径
        :param out_file_path: 推理结果存放路径
        """

        '''模型参数'''
        self.weights = weights  # model.pt path(s)
        self.imgsz = [640, 640]  # inference size (pixels)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = False  # show results
        self.save_txt = False  # save results to *.txt
        self.save_conf = False  # save confidences in --save-txt labels
        self.save_crop = False  # save cropped prediction boxes
        self.nosave = False  # do not save images/videos
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models
        self.project = 'runs/detect'  # save results to project/name
        self.name = 'exp'  # save results to project/name
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference

        self.stride = ""
        self.pt = False  # pt flag
        self.classify = False
        self.modelc = ""
        self.model = ""

        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        self.output = out_file_path

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # 加载模型
        self.load_model()

        self.predict_info = ""  # 预测信息

    def load_model(self):
        """
        加载模型
        """
        # Load model
        w = self.weights[0] if isinstance(self.weights, list) else self.weights
        self.classify, suffix = False, Path(w).suffix.lower()
        self.pt, onnx, tflite, pb, saved_model = (suffix == x for x in
                                                  ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        if self.pt:
            self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            self.names = self.model.module.names if hasattr(self.model,
                                                            'module') else self.model.names  # get class names
            if self.half:
                self.model.half()  # to FP16
            if self.classify:  # second-stage classifier
                self.modelc = load_classifier(name='resnet50', n=2)  # initialize
                self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(
                    self.device).eval()
        else:  # TensorFlow models
            raise TypeError("model type not support.")

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
        # view_img = self.view_img
        # save_txt = self.save_txt
        # augment = self.augment
        # conf_thres = self.conf_thres
        # iou_thres = self.iou_thres
        # agnostic_nms = self.agnostic_nms
        # update = self.update

        save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        ascii = is_ascii(self.names)  # names are ascii (use PIL for UTF-8)

        # Dataloader
        if webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=self.stride, auto=self.pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=self.stride, auto=self.pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        if self.pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            t1 = time_sync()
            if self.pt:
                self.visualize = increment_path(self.save_dir / Path(path).stem,
                                                mkdir=True) if self.visualize else False
                pred = self.model(img, augment=self.augment, visualize=self.visualize)[0]
            else:
                raise TypeError("model type not support.")

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)
            t2 = time_sync()

            # Second-stage classifier (optional)
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, pil=not ascii)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (
                                self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg',
                                             BGR=True)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                im0 = annotator.result()
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

        if self.save_txt or save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)

        print(f'Done. ({time.time() - t0:.3f}s)')

        return self.save_dir


if __name__ == "__main__":
    check_requirements(exclude=('tensorboard', 'thop'))
    yolo_handle = YOLOPredict(r'./weight/best.pt', "")
    yolo_handle.detect(r'data/images')
    yolo_handle.detect(r'data/video')

# if __name__ == '__main__':
#     print(
#         "This is not for run, may be you want to run 'detect.py' or 'visual_interface.py', pls check your file name. Thx ! ")
