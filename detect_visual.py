# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 19:02
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : detect_visual.py
# @Software: PyCharm
# @Brief   :
import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *


class YOLOPredict(object):
    def __init__(self, device, weights, imgsz):
        # 加载模型
        self.model, self.half, self.names, self.colors, self.device = self.load_model(device, weights, imgsz)
        self.predict_info = "..."

    @staticmethod
    def load_model(device, weights, imgsz):
        """
        加载模型
        :param device:
        :param weights:
        :param imgsz:
        :return:
        """
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

    def detect(self, out, source, view_img, save_txt, imgsz, augment, conf_thres, iou_thres,
               cclasses, agnostic_nms, update, info_widget=None, save_img=False):
        """
        进行推理操作
        :param out:
        :param source:
        :param view_img:
        :param save_txt:
        :param imgsz:
        :param augment:
        :param conf_thres:
        :param iou_thres:
        :param cclasses:
        :param agnostic_nms:
        :param update:
        :param info_widget: QT 文本控件，用来显示推理信息
        :param save_img:
        :return:
        """

        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # # 加载模型
        # model, half, names, colors, device = self.load_model(self.device, weights, imgsz)

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
            dataset = LoadImages(source, img_size=imgsz)

        for path, img, im0s, vid_cap in dataset:
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

                if info_widget is not None:
                    # QT 控件打印信息
                    self.predict_info = '%sDone. (%.3fs)' % (s, t2 - t1)

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

        return save_path


if __name__ == '__main__':
    parameter_agnostic_nms = False
    parameter_augment = False
    parameter_classes = None
    parameter_conf_thres = 0.4
    parameter_device = ''
    parameter_img_size = 640
    parameter_iou_thres = 0.5
    parameter_output = 'inference/output'
    parameter_save_txt = False
    parameter_source = './area_dangerous'
    parameter_update = False
    parameter_view_img = False
    parameter_weights = ['./weights/helmet_head_person_m.pt']
    predict = YOLOPredict(parameter_device, parameter_weights, parameter_img_size)

    # with torch.no_grad():
    predict.detect(parameter_output, parameter_source, parameter_view_img, parameter_save_txt,
                   parameter_img_size, parameter_augment, parameter_conf_thres, parameter_iou_thres,
                   parameter_classes, parameter_agnostic_nms, parameter_update)

    # detect(parameter_output, parameter_source, parameter_weights, parameter_view_img, parameter_save_txt,
    #        parameter_img_size, parameter_augment, parameter_conf_thres, parameter_iou_thres, parameter_classes,
    #        parameter_agnostic_nms, parameter_device, parameter_update)
