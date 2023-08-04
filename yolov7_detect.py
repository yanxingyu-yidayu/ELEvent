import json
import os

import cv2
import torch
from numpy import random
import numpy as np
from CodeDependency.yolov7.models.experimental import attempt_load
from CodeDependency.yolov7.utils.datasets import letterbox
from CodeDependency.yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from CodeDependency.yolov7.utils.plots import plot_one_box
from CodeDependency.yolov7.utils.torch_utils import select_device


#
# from utils.datasets import letterbox, LoadStreams, LoadImages
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Yolov7Detector(object):
    def __init__(self, weights='yolov7.pt', confidence=0.25, iou_thres=0.45, device='', imgsz=640):
        self.confidence = confidence
        self.iou_thres = iou_thres
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride

        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # 热启动一次
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once

    def inference_image(self, opencv_img):
        # 图片预处理
        # Padded resize
        img = letterbox(opencv_img, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.confidence, self.iou_thres, classes=None, agnostic=False)

        result_list = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], opencv_img.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    result_list.append(
                        [self.names[int(cls)], round(float(conf), 2), int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                         int(xyxy[3])])
        return result_list  # 种类，置信度，框位置

    def draw_image(self, result_list, opencv_img):
        if len(result_list) == 0:
            return opencv_img
        for result in result_list:
            label = result[0] + ',' + str(result[1])
            plot_one_box(result[2:6], opencv_img, label=label, color=self.colors[self.names.index(result[0])],
                         line_thickness=1)
        return opencv_img

    def imshow(self, result_list, opencv_img):
        result_img = self.draw_image(result_list, img)
        cv2.imshow("result", result_img)
        cv2.waitKey(0)

    def start_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_nums = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video fps={},nums={},width={},height={}".format(frame_fps, frame_nums, frame_width, frame_height))

        # 创建同名的JSON文件路径
        # json_file = os.path.splitext(video_file)[0] + ".json"
        output_folder = 'tmpData/video/'
        output_filename = video_file  # 输出视频文件名
        output_path = os.path.join(output_folder, output_filename)  # 输出视频文件的完整路径

        output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_fps,
                                       (frame_width, frame_height))  # 输出视频文件的完整路径和参数设置

        # 创建一个包含文件信息的字典
        file_info = {
            "name": os.path.basename(video_file),
            "path": os.path.abspath(video_file),
            "door_info": [0] * frame_nums,
            "door_conf": [0] * frame_nums,
            "ev_info": [0] * frame_nums,
            "ev_conf": [0] * frame_nums,
            "gt_info": [0] * frame_nums,
            "gt_conf": [0] * frame_nums
        }
        count = 0
        while True:
            if count >= frame_nums:
                break
            ret, frame = cap.read()
            if not ret:
                break
            # if count % 5 != 0:
            #     count += 1
            #     continue
            result_list = self.inference_image(frame)
            for result in result_list:
                if result[0] == 'electromobile':
                    file_info["ev_info"][count] += 1
                    file_info["ev_conf"][count] = max(file_info["ev_conf"][count], result[1])
                if result[0] == 'door':
                    file_info["door_info"][count] += 1
                    file_info["door_conf"][count] = max(file_info["door_conf"][count], result[1])
                if result[0] == 'gas tank':
                    file_info["gt_info"][count] += 1
                    file_info["gt_conf"][count] = max(file_info["gt_conf"][count], result[1])

            frame = self.draw_image(result_list, frame)
            output_video.write(frame)
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            count += 1
            print(count)
            print(file_info)
        cap.release()
        output_video.release()
        # cv2.destroyAllWindows()

        # # 将字典写入JSON文件
        # with open(json_file, 'w') as file:
        #     json.dump(file_info, file, indent=4)

        return json.dumps(file_info)


    def start_camera(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(frame_fps, frame_width, frame_height)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_list = self.inference_image(frame)
            frame = self.draw_image(result_list, frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # def write_txt(self, result_list):


if __name__ == '__main__':
    detector = Yolov7Detector()
    # img = cv2.imread('testIMG.png')
    # result_list = detector.inference_image(img)
    # print(result_list)
    # result_img = detector.draw_image(result_list, img)
    # cv2.imshow("result", result_img)
    # cv2.waitKey(0)
    # detector.imshow(result_list, img)
    result = detector.start_video('72.mp4')
    print(result)
