import json
import os

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class Yolov8Detector(object):
    def __init__(self, weights='yolov8.pt', confidence=0.3):
        # 模型加载
        self.ann = None
        self.model = YOLO(weights)
        self.confidence = confidence
        self.names = self.model.model.names
        self.names2id = {}
        for k, v in self.names.items():
            self.names2id[v] = k

    def inference_image(self, opencv_img, show_log=False):
        results = self.model(opencv_img, verbose=show_log)
        bbox = results[0].boxes
        result_list = []
        for idx in range(len(bbox.data)):
            xmin = int(bbox.data[idx][0])
            ymin = int(bbox.data[idx][1])
            xmax = int(bbox.data[idx][2])
            ymax = int(bbox.data[idx][3])
            conf = round(float(bbox.data[idx][4]), 2)
            cls_idx = int(bbox.data[idx][5])
            if conf >= self.confidence:
                result_list.append([self.names[cls_idx], conf, xmin, ymin, xmax, ymax])
        # for idx in range(len(bbox.boxes)):
        #     xmin = int(bbox.boxes[idx][0])
        #     ymin = int(bbox.boxes[idx][1])
        #     xmax = int(bbox.boxes[idx][2])
        #     ymax = int(bbox.boxes[idx][3])
        #     conf = round(float(bbox.boxes[idx][4]), 2)
        #     cls_idx = int(bbox.boxes[idx][5])
        #     if conf >= self.confidence:
        #         result_list.append([self.names[cls_idx], conf, xmin, ymin, xmax, ymax])
        return result_list  # [[name,conf,xmin,ymin,xmax,ymax],]

    def draw_image(self, result_list, opencv_img):
        self.ann = Annotator(opencv_img)
        for result in result_list:
            lbl = result[0] + ',' + str(result[1])
            self.ann.box_label(result[2:6], lbl, color=colors(self.names2id[result[0]], True))
        return self.ann.im

    def imshow(self, result_list, opencv_img):
        if len(result_list) > 0:
            opencv_img = self.draw_image(result_list, opencv_img)
        cv2.imshow('result', opencv_img)
        cv2.waitKey(0)

    def start_camera(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 获取视频帧宽度和高度
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video fps={},width={},height={}".format(frame_fps, frame_width, frame_height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_list = self.inference_image(frame)
            result_img = self.draw_image(result_list, frame)
            cv2.imshow('frame', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    def start_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_nums = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video{} fps={},nums={},width={},height={}".format(video_file, frame_fps, frame_nums, frame_width, frame_height))

        output_folder = 'web/download/'
        output_filename = video_file  # 输出视频文件名
        output_path = os.path.join(output_folder, output_filename)  # 输出视频文件的完整路径
        output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, output_fourcc, frame_fps, (frame_width, frame_height))

        # output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_fps,
        #                                (frame_width, frame_height))  # 输出视频文件的完整路径和参数设置

        door_info = [0] * frame_nums
        door_conf = [0] * frame_nums
        ev_info = [0] * frame_nums
        ev_conf = [0] * frame_nums
        gt_info = [0] * frame_nums
        gt_conf = [0] * frame_nums
        count = 0
        while True:
            if count >= frame_nums:
                break
            ret, frame = cap.read()
            if not ret:
                break
            result_list = self.inference_image(frame)

            # print(result_list)

            for result in result_list:
                if result[0] == 'electromobile':
                    ev_info[count] += 1
                    ev_conf[count] = max(ev_conf[count], result[1])
                if result[0] == 'door':
                    door_info[count] += 1
                    door_conf[count] = max(door_conf[count], result[1])
                if result[0] == 'gas tank':
                    gt_info[count] += 1
                    gt_conf[count] = max(gt_conf[count], result[1])

            frame = self.draw_image(result_list, frame)
            output_video.write(frame)
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            count += 1
            # print(count)
            # print(file_info)
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

        # return json.dumps(file_info)
        return door_info,door_conf,ev_info,ev_conf,gt_info,gt_conf

    def start_video2(self, video_file):
        cap = cv2.VideoCapture(video_file)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_nums = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video fps={},nums={},width={},height={}".format(frame_fps, frame_nums, frame_width, frame_height))

        output_folder = 'web/download/'
        output_filename = "results.mp4"  # 输出视频文件名
        output_path = os.path.join(output_folder, output_filename)  # 输出视频文件的完整路径
        print(output_path)

        # output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_fps,
        #                                (frame_width, frame_height))  # 输出视频文件的完整路径和参数设置
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 25
        frame_size = (frame_width, frame_height)  # 根据实际帧大小进行设置

        # 创建VideoWriter对象
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        count = 0
        while True:
            if count >= frame_nums:
                break
            ret, frame = cap.read()
            if not ret:
                break
            result_list = self.inference_image(frame)
            print(result_list)
            frame = self.draw_image(result_list, frame)
            out.write(frame)
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            count += 1
            print(count)
            # print(file_info)

        out.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    detector = Yolov8Detector()
    detector.start_video2('72.mp4')

