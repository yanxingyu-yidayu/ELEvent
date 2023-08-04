import time

import cv2
import numpy as np
from PIL import Image

from CodeDependency.detr.detr import Detection_Transformers

class DETRDetector(object):
    def __init__(self, weights='yolov8.pt', confidence=0.3):
        self.model = Detection_Transformers()
        # detr = Detection_Transformers()

    def inference_image(self,image, count=count):
        '''
                1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。
                2、如果想要获得预测框的坐标，可以进入detr.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
                3、如果想要利用预测框截取下目标，可以进入detr.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
                在原图上利用矩阵的方式进行截取。
                4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入detr.detect_image函数，在绘图部分对predicted_class进行判断，
                比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        # r_image = detr.detect_image(image, crop=crop, count=count)
        # r_image.show()
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, self.min_length)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images = images.cuda()
                images_shape = images_shape.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            results = self.bbox_util(outputs, images_shape, self.confidence)

            if results[0] is None:
                return image

            _results = results[0].cpu().numpy()
            top_label = np.array(_results[:, 5], dtype='int32')
            top_conf = _results[:, 4]
            top_boxes = _results[:, :4]

        result_list=[]
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))


            result_list.append([predicted_class],float("{:.2f}".format(score)),left,bottom,right,top)

        return result_list


    def draw_image(self,result_list,img):
        if len(result_list)==0:
            return  img
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // self.min_length, 1))
        for result in result_list:
            draw = ImageDraw.Draw(img)
            predicted_class=result[0]
            score=result[1]
            left=result[2]
            bottom=result[3]
            right=result[4]
            top=result[5]

            label = '{} {:.2f}'.format(predicted_class, score)

            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return img


    def start_video(self, video_path):
        cap = cv2.Videocap(video_path)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 获取视频帧宽度和高度
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video fps={},width={},height={}".format(frame_fps, frame_width, frame_height))

        # # 视频检测结果保存
        # if video_save_path != "":
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #     out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        while True:
            # 读取某一帧
            ref, frame = cap.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(detr.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = cv2.putText(frame, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            # if video_save_path != "":
            #     out.write(frame)

        cap.release()
        cv2.destroyAllWindows()
        # if video_save_path != "":
        #     print("Save processed video to the path :" + video_save_path)
        #     out.release()


if __name__ == '__main__':
    video_path = 0
    video_save_path = ""
