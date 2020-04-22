import os
import cv2
import random
import gluoncv
import mxnet as mx
import numpy as np
from mxnet import ndarray
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'

CLASSES =['blouse', 'blazer', 'tee', 'tank', 'top', 'sweater', 'hoodie', 'cardigan', 'jacket', 'skirt', 'shorts', 'jeans', 'joggers', 'sweatpants', 'cutoffs', 'sweatshorts', 'leggings', 'dress', 'romper', 'coat', 'kimono', 'jumpsuit']



def predict_res(img,resized_img,bboxes,scores=None,labels=None, thresh=0.5,class_names=None):
    this_img = img.copy()
    this_img[:, :, (0, 1, 2)] = this_img[:, :, (2, 1, 0)]
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    #计算回去坐标
    height = this_img.shape[0]
    width = this_img.shape[1]
    resized_height=resized_img.shape[0]
    resized_width=resized_img.shape[1]
    ouput_width_ratio=float(width/resized_width)
    ouput_height_ratio=float(height/resized_height)
    bboxes[:, (0, 2)] *= ouput_width_ratio
    bboxes[:, (1, 3)] *= ouput_height_ratio
    res=[]
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        x_min, y_min, x_max, y_max = [int(x) for x in bbox]
        res.append([x_min, y_min, x_max, y_max,class_name])
    return res

if __name__ == '__main__':
    net = model_zoo.faster_rcnn_resnet50_v1b_voc(pretrained=True, force_nms=True)
    net.reset_class(CLASSES)
    net.load_parameters('./params/weight.params')
    #读取视频
    video = cv2.VideoCapture('D:\\BaiduNetdiskDownload\\fashion.mp4')
    # 接下来获取视频的宽和高
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 提前指定好视频输出的格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter('output.mp4', fourcc, 24, (w, h))
    n = 0
    while True:
        try:
            img, frame = video.read()
            font = cv2.FONT_HERSHEY_COMPLEX
            frame_nd=mx.nd.array(frame)
            x, resized_img = data.transforms.presets.rcnn.transform_test(imgs=frame_nd)
            box_ids, scores, bboxes = net(x)
            res = predict_res(frame, resized_img=resized_img, bboxes=bboxes[0], scores=scores[0], labels=box_ids[0],
                              thresh=0.9, class_names=CLASSES)
            if len(res)==1:
                x_min, y_min, x_max, y_max,class_name=res[0]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=2)
                cv2.putText(frame, class_name, (x_min, y_min-2), font, 1, (0, 0, 255), 1, False)
            # cv2.imshow('frame', frame)
            cv2.imwrite('temp/frame%s.jpg'%n,frame)
            print('正在处理:%s'%n)
            videoWriter.write(frame)
            cv2.waitKey(1)
            n=n+1
        except Exception as err:
            print(err)
            break
    video.release()
    videoWriter.release()
    cv2.destroyAllWindows()
print('完成！')