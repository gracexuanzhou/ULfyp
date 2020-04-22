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


def opencv_show(img, bboxes, scores=None, labels=None, thresh=0.5, class_names=None):
    img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
    if len(bboxes) < 1:
        return img
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            cv2.putText(img, '{:s} {:s}'.format(class_name, score), (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255), 3)
    return img


if __name__ == '__main__':
    net = model_zoo.faster_rcnn_resnet50_v1b_voc(pretrained=True, force_nms=True)
    net.reset_class(CLASSES)
    net.load_parameters('./params/faster_rcnn_resnet50_v1b_voc_best.params')
    img_path = './test_imgs/Cutout_Muscle_Tee_img_00000023.jpg'
    input_img = cv2.imread(img_path, 0)
    x, orig_img = data.transforms.presets.rcnn.load_test(img_path)
    box_ids, scores, bboxes = net(x)
    img = opencv_show(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes, thresh=0.5)
    output_imgs = img_path.replace('test_imgs', 'predict')
    img = cv2.resize(img, input_img.shape[::-1])
    cv2.imwrite(output_imgs, img)
    # ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes, thresh=0.5)
    # plt.axis('off')
    # plt.savefig(output_imgs,bbox_inches='tight')
    # plt.show()
