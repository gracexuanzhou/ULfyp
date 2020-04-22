import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluoncv.utils import makedirs
import shutil
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import xml.etree.cElementTree as ET
from sklearn.model_selection import train_test_split

INPUT_DIR='D:\\BaiduNetdiskDownload\\'
OUPUT_DIR='D:\\BaiduNetdiskDownload\\dataset'
Annotation_Imgs='D:\\BaiduNetdiskDownload\\Anno\\list_category_img.txt'
Annotation_Bbox='D:\\BaiduNetdiskDownload\\Anno\\list_bbox.txt'
Annotation_Label='D:\\BaiduNetdiskDownload\\Anno\\list_category_cloth.txt'
makedirs(os.path.join(OUPUT_DIR,'Annotations'))
makedirs(os.path.join(OUPUT_DIR,'JPEGImages'))
makedirs(os.path.join(OUPUT_DIR,'ImageSets'))
makedirs(os.path.join(OUPUT_DIR,'ImageSets','Main'))

def get_imgs_and_labels():
    with open(Annotation_Imgs, 'r') as f:
        data=f.readlines()[2:]
        data=[con.strip() for con in data]
        imgs=[con.split() for con in data]
    return imgs

def copy_rename_img(all_file_path):
    file_name=os.path.split(all_file_path)[-1]
    clothes_name = os.path.split(all_file_path)[0].split('\\')[-1]
    new_file_name = clothes_name + '_' + file_name
    new_all_file_path=os.path.join(OUPUT_DIR,'JPEGImages',new_file_name)
    shutil.copyfile(all_file_path,new_all_file_path)
    return new_all_file_path

# move imgs to dir
def move_imgs(imgs):
    for path,label in imgs:
        path=path.replace('/','\\\\')
        input_path=os.path.join(INPUT_DIR,path)
        new_all_file_path=copy_rename_img(input_path)
        print('move jpgs to:%s'%new_all_file_path)

def get_annotation():
    with open(Annotation_Bbox,'r') as f:
        data=f.readlines()[2:]
        data=[con.strip() for con in data]
        data=[con.split() for con in data]
    return  data
def make_xml(xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple, image_name,img_height,img_width,label):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name
    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(xmin_tuple))
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img_width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(img_height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in range(len(xmin_tuple)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = label
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(xmin_tuple[i])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(ymin_tuple[i])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(xmax_tuple[i])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(ymax_tuple[i])
    xml = tostring(node_root, pretty_print = True)
    dom = parseString(xml)
    #print xml 打印查看结果
    # print(xml)
    xml_name = os.path.join(OUPUT_DIR,'Annotations', image_name.replace('.jpg','') + '.xml')
    with open(xml_name, 'wb') as f:
        f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))


def get_label():
    with open(Annotation_Imgs, 'r') as f:
        data=f.readlines()[2:]
        data=[con.split() for con in data]
    with open(Annotation_Label, 'r') as f:
        label=f.readlines()[2:]
        label = [con.split()[0] for con in label]
    img_to_label={}
    for path,number in data:
        idx=int(number)-1
        file_name = os.path.split(path)[-1]
        clothes_name = os.path.split(path)[0].split('/')[-1]
        new_file_name = clothes_name + '_' + file_name
        img_to_label[str(new_file_name)]=label[idx]
    return img_to_label



def get_xml_file():
    annotation_file=get_annotation()
    img_to_label=get_label()
    for path,x_min,y_min,x_max,y_max in annotation_file:
        x_min=int(x_min)
        y_min=int(y_min)
        x_max=int(x_max)
        y_max=int(y_max)
        x_mins = []
        y_mins = []
        x_maxs = []
        y_maxs = []
        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_mins.append(y_min)
        y_maxs.append(y_max)
        file_name = os.path.split(path)[-1]
        clothes_name = os.path.split(path)[0].split('/')[-1]
        new_file_name = clothes_name + '_' + file_name
        img_file_path=os.path.join(OUPUT_DIR, 'JPEGImages',new_file_name)
        # img=cv2.imread(img_file_path)
        # cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255,0,0))
        # plt.imshow(img)
        # plt.show()
        # break
        img_height,img_width=cv2.imread(img_file_path,0).shape
        label=img_to_label[new_file_name]
        make_xml(xmin_tuple=x_mins, ymin_tuple=y_mins, xmax_tuple=x_maxs,
                 ymax_tuple=y_maxs, image_name=new_file_name, img_height=img_height, img_width=img_width,label=label)




def load_label(path):
    """Parse xml file and return labels."""
    anno_path=path
    root = ET.parse(anno_path).getroot()
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    label = []
    for obj in root.iter('object'):
        try:
            difficult = int(obj.find('difficult').text)
        except ValueError:
            difficult = 0
        cls_name = obj.find('name').text.strip().lower()
        xml_box = obj.find('bndbox')
        xmin = (float(xml_box.find('xmin').text) - 1)
        ymin = (float(xml_box.find('ymin').text) - 1)
        xmax = (float(xml_box.find('xmax').text) - 1)
        ymax = (float(xml_box.find('ymax').text) - 1)
    print(xmin,ymin,xmax,ymax)


if __name__ == '__main__':
    # imgs=get_imgs_and_labels()
    # move_imgs(imgs)
    # get_xml_file()
    all_data_file = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(OUPUT_DIR, 'JPEGImages'))]
    # with open('D:\\BaiduNetdiskDownload\\Eval\\list_eval_partition.txt','r') as f:
    #     data=f.readlines()[2:]
    #     data = [con.split() for con in data]
    # train_data=[]
    # test_data=[]
    # for path,category in data:
    #     file_name = os.path.split(path)[-1]
    #     clothes_name = os.path.split(path)[0].split('/')[-1]
    #     new_file_name = clothes_name + '_' + file_name
    #     new_file_name = os.path.splitext(new_file_name)[0]
    #     if new_file_name in all_data:
    #         if category=='train':
    #             train_data.append(new_file_name)
    #         else:
    #             test_data.append(new_file_name)
    # print(len(all_data))
    # test_data = random.sample(all_data, 5000)
    # train_data =random.sample(list(set(all_data) - set(test_data)),50000)
    # print(len(train_data))
    # print(len(test_data))

    # labels=get_label()
    # all_data=[]
    # all_data_label=[]
    # for key,label in labels.items():
    #     path=os.path.splitext(key)[0]
    #     if path in all_data_file:
    #         all_data.append(path)
    #         all_data_label.append(label)
    # df=pd.DataFrame(data={'file_path':all_data,'label':all_data_label})
    # df.to_csv('./df.csv', index=False)
    # print(df['label'].value_counts())
    df=pd.read_csv('./df.csv')
    filter_label=(df['label'].value_counts()[df['label'].value_counts() >= 3000].index)
    df=df[df['label'].isin(filter_label)]
    unique_label=df['label'].unique().tolist()
    unique_label=[con.lower() for con in unique_label]
    print(unique_label)
    # print(len(all_data))
    _,test_df=train_test_split(df,test_size=55000,stratify=df['label'])
    train_data,test_data=train_test_split(test_df,test_size=5000, stratify=test_df['label'])
    train_data=train_data['file_path'].tolist()
    test_data=test_data['file_path'].tolist()
    print(len(train_data))
    print(len(test_data))
    with open(os.path.join(OUPUT_DIR,'ImageSets','Main','train.txt'), "w", encoding="utf-8", ) as f:
        f.write("\n".join(train_data))
    with open(os.path.join(OUPUT_DIR,'ImageSets','Main','test.txt'), "w", encoding="utf-8", ) as f:
        f.write("\n".join(test_data))

    # labels = get_label()
    # names=[]
    # for img in labels:
    #     name=labels[img]
    #     names.append(name)
    # names=list(set(names))
    # names=[name.lower() for name in names]
    # print(names)
    # print(len(names))