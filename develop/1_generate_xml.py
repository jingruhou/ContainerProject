# coding: utf-8
import os
import cv2
import glob
import random
import xml.etree.ElementTree as ET

from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

"""
    @Time        : 2020/6/29/0029 19:43
    @Author      : houjingru@semptian.com
    @FileName    : 1_generate_xml.py
    @Software    : PyCharm
    @Environment :Python2
"""

"""
    使用说明
        1、执行路径在图片文件夹下面
        2、图片文件夹下面必须有一个人工标注好的电视台台标xml文件
        3、生成的坐标值是在人工标注的坐标点基础上随机增加（1,3）个像素，不考虑减少像素，图标必须标注全
        4、生成的xml文件和jpg同在一个目录下面
"""


def make_xml(folder, filename, path, width, height, depth,
             name, xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple):
    """
        创建VOC格式的数据集-xml
    :param folder:文件夹
    :param filename:文件名称
    :param path:文件路径
    :param width:宽
    :param height:高
    :param depth:深度
    :param name:标签名称
    :param xmin_tuple:x
    :param ymin_tuple:y
    :param xmax_tuple:x_width
    :param ymax_tuple:y_height
    :return:
    """

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = folder

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = filename

    node_path = SubElement(node_root, 'path')
    node_path.text = path

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'Unknown'

    # node_object_num = SubElement(node_root, 'object_num')
    # node_object_num.text = str(len(xmin_tuple))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = width
    node_height = SubElement(node_size, 'height')
    node_height.text = height
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = depth

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    for i in range(len(xmin_tuple)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = name
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
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

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    print xml  # 打印查看结果
    return dom


def list_dir(dir_path):
    """
        递归读取图像文件
    :param dir_path:根目录
    :return:所有图像路径的list
    """
    frame_list = []
    dir_files = os.listdir(dir_path)  # 得到该文件夹下所有的文件
    for file in dir_files:
        file_path = os.path.join(dir_path, file)  # 路径拼接成绝对路径
        if os.path.isfile(file_path):  # 如果是文件，就追加到这个文件路径到list
            if file_path.endswith(".jpg"):
                frame_list.append(file_path)

    return frame_list


if __name__ == '__main__':

    rootDir = r"E:\HY_TV_LOGO_DATA\tv_video_train\29hainan\hainan_gg_8"

    # 1、读取文件夹下面的xml文件,获取各个节点及对应的数据值
    xml_file = glob.glob(rootDir + "\\*.xml")
    tree = ET.parse(xml_file[0])
    root = tree.getroot()

    # folder、filename、path节点
    for folder in root.iter('folder'):
        print(folder.tag, folder.text)
    for filename in root.iter('filename'):
        print(filename.tag, filename.text)
    for path in root.iter('path'):
        print(path.tag, path.text)

    # 帧尺寸及深度
    for width in root.iter('width'):
        print(width.tag, width.text)
    for height in root.iter('height'):
        print(height.tag, height.text)
    for depth in root.iter('depth'):
        print(depth.tag, depth.text)

    # 标签名称
    for name in root.iter('name'):
        print(name.tag, name.text)

    # 位置坐标
    for xmin in root.iter('xmin'):
        print(xmin.tag, xmin.text)
    for ymin in root.iter('ymin'):
        print(ymin.tag, ymin.text)
    for xmax in root.iter('xmax'):
        print(xmax.tag, xmax.text)
    for ymax in root.iter('ymax'):
        print(ymax.tag, ymax.text)

    # 2、读取文件夹下面的所有JPG图片
    file_list = list_dir(rootDir)
    print("Frames Num:", (len(file_list)))

    # 3、读取JPG文件，创建同名称的xml文件
    for jpg_file in file_list:
        # folder
        folder_ = rootDir.split("\\")[-1]
        # filename
        filename_ = jpg_file.split(".")[0].split("\\")[-1]
        # image_name
        img = cv2.imread(jpg_file)
        img_shape = img.shape
        img_height = str(img_shape[0])
        img_width = str(img_shape[1])
        img_depth = str(img_shape[2])

        xmin_tuple = []
        ymin_tuple = []
        xmax_tuple = []
        ymax_tuple = []

        # 生成随机数
        random1 = random.randint(-8, -3)
        random2 = random.randint(-8, -3)
        random3 = random.randint(3, 8)
        random4 = random.randint(3, 8)

        # 4、基于读取的坐标值生成随机坐标值-(1,3)
        xmin_tuple.append(int(xmin.text) + random1)
        ymin_tuple.append(int(ymin.text) + random2)
        xmax_tuple.append(int(xmax.text) + random3)
        ymax_tuple.append(int(ymax.text) + random4)
        # xmin 为坐标值，有多少物体，len(xmin_tuple)就是多少

        # 5、创建xml Dom
        # make_xml(folder, filename, path, width, height, depth, name, xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple)
        dom = make_xml(folder_, filename_, jpg_file, img_width, img_height, img_depth,
                       name.text, xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple)
        xml_name = os.path.join(rootDir, jpg_file.split(".")[0] + '.xml')
        # 6、将Dom写入到文件
        with open(xml_name, 'wb') as f:
            # f.write(dom.toprettyxml(indent='\t', newl='\n', encoding='utf-8'))
            f.write(dom.toprettyxml(indent='\t', newl='', encoding='utf-8')[38:])  # 删除换行符、xml文件头部