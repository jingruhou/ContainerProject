# encoding=utf8
"""
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
import time

# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,  # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)  # use test mode (e.g., don't perform dropout)
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=5):
        """
        SSD detection
        """
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

	# HJR
        start_time = time.time()

        # Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']
	
	# HJR
        end_time = time.time()
        print("Detection time:%s" % (end_time - start_time))

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i]  # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i]  # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i]  # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i]  # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result


def main(args):
    """main """
    detection = CaffeDetection(args.gpu_id,args.model_def, args.model_weights, args.image_resize, args.labelmap_file)
    im_names = ['001.jpg', '002.jpg', '003.jpg', '004.jpg', '005.jpg', '006.jpg', '007.jpg', '008.jpg', '009.jpg',
                '010.jpg', '011.jpg', '012.jpg', '013.jpg', '014.jpg', '015.jpg', '016.jpg', '017.jpg', '018.jpg',
                '019.jpg', '020.jpg', '021.jpg', '022.jpg', '023.jpg', '024.jpg', '025.jpg', '026.jpg', '027.jpg',
                '028.jpg', '029.jpg', '030.jpg', '031.jpg', '032.jpg', '033.jpg', '034.jpg', '035.jpg', '036.jpg',
                '037.jpg', '038.jpg', '039.jpg', '040.jpg', '041.jpg', '042.jpg', '043.jpg', '044.jpg', '045.jpg',
                '046.jpg', '047.jpg', '048.jpg', '049.jpg', '050.jpg', '051.jpg', '052.jpg', '053.jpg', '054.jpg',
                '055.jpg', '056.jpg', '057.jpg', '058.jpg', '059.jpg', '060.jpg', '061.jpg', '062.jpg', '063.jpg',
                '064.jpg', '065.jpg', '066.jpg', '067.jpg', '068.jpg', '069.jpg', '070.jpg', '071.jpg', '072.jpg',
                '073.jpg', '074.jpg', '075.jpg', '076.jpg', '077.jpg', '078.jpg', '079.jpg', '080.jpg', '081.jpg',
                '082.jpg', '083.jpg', '084.jpg', '085.jpg', '086.jpg', '087.jpg', '088.jpg', '089.jpg', '090.jpg',
                '091.jpg', '092.jpg', '093.jpg', '094.jpg', '095.jpg', '096.jpg', '097.jpg', '098.jpg', '099.jpg',
                '100.jpg']
    for im_name in im_names:
        image_file = args.image_file + im_name
        result = detection.detect(image_file)
        print result

        img = Image.open(image_file)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        print width, height
        for item in result:
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
            draw.text([xmin, ymin], item[-1] + " " + str(item[-2]), (0, 0, 255))
	    # 
	                

            print item
            print [xmin, ymin, xmax, ymax]
            print [xmin, ymin], item[-1]
        img.save(image_file[:-4] + '_dets.jpg')
        print('Saved: ' + image_file[:-4] + '_dets.jpg')


# /opt/caffe/models/VGGNet/TV_LOGO/SSD_TV_LOGO_300x300/
# /opt/caffe/data/VOCdevkit/TV_LOGO_Dataset
def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file', default='/opt/caffe/data/VOCdevkit/TV_LOGO_Dataset/labelmap_voc.prototxt')
    parser.add_argument('--model_def', default='/opt/caffe/models/VGGNet/TV_LOGO/SSD_TV_LOGO_300x300/deploy.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_weights', default='/opt/caffe/models/VGGNet/TV_LOGO/SSD_TV_LOGO_300x300/VGG_TV_LOGO_SSD_TV_LOGO_300x300_iter_21694.caffemodel')
    parser.add_argument('--image_file', default='/home/images/')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
