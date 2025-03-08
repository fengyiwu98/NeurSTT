"""
NeurSTT: Neural Spatial-Temporal Tensor Representation for Infrared Small Target Detection

This code is part of the official implementation of the paper:
"Neural Spatial-Temporal Tensor Representation for Infrared Small Target Detection"

Paper Source: https://arxiv.org/abs/2412.17302
Authors: Fengyi Wu, Simin Liu, Haoan Wang, Bingjie Tao, Junhai Luo, Zhenming Peng

Contact Information:
Fengyi Wu
Email: wufengyi98@163.com
"""
import torch
from torch import nn, optim 
dtype = torch.FloatTensor
import numpy as np 
import os
import cv2
from metric import SegmentationMetricTPFNFP
from metric import ROCMetric

#######Parameters#######
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datas = ['exp1']
thres = 0.4
frame = 160 #You can adjust it according to your sequence length
alg_list = ['NeurSTT']

#######Evaluation#######
for data in datas:

    txt_path = os.path.join('./result', data )
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    for alg in alg_list:

        image_path = os.path.join('./result', data , alg, './Seg')
        label_path = './data/' + data+ '.gt'


        ###### metric
        metric1 = SegmentationMetricTPFNFP(nclass=1)
        metric2 = ROCMetric(nclass=1, bins=8)



        ###load image and lable

        # get image  and size
        dir = image_path
        imgList = os.listdir(dir)
        a = cv2.imread(os.path.join(dir, imgList[0]), 0)
        m, n = a.shape
        images = torch.zeros(m, n, frame)
        for count in range(0, frame):
            im_name = imgList[count]
            im_path = os.path.join(dir, im_name)
            img = cv2.imread(im_path, 0)
            ##segment
            if alg != 'NeurSTT':
                maxvalue = np.max(img)
                thresh = thres * maxvalue
                img[img >= thresh] = 255
                img[img <thresh] = 0
            # ##get images
            images[:, :, count] = torch.from_numpy(img)/255.0   



        # # get lable 
        imgList2 = os.listdir(label_path)
        labels = torch.zeros_like(images)
        for count2 in range(0, frame):
            im_name2 = imgList2[count2+0]
            im_path2 = os.path.join(label_path, im_name2)
            mask = cv2.imread(im_path2, 0)
            labels[:, :, count2] = torch.from_numpy(cv2.imread(im_path2, 0))
        labels = torch.reshape(labels, (frame,m , n)) / 255.0
        labels = labels.to(device)

        ####measure
        labels = torch.reshape(labels, (frame, m, n))
        img_real3 = torch.reshape(images, (frame, m, n))

        metric1.update(labels, img_real3)
        miou, prec, recall, fmeasure = metric1.get()
        print('fmeasure:%f'% fmeasure)
        print('precision:%f'%prec)
        print('recall:%f'%recall)
        print('miou:%f'%miou)

        metric_output = "algorithm: %s; fmeasure  : %f ;miou : %f;  precision: %f ; recall : %f" % (
        alg, fmeasure,miou, prec, recall)
        metric_name = txt_path + '/' + 'eval_log.txt'
        with open(metric_name, "a+") as f:
            f.write(metric_output + '\n')
            f.close

    with open(metric_name, "a+") as f:
        f.write("-------------------------------------------------------------" + '\n')
        f.close
