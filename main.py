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
import cv2
dtype = torch.FloatTensor
import math
from metric import SegmentationMetricTPFNFP, ROCMetric
from models.utils import *
from models import get_model
import os
import time
from argparse import ArgumentParser
import numpy as np


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implementation of NeurSTT')

    parser.add_argument('--model_name', type=str, default='NeurSTT', help='model name')
    #
    # Dataset parameters
    #
    parser.add_argument('--dataset', type=str, default='exp1', help='choose datasets')
    #
    # Training parameters
    #
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument('--w_decay', type=float, default=0.1, help='GPU number')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    #
    # Net parameters
    #
    parser.add_argument('--frame', type=int, default=80, help='value of L')
    parser.add_argument('--gamma', type=float, default=0.25, help='value of gamma')
    parser.add_argument('--kappa', type=int, default=1, help='value of kappa')
    parser.add_argument('--phi', type=float, default=5e-5, help='value of phi')
    parser.add_argument('--max_iter', type=int, default=1500, help='max iteration')

    args = parser.parse_args()

    if args.seed != 0:
        set_seeds(args.seed)

    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


###### metric
metric1 = SegmentationMetricTPFNFP(nclass=1)
metric2 = ROCMetric(nclass=1, bins=8)


###################
class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.frame = args.frame
        self.max_iter = args.max_iter
        self.w_decay = args.w_decay
        self.lr_real = args.lr
        self.phi = args.phi
        self.kappa = args.kappa
        self.gamma = args.gamma
        self.down = 4
        self.omega = 2
        self.data = args.dataset
        self.model_name = args.model_name
        self.image_path = os.path.join('./data/', self.data + '/images')
        self.label_path = os.path.join('./data/', self.data + '.gt')
        self.save_log_path = os.path.join('./result/', self.data, self.model_name)
        self.save_image_path = os.path.join('./result/', self.data, self.model_name + '/T')
        self.save_image_path_seg = os.path.join('./result/', self.data, self.model_name + '/Seg')
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        if not os.path.exists(self.save_image_path):
            os.makedirs(self.save_image_path)
        if not os.path.exists(self.save_log_path):
            os.makedirs(self.save_log_path)
        if not os.path.exists(self.save_image_path_seg):
            os.makedirs(self.save_image_path_seg)
        for k, v in sorted(vars(self.args).items()):
            print(k, '=', v)

    def training(self):
        # Load images
        imgList = sorted(os.listdir(self.image_path), key=lambda x: int(x.split('.')[0]))
        a = cv2.imread(os.path.join(self.image_path, imgList[0]), 0)
        m, n = a.shape
        length = math.floor(len(imgList) / self.frame)
        result_all = np.empty((m, n, 0))
        file_len = len(imgList)

        # Load labels (Evaluation Only)
        imgList2 = sorted(os.listdir(self.label_path), key=lambda x: int(x.split('.')[0]))
        labels = torch.zeros(m, n, file_len)
        for count2 in range(0, file_len):
            im_name2 = imgList2[count2]
            im_path2 = os.path.join(self.label_path, im_name2)
            labels[:, :, count2] = torch.from_numpy(cv2.imread(im_path2, 0))
        labels = torch.reshape(labels, (file_len, m, n)) / 255.0
        labels = labels.to(self.device)
        alltime = 0

        for i in range(length):


            img = torch.zeros(m, n, self.frame)

            for count in range(self.frame):
                im_name = imgList[count + self.frame * i]
                im_path = os.path.join(self.image_path, im_name)
                image = cv2.imread(im_path, 0) / 255
                img[:, :, count] = torch.from_numpy(image)


            X = img.to(self.device)
            n_1, n_2, n_3 = X.shape

            mid_channel = n_2
            r_1 = int(n_1 / self.down)
            r_2 = int(n_2 / self.down)
            r_3 = int(n_3 / self.down)

            mask = torch.ones(X.shape).to(self.device)
            mask[X == 0] = 0
            X[mask == 0] = 0

            U_input = torch.from_numpy(np.array(range(1, n_1 + 1))).reshape(1, n_1).type(dtype).to(self.device)
            V_input = torch.from_numpy(np.array(range(1, n_2 + 1))).reshape(1, n_2).type(dtype).to(self.device)
            W_input = torch.from_numpy(np.array(range(1, n_3 + 1))).reshape(1, n_3).type(dtype).to(self.device)

            U_input_tv = torch.from_numpy(np.array(range(1, n_1 + 1))).reshape(1, n_1).type(dtype).to(self.device)
            V_input_tv = torch.from_numpy(np.array(range(1, n_2 + 1))).reshape(1, n_2).type(dtype).to(self.device)
            W_input_tv = torch.from_numpy(np.array(range(1, n_3 + 1))).reshape(1, n_3).type(dtype).to(self.device)

            self.model = get_model(args.model_name,r_1, r_2, r_3, mid_channel)
            self.model = self.model.to(self.device)

            params = [x for x in self.model.parameters()]
            s = sum([np.prod(list(p.size())) for p in params])

            print('Number of params: %d' % s)
            T = torch.zeros(n_1, n_2, n_3).type(dtype).to(self.device)

            optimizer = optim.Adam(params, lr=self.lr_real, weight_decay=self.w_decay)

            start_time = time.time()
            for iter in range(self.max_iter):
                outputs = self.model(U_input, V_input, W_input, U_input_tv, V_input_tv, W_input_tv, self.omega)
                B, out_tv, dx, dy, dz = (tensor.to(self.device) for tensor in outputs)
                T = soft()(X - B, self.gamma / 2)

                dx = dx.unsqueeze(-1).unsqueeze(-1)
                dy = dy.unsqueeze(-1).unsqueeze(-1)
                dz = self.kappa * dz.unsqueeze(-1).unsqueeze(-1)
                du = torch.cat((dx, dy, dz), dim=3)
                loss_du = torch.norm(du, 1).to(self.device)

                loss_nuc = nuclear_norm(B,iter, self.frame).to(self.device)

                loss = loss_nuc
                loss = loss + torch.norm(T * mask, 1)
                loss = loss + self.phi * loss_du

                print("iter:{}, loss:{}".format(iter, loss))

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            img_T = T.cpu().detach().numpy() * 255.0
            img_T2 = Segmentation(img_T)
            result_all = np.concatenate((result_all, img_T2), axis=2)

            iii = 0
            for j in range(0, self.frame):
                img_name1 = str('{:0=3}'.format(j + self.frame * i))
                save_image_path1 = self.save_image_path + '/' + img_name1 + '.png'
                save_image_path2 = self.save_image_path_seg + '/' + img_name1 + '.png'
                imgs = img_T[:, :, iii] * 255
                imgs_seg = img_T2[:, :, iii] * 255
                iii += 1
                cv2.imwrite(save_image_path1, imgs)
                cv2.imwrite(save_image_path2, imgs_seg)
            end_time = time.time()
            tmp_time = end_time - start_time
            alltime = alltime + tmp_time

        alg_time = alltime / file_len
        print('time:%f' % alg_time)

        # Measure performance
        img_real3 = torch.from_numpy(result_all)
        labels = torch.reshape(labels, (file_len, m, n))
        img_real3 = torch.reshape(img_real3, (file_len, m, n))

        metric1.update(labels, img_real3)
        iou, _, _, fmeasure = metric1.get()
        print('fmeasure:%f' % fmeasure)
        print('miou:%f' % iou)

        metric_output = "fmeasure  : %f ; iou : %f;time : %f" % (
        fmeasure, iou, alg_time)
        metric_name = self.save_log_path + '/' + 'log.txt'
        with open(metric_name, "a+") as f:
            f.write(metric_output + '\n')
            f.close


# Example usage
if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.training()

