import os
import cv2
import json
import torch
from torch import nn
import logging
import detectron2
import numpy as np
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from food.dataloader import build_detection_test_loader
from food.evaluation.archs import resnet101
import random
from matplotlib import pyplot as plt
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PrototypicalCalibrationBlock:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.alpha = self.cfg.TEST.PCB_ALPHA

        self.imagenet_model = self.build_model()
        self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
        self.prototypes = self.build_prototypes()

        self.exclude_cls = self.clsid_filter()

    def build_model(self):
        logger.info("Loading ImageNet Pre-train Model from {}".format(self.cfg.TEST.PCB_MODELPATH))
        if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    def build_prototypes(self):

        all_features, all_labels = [], []
        for index in range(len(self.dataloader.dataset)):
            inputs = [self.dataloader.dataset[index]]
            assert len(inputs) == 1
            # load support images and gt-boxes
            img = cv2.imread(inputs[0]['file_name'])  # BGR
            img_h, img_w = img.shape[0], img.shape[1]
            ratio = img_h / inputs[0]['instances'].image_size[0]
            inputs[0]['instances'].gt_boxes.tensor = inputs[0]['instances'].gt_boxes.tensor * ratio
            boxes = [x["instances"].gt_boxes.to(self.device) for x in inputs]

            # extract roi features
            features = self.extract_roi_features(img, boxes)
            all_features.append(features.cpu().data)

            gt_classes = [x['instances'].gt_classes for x in inputs]
            all_labels.append(gt_classes[0].cpu().data)

        # concat
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        assert all_features.shape[0] == all_labels.shape[0]

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features[i].unsqueeze(0))

        # x = all_features
        # x_T = x.t()
        # x_ = torch.matmul(x_T, x) / 1000
        # x_att = nn.Softmax(dim=1)(x_)
        # # x_soft = torch.matmul(x, x_att)
        # # x_soft.shape()
        # # fc = nn.Linear(1000, 1000)
        # # nn.init.xavier_normal_(fc.weight)
        # # x_att = fc(x_soft)
        # # # dropout = nn.Dropout(x_att)
        # x = x + torch.matmul(x, x_att)
        # all_features = x

        # T-SNE
        feat = all_features.cpu().data.numpy()
        feat_label = all_labels.cpu().data.numpy()
        # feat = feat[150:]
        # feat_label = feat_label[150:]
        ts = TSNE(n_components=2, init='pca', random_state=0)
        feat = ts.fit_transform(feat)
        fig = self.plot_embedding(feat, feat_label, 'T-SNE of RoI features')
        plt.savefig('/home/subinyi/Users/DeFRCN-main/T-SNE/tsne5.svg', format='svg', bbox_inches='tight')
        # plt.show()
        # quit()

        prototypes_dict = {}
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)

        # b = []
        # for i in range(20):
        #     proto = prototypes_dict[i]
        #     a = np.array(proto)
        #     b = np.append(a,b)
        # b = np.reshape(b,[20, 1000])
        # x = b
        # x_T = x.T
        # x_T = torch.from_numpy(x_T)
        # x = torch.from_numpy(x)
        # x_ = torch.matmul(x_T, x) / 1000
        # x_att = nn.Softmax(dim=1)(x_)
        # # x_soft = torch.matmul(x, x_att)
        # # x_soft.shape()
        # # fc = nn.Linear(1000, 1000)
        # # nn.init.xavier_normal_(fc.weight)
        # # x_att = fc(x_soft)
        # # # dropout = nn.Dropout(x_att)
        # x = x + torch.matmul(x, x_att)
        # prototypes_dict_new = {}
        # for i in range(20):
        #     prototypes_dict_new[i] = torch.reshape(x[i,:], [1, 1000])
        # prototypes_dict = prototypes_dict_new
        return prototypes_dict # [20,1000]

    def plot_embedding(self, data, label, title, show=None):
        if show is not None:
            temp = [i for i in range(len(data))]
            random.shuffle(temp)
            data = data[temp]
            data = data[:show]
            label = torch.tensor(label)[temp]
            label = label[:show]
            label.numpy().tolist()

        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min)/(x_max - x_min)
        fig = plt.figure()

        data = data.tolist()
        label = label.squeeze().tolist()

        for i in range(len(data)):
            la = str(label[i])
            lz = label[i]
            if lz in [15, 16, 17, 18, 19]:
                plt.text(data[i][0], data[i][1], la, fontsize=6, backgroundcolor=plt.cm.tab10((label[i]-15)/10))
            # plt.plot(data[i][0], data[i][1])
        plt.title(title, fontsize=12)
        return fig

    def extract_roi_features(self, img, boxes):
        """
        :param img:
        :param boxes:
        :return:
        """

        mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
        std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255. - mean) / std]
        images = ImageList.from_tensors(images, 0)
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW [1, 2048, 11, 16]

        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2) # [1, 2048]

        activation_vectors = self.imagenet_model.fc(box_features) # [1, 1000]

        return activation_vectors

    def execute_calibration(self, inputs, dts):

        img = cv2.imread(inputs[0]['file_name'])

        ileft = (dts[0]['instances'].scores > self.cfg.TEST.PCB_UPPER).sum()
        iright = (dts[0]['instances'].scores > self.cfg.TEST.PCB_LOWER).sum()
        assert ileft <= iright
        boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]

        features = self.extract_roi_features(img, boxes) # [100, 1000]

        x = features
        x_T = x.t()
        x_ = torch.matmul(x_T, x)/1000
        x_att = nn.Softmax(dim=1)(x_)
        # x_soft = torch.matmul(x, x_att)
        # x_soft.shape()
        # fc = nn.Linear(1000, 1000)
        # nn.init.xavier_normal_(fc.weight)
        # x_att = fc(x_soft)
        # # dropout = nn.Dropout(x_att)
        x = x + torch.matmul(x, x_att)
        features = x

        for i in range(ileft, iright):
            tmp_class = int(dts[0]['instances'].pred_classes[i])
            if tmp_class in self.exclude_cls:
                continue
            tmp_cos1 = cosine_similarity(features[i - ileft].cpu().data.numpy().reshape((1, -1)),
                                        self.prototypes[tmp_class].cpu().data.numpy())
            tmp_cos = cosine_similarity(features[i - ileft].cpu().data.numpy().reshape((1, -1)),
                                        self.prototypes[tmp_class].cpu().data.numpy())[0][0]  # [100, 1000] [20, 1000]


            # protob = []
            # for ii in self.prototypes:
            #     proto = self.prototypes[ii].cpu().data.numpy()
            #     protob.append(proto)
            # protob = np.stack(protob).squeeze(axis=1)
            # protob = torch.from_numpy(protob)
            # proto_novelb = protob[15:, :]
            # proto_novel = proto_novelb.unsqueeze(axis=0)
            # proto_novel_att = self.ScaledDotProductAttention(proto_novel,proto_novel,proto_novel)[0]
            # prob = []
            # for j in range(5):
            #     proto_novelb[j] = torch.reshape(features[i - ileft], [1, 1000])
            #     query_feature = proto_novelb.unsqueeze(axis=0)
            #     query_feature_att = self.ScaledDotProductAttention(query_feature, query_feature, query_feature)[0]
            #     diff = torch.from_numpy((query_feature_att.cpu().data.numpy() - proto_novel_att.cpu().data.numpy())).pow(
            #         2).sum(-1).sum() / 5000
            #     prob.append(diff)
            # prob = torch.stack(prob)
            # score_ = nn.Softmax(dim=0)(prob)
            # # cls = torch.argmax(score_, dim=0)+15
            # # dts[0]['instances'].pred_classes[i] = cls
            # tmp_cos1 = score_[tmp_class - 15]
            # tmp_cos = tmp_cos1 + tmp_cos
            # self.alpha = 0.7
            # dts[0]['instances'].scores[i] = dts[0]['instances'].scores[i] * self.alpha + tmp_cos * (1 - self.alpha)

            '''coco'''
            # in_ids = [0,1,2,3,4,5,6,8,14,15,16,17,18,19,39,56,57,58,60,62]
            # prob = []
            # for j in in_ids:
            #     protob = torch.reshape(features[i - ileft], [1, 1000])
            #     diff = torch.from_numpy((protob.cpu().data.numpy()-self.prototypes[j].cpu().data.numpy())).pow(2).sum().sqrt()
            #     diff_cos = -20 * torch.from_numpy(cosine_similarity(protob.cpu().data.numpy(), self.prototypes[j].cpu().data.numpy()))[0][0]
            #     prob.append(diff_cos)
            # prob = torch.stack(prob)
            # score_ = nn.Softmax(dim=0)(-prob)
            # # cls = torch.argmax(score_, dim=0)+15
            # # dts[0]['instances'].pred_classes[i] = cls
            # ind = in_ids.index(tmp_class)
            # tmp_cos1 = score_[ind]
            # tmp_cos = tmp_cos1+tmp_cos
            # self.alpha = 0.7
            # dts[0]['instances'].scores[i] = dts[0]['instances'].scores[i] * self.alpha + tmp_cos * (1 - self.alpha)
            # if dts[0]['instances'].scores[i] >1:
            #     dts[0]['instances'].scores[i]=1

            '''voc'''
            in_ids = [15,16,17,18,19]
            prob = []
            for j in in_ids:
                protob = torch.reshape(features[i - ileft], [1, 1000])
                diff = torch.from_numpy((protob.cpu().data.numpy() - self.prototypes[j].cpu().data.numpy())).pow(
                    2).sum().sqrt()
                diff_cos = -20*torch.from_numpy(cosine_similarity(protob.cpu().data.numpy(), self.prototypes[j].cpu().data.numpy()))[0][0]
                prob.append(diff_cos)
            prob = torch.stack(prob)
            score_ = nn.Softmax(dim=0)(-prob)
            # cls = torch.argmax(score_, dim=0)+15
            # dts[0]['instances'].pred_classes[i] = cls
            ind = in_ids.index(tmp_class)
            tmp_cos1 = score_[ind]
            tmp_cos = tmp_cos1 + tmp_cos
            self.alpha = 0.3
            dts[0]['instances'].scores[i] = (1 - self.alpha)*dts[0]['instances'].scores[i] + self.alpha * tmp_cos
            # if dts[0]['instances'].scores[i] >1:
            #     dts[0]['instances'].scores[i]=1

            # dts[0]['instances'].scores[i] = dts[0]['instances'].scores[i] * self.alpha + tmp_cos * (1 - self.alpha)
        return dts

    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                               66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids


    def ScaledDotProductAttention(self, q, k, v):
        temperature = 1000
        # attn_dropout = 0.1
        attn = torch.bmm(k.transpose(1, 2), q)
        attn = attn / temperature
        raw_attn = attn
        log_attn = F.log_softmax(attn, 2)
        attn = nn.Softmax(dim=2)(attn)
        # attn = nn.Dropout(attn_dropout)(attn)
        output = torch.bmm(v, attn)
        return output, attn, log_attn, raw_attn

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
