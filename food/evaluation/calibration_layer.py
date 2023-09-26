import os
import cv2
import json
import torch
import logging
import detectron2
import numpy as np
import random
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from sklearn.metrics.pairwise import cosine_similarity
from food.dataloader import build_detection_test_loader
from food.evaluation.archs import resnet101

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
        self.prototypes_1, self.prototypes_2, self.prototypes_3, self.prototypes_4 = self.build_prototypes()

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

        all_features_1, all_features_2, all_features_3, all_features_4, all_labels = [], [], [], [], []
        for index in range(len(self.dataloader.dataset)):
            inputs = [self.dataloader.dataset[index]]
            assert len(inputs) == 1
            # load support images and gt-boxes
            img = cv2.imread(inputs[0]['file_name'])  # BGR
            img_h, img_w = img.shape[0], img.shape[1]
            ratio = img_h / inputs[0]['instances'].image_size[0]
            a = inputs[0]['instances'].gt_boxes
            b = inputs[0]['instances'].gt_boxes.tensor
            inputs[0]['instances'].gt_boxes.tensor = inputs[0]['instances'].gt_boxes.tensor * ratio
            boxes = [x["instances"].gt_boxes.to(self.device) for x in inputs]

            # extract roi features
            features = self.extract_roi_features(img, boxes)
            all_features_4.append(features[0].cpu().data)
            all_features_3.append(features[1].cpu().data)
            all_features_2.append(features[2].cpu().data)
            all_features_1.append(features[3].cpu().data)

            gt_classes = [x['instances'].gt_classes for x in inputs]
            all_labels.append(gt_classes[0].cpu().data)

        # concat
        all_features_4 = torch.cat(all_features_4, dim=0)
        all_features_3 = torch.cat(all_features_3, dim=0)
        all_features_2 = torch.cat(all_features_2, dim=0)
        all_features_1 = torch.cat(all_features_1, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        assert all_features_4.shape[0] == all_labels.shape[0]
        assert all_features_3.shape[0] == all_labels.shape[0]
        assert all_features_2.shape[0] == all_labels.shape[0]
        assert all_features_1.shape[0] == all_labels.shape[0]

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features_4[i].unsqueeze(0))

        prototypes_dict_4 = {}
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict_4[label] = torch.mean(features, dim=0, keepdim=True)

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features_3[i].unsqueeze(0))

        prototypes_dict_3 = {}
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict_3[label] = torch.mean(features, dim=0, keepdim=True)

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features_2[i].unsqueeze(0))

        prototypes_dict_2 = {}
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict_2[label] = torch.mean(features, dim=0, keepdim=True)

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features_1[i].unsqueeze(0))

        prototypes_dict_1 = {}
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict_1[label] = torch.mean(features, dim=0, keepdim=True)

        # T-SNE
        # j = 8
        # i = 1
        # for layer_feat in [all_features_1,all_features_2,all_features_3,all_features_4]:
        #     feat = layer_feat.cpu().data.numpy()
        #     feat_label = all_labels.cpu().data.numpy()
        #     ts = TSNE(n_components=2, init='pca', random_state=0)
        #     feat = ts.fit_transform(feat)
        #     fig = self.plot_embedding(feat, feat_label, 'T-SNE of layer ' r'$C_{}$'.format(i+1))
        #     fig_path ='/home/subinyi/Users/RDD-main/DeFRCN-main/TSNE_figure/tsne{}_{}.svg'.format(j,i)
        #     plt.savefig(fig_path, format='svg', bbox_inches='tight')
        #     i+=1
        # quit()

        return prototypes_dict_1, prototypes_dict_2, prototypes_dict_3, prototypes_dict_4

    def plot_embedding(self, data, label, title, show=90):
        if show is not None:
            # temp = [i for i in range(len(data))]
            # random.shuffle(temp)
            # data = data[temp]
            ids=[]
            for id_cls in [0,1,2,3,4,5,6,9,12]:
                argid = np.where(label==id_cls)
                ids = np.append(ids, argid)
            ids = ids.astype('int32')
            label = label[ids]
            label[label == 9] = 7
            label[label == 12] = 8
            data = data[ids]
            label = torch.tensor(label)
            # label = label[:show]
            label.numpy().tolist()


        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min)/(x_max - x_min)
        fig = plt.figure()

        data = data.tolist()
        label = label.squeeze().tolist()
        import seaborn as sns
        plt.rcParams.update({'font.size': 18})
        sns.set(rc={'figure.figsize': (8, 5)})
        palette = sns.color_palette("bright", 9)
        data = np.array(data)
        ax = sns.scatterplot(data[:, 0], data[:, 1], hue=label, legend='full', palette=palette,s=300)
        plt.setp(ax.get_legend().get_texts(), fontsize='18')  # for legend text
        plt.title(title, fontsize=24)
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
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW

        # cov_feature4 = self.imagenet_model(images.tensor[:, [2, 1, 0]])['layer4']
        box_features_layer1 = self.roi_pooler([conv_feature[0]],boxes).squeeze(2).squeeze(2)
        box_features_layer2 = self.roi_pooler([conv_feature[1]],boxes).squeeze(2).squeeze(2)
        box_features_layer3 = self.roi_pooler([conv_feature[2]],boxes).squeeze(2).squeeze(2)
        box_features_layer4 = self.roi_pooler([conv_feature[3]],boxes).squeeze(2).squeeze(2)
        box_features_layer4 = self.imagenet_model.fc(box_features_layer4)
        activation_vectors = list([box_features_layer4, box_features_layer3, box_features_layer2, box_features_layer1])

        # box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)
        # activation_vectors = self.imagenet_model.fc(box_features)
        return activation_vectors

    def execute_calibration(self, inputs, dts):

        img = cv2.imread(inputs[0]['file_name'])

        ileft = (dts[0]['instances'].scores > self.cfg.TEST.PCB_UPPER).sum()
        iright = (dts[0]['instances'].scores > self.cfg.TEST.PCB_LOWER).sum()
        assert ileft <= iright
        boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]

        features = self.extract_roi_features(img, boxes)

        for i in range(ileft, iright):
            tmp_class = int(dts[0]['instances'].pred_classes[i])
            if tmp_class in self.exclude_cls:
                continue
            tmp_cos_1 = cosine_similarity(features[3][i - ileft].cpu().data.numpy().reshape((1, -1)),
                                          self.prototypes_1[tmp_class].cpu().data.numpy())[0][0]
            tmp_cos_2 = cosine_similarity(features[2][i - ileft].cpu().data.numpy().reshape((1, -1)),
                                          self.prototypes_2[tmp_class].cpu().data.numpy())[0][0]
            tmp_cos_3 = cosine_similarity(features[1][i - ileft].cpu().data.numpy().reshape((1, -1)),
                                        self.prototypes_3[tmp_class].cpu().data.numpy())[0][0]
            tmp_cos_4 = cosine_similarity(features[0][i - ileft].cpu().data.numpy().reshape((1, -1)),
                                        self.prototypes_4[tmp_class].cpu().data.numpy())[0][0]
            # tmp_cos = (tmp_cos_1 + tmp_cos_2 + tmp_cos_3 + tmp_cos_4)/4
            tmp_cos = tmp_cos_4
            dts[0]['instances'].scores[i] = dts[0]['instances'].scores[i] * self.alpha + tmp_cos * (1 - self.alpha)
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
                exclude_ids = [list(range(0, 15)), 20,21,22,23]
            else:
                raise NotImplementedError
        return exclude_ids


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
