import os
from tqdm import tqdm
from utils.config import opt
import pdb
import torch
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.autograd import Variable
from trainer import FasterRCNNTrainer
from torch.utils import data as data_
import cv2
from utils import array_tool as at
import numpy as np
from model.faster_rcnn_vgg16 import *


# def visavis(i):
# 	image, bboxs, _, _ = loader[i]
# 	image = np.transpose(image, (1,2,0)).astype(np.uint8)
# 	img = img = np.zeros(image.shape, np.uint8)
# 	bboxs = [list(map(int, bbox)) for bbox in bboxs]
# 	print(len(bboxs))
# 	for bbox in bboxs:
# 		print(bbox)
# 		ymin, xmin, ymax, xmax = bbox
# 		cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
# 	cv2.imwrite("my.png", img)


# visavis(0)
def to_tensor(x):
	if type(x) is np.ndarray:
		return torch.from_numpy(x)
	return x
def train(**kwargs):
	opt._parse(kwargs)
	dataset = Dataset(opt)
	print('load data')
	dataloader = data_.DataLoader(dataset, \
								batch_size=1, \
								shuffle=True, \
								# pin_memory=True,
								num_workers=opt.num_workers)
	print('Loading Model')
	# faster_rcnn = FasterRCNNVGG16()
	print('model construct completed')
	# trainer = FasterRCNNTrainer(faster_rcnn).cuda()
	lr_ = opt.lr
	extractor, classifier = decom_vgg16()
	img, bbox_, label_, scale = dataset[1]
	_, H, W = img.shape
	img_size = (H, W)
	img, bbox_, label_ = to_tensor(img), to_tensor(bbox_), to_tensor(label_)
	scale = at.scalar(scale)
	img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
	img, bbox, label = Variable(img), Variable(bbox), Variable(label)
	pdb.set_trace()
	features = extractor(img)
	
	rpn = RegionProposalNetwork(
								512, 512,
								ratios=ratios,
								anchor_scales=anchor_scales,
								feat_stride=self.feat_stride)

	rpn_locs, rpn_scores, rois, roi_indices, anchor = \
									self.faster_rcnn.rpn(features, img_size, scale
									)
	

train()