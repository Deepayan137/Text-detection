import os
import warnings
import ipdb
import matplotlib
from tqdm import *
import pdb
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import cv2
import numpy as np
warnings.filterwarnings("ignore")
matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def write_image(image, bboxes,image_name):
  image = image.transpose((1, 2, 0)).astype(np.uint8)
  img = image.copy()
  for bbox in bboxes:
    ymin, xmin, ymax, xmax = bbox
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
  cv2.imwrite(image_name, img)


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)

            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                # ori_img_ = (at.tonumpy(img[0]))
                losses = trainer.get_meter_data()
                print(losses)
                write_image(ori_img_, at.tonumpy(bbox[0]), 'gt.png')
                _bboxes = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                _bboxes = at.tonumpy(_bboxes[0])
                # plot predicted bboxes
                write_image(ori_img_, _bboxes, 'pred.png')
                print('saved an image')
                
        if epoch == 13: 
            break

if __name__ == '__main__':
    train()