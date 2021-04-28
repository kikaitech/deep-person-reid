from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys

import torchreid

from torchreid.data import ImageDataset
import os
import os.path as osp
import torch
import torch.nn as nn


query_data = []
train_data = []
gallery_data = []

path_query = "/home/kikai/Desktop/nobi/deep-person-reid/data/nobi/query/"
path_train = "/home/kikai/Desktop/nobi/deep-person-reid/data/nobi/train/"
path_gallery = "/home/kikai/Desktop/nobi/deep-person-reid/data/nobi/gallery/"

for i in os.listdir(path_query):
    data = i.split("_")
    path = path_query + i
    camera_id = data[1]
    person_id = data[2]
    query_data.append((path, person_id, camera_id))

for i in os.listdir(path_train):
    data = i.split("_")
    path = path_train + i
    camera_id = data[1]
    person_id = data[2]
    train_data.append((path, person_id, camera_id))

for i in os.listdir(path_gallery):
    data = i.split("_")
    path = path_gallery + i
    camera_id = data[1]
    person_id = data[2]
    gallery_data.append((path, person_id, camera_id))


class NobiDataset(ImageDataset):
    dataset_dir = 'nobi'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        print(self.dataset_dir)

        train = train_data
        query = query_data
        gallery = gallery_data

        super(NobiDataset, self).__init__(train, query, gallery, **kwargs)


torchreid.data.register_image_dataset('nobi', NobiDataset)

torch.backends.cudnn.benchmark = True
datamanager = torchreid.data.ImageDataManager(
    root='/home/kikai/Desktop/nobi/deep-person-reid/data/nobi',
    sources='nobi',
    height=256,
    width=256,
    batch_size_train=32,
    batch_size_test=32,
    use_gpu=True
)


model = torchreid.models.build_model(
    name='osnet_x0_25',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    use_gpu=True
)

model = nn.DataParallel(model).cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='amsgrad',
    lr=0.0015
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=40
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir='log/nobi',
    max_epoch=300,
    eval_freq=10,
    print_freq=10
)
