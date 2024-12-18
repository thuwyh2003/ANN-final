import logging
from PIL import Image
import os

import jittor as jt
from jittor.dataset.dataset import Dataset
from jittor.transform import Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, CenterCrop

from .dataset_jt import CUB, CarsDataset, NABirds, dogs, INat2017
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        jt.sync_all()  # 替代 PyTorch 的 barrier

    def build_transform(is_train, resize, crop, autoaug=False):
        transforms = [Resize((resize, resize), Image.BILINEAR)]
        if is_train:
            transforms += [RandomCrop((crop, crop)), RandomHorizontalFlip()]
            if autoaug:
                transforms.append(AutoAugImageNetPolicy())
        else:
            transforms.append(CenterCrop((crop, crop)))
        transforms += [
            ToTensor(),
        ]
        return Compose(transforms)

    if args.dataset == 'CUB_200_2011':
        train_transform = build_transform(True, 600, 448)
        test_transform = build_transform(False, 600, 448)
        train_loader = CUB(root=args.data_root, is_train=True, transform=train_transform,shuffle=True, batch_size=args.train_batch_size)
        test_loader = CUB(root=args.data_root, is_train=False, transform=test_transform,shuffle=False,batch_size=args.eval_batch_size)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root, 'devkit/cars_train_annos.mat'),
                               os.path.join(args.data_root, 'cars_train'),
                               os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                               transform=build_transform(True, 600, 448, autoaug=True))
        testset = CarsDataset(os.path.join(args.data_root, 'cars_test_annos_withlabels.mat'),
                              os.path.join(args.data_root, 'cars_test'),
                              os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                              transform=build_transform(False, 600, 448))
    elif args.dataset == 'dog':
        train_transform = build_transform(True, 600, 448)
        test_transform = build_transform(False, 600, 448)
        trainset = dogs(root=args.data_root, train=True, cropped=False, transform=train_transform)
        testset = dogs(root=args.data_root, train=False, cropped=False, transform=test_transform)
    elif args.dataset == 'nabirds':
        train_transform = build_transform(True, 600, 448)
        test_transform = build_transform(False, 600, 448)
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform = build_transform(True, 400, 304, autoaug=True)
        test_transform = build_transform(False, 400, 304)
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)

    if args.local_rank == 0:
        jt.sync_all()

    # train_loader = trainset.set_attrs(
    #     batch_size=args.train_batch_size,
    #     shuffle=(args.local_rank == -1),
    #     num_workers=4,
    #     drop_last=True
    # )
    # test_loader = testset.set_attrs(
    #     batch_size=args.eval_batch_size,
    #     shuffle=False,
    #     num_workers=4
    # ) if testset is not None else None

    return train_loader, test_loader
    # return trainset,testset