import jittor as jt
import logging
import os
from PIL import Image
from jittor import transform
from jittor.dataset import Dataset, DataLoader
from .dataset_jt import CUB, CarsDataset, NABirds, dogs, INat2017
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)
class Normalize(object):
    """Normalize the image by subtracting mean and dividing by std."""
    def __init__(self, mean, std):
        self.mean = jt.array(mean)
        self.std = jt.array(std)

    def __call__(self, tensor):
        return (tensor - self.mean[None, :, None, None]) / self.std[None, :, None, None]
def get_loader(args):
    if args.local_rank not in [-1, 0]:
        jt.sync_all()

    if args.dataset == 'CUB_200_2011':
        train_transform = transform.Compose([
            transform.Resize((600, 600)),
            transform.RandomCrop((448, 448)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transform.Compose([
            transform.Resize((600, 600)),
            transform.CenterCrop((448, 448)),
            transform.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'car':
        trainset = CarsDataset(
            os.path.join(args.data_root, 'devkit/cars_train_annos.mat'),
            os.path.join(args.data_root, 'cars_train'),
            os.path.join(args.data_root, 'devkit/cars_meta.mat'),
            transform=transform.Compose([
                transform.Resize((600, 600)),
                transform.RandomCrop((448, 448)),
                transform.RandomHorizontalFlip(),
                AutoAugImageNetPolicy(),
                transform.ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
        testset = CarsDataset(
            os.path.join(args.data_root, 'cars_test_annos_withlabels.mat'),
            os.path.join(args.data_root, 'cars_test'),
            os.path.join(args.data_root, 'devkit/cars_meta.mat'),
            transform=transform.Compose([
                transform.Resize((600, 600)),
                transform.CenterCrop((448, 448)),
                transform.ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )

    elif args.dataset == 'dog':
        train_transform = transform.Compose([
            transform.Resize((600, 600)),
            transform.RandomCrop((448, 448)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transform.Compose([
            transform.Resize((600, 600)),
            transform.CenterCrop((448, 448)),
            transform.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = dogs(root=args.data_root, train=True, cropped=False, transform=train_transform, download=False)
        testset = dogs(root=args.data_root, train=False, cropped=False, transform=test_transform, download=False)

    elif args.dataset == 'nabirds':
        train_transform = transform.Compose([
            transform.Resize((600, 600)),
            transform.RandomCrop((448, 448)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transform.Compose([
            transform.Resize((600, 600)),
            transform.CenterCrop((448, 448)),
            transform.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)

    elif args.dataset == 'INat2017':
        train_transform = transform.Compose([
            transform.Resize((400, 400)),
            transform.RandomCrop((304, 304)),
            transform.RandomHorizontalFlip(),
            AutoAugImageNetPolicy(),
            transform.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transform.Compose([
            transform.Resize((400, 400)),
            transform.CenterCrop((304, 304)),
            transform.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)

    if args.local_rank == 0:
        jt.sync_all()

    # Data samplers
    # train_sampler = jt.data.RandomSampler(trainset) if args.local_rank == -1 else jt.data.DistributedSampler(trainset)
    # test_sampler = jt.data.SequentialSampler(testset) if args.local_rank == -1 else jt.data.DistributedSampler(testset)

    # Create DataLoader instances
    train_loader = DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        # pin_memory=True
    )

    test_loader = DataLoader(
        testset,
        batch_size=args.eval_batch_size,
        shuffle = False,
        num_workers=0,
        # pin_memory=True
    ) if testset is not None else None

    return trainset, testset
