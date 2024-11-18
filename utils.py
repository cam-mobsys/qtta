import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import tensorflow_datasets as tfds
import random, math
from PIL import Image
from copy import deepcopy
import argparse
import os, sys

from tinyml.tinytl.tinytl.utils import profile_memory_cost
from RealisticTTA.methods.norm import Norm
from RealisticTTA.conf import cfg as ttbn_cfg
# from run_all_cifar10 import BalancedBatchNorm
from new_runs import BalancedBatchNorm
from tent import tent
from EATA import eata
import FOA.tta_library.sar as sar
import cotta.cifar.cotta as cotta
# import FOA.tta_library.cotta as cotta
from FOA.tta_library.sam import SAM
from FOA.tta_library.lame import LAME
from typing import Dict, Type, Any, Callable, Union, List, Optional, Tuple


BENCHMARK_CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'frosted_glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic',
    'pixelate',
    'jpeg_compression',
    'gaussian_blur',
    'saturate',
    'spatter',
    'speckle_noise',
    # 'none'
]

def calculate_acc(model, dataloader, stop_idx):
    count_right, count_seen = 0, 0
    for idx, (img, lbl) in enumerate(dataloader):
        if idx > stop_idx:
            break
        output = model(img)
        for o, l in zip(output, lbl):
            if o.argmax() == l:
                count_right += 1
            count_seen += 1
    return  count_right, count_seen


class CorruptCIFAR(torchvision.datasets.VisionDataset):
    """Corrupt`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This adds corruptions

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (Callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (Callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(
        self,
        root: str,
        corruption_level: str,
        train: bool,
        download: bool,
        split: int,
        seed: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        kwargs: Optional[Callable] = None,
    ) -> None:
        super(CorruptCIFAR, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.train = False
        base_folder = "data/cifar100c/CIFAR100-C"

        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        labels = list(np.load(os.path.join(root, base_folder, "labels.npy")))
        # for corr in corruptions:
        self.corr_num = int(corruption_level.split("_")[-1])
        corr = corruption_level.split(f"_{self.corr_num}")[0]
        file_path = os.path.join(root, base_folder, "{}.npy".format(corr))
        # start,end = (self.corr_num-1)*10000+(split*seed), min((self.corr_num-1)*10000+((seed+1)*split), self.corr_num*10000)
        # self.data.append(np.load(file_path)[((self.corr_num-1)*10000+(split*seed)):min((self.corr_num-1)*10000+((seed+1)*split), self.corr_num*10000)])
        start, end = (self.corr_num-1)*10000, (self.corr_num-1)*10000 + 100
        self.data.append(np.load(file_path)[start:end])
        # raise Exception(len(np.load(file_path)))
        self.targets.extend(labels[start:end])

        self.data = np.vstack(self.data)
        # self.classes = eval_utils.cifar100_classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # """
        # Args:
        #     index (int): Index

        # Returns:
        #     tuple: (image, target) where target is index of the target class.
        # """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR10CorruptDataset(Dataset):
    """Corrupted CIFAR10C dataset."""

    def __init__(self, corruption_category_name, split=4000, transform=None):
        """
        Arguments:
            corruption_category_name (string): tfds name for the corruption to be downloaded.
            split (array of strings): dictates whether this is a train or test split
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if split == "train":
            real_split = ["test[:60%]"]
        elif split == "val":
            real_split = ["test[60%:]"]
        else:
            real_split = ["test"]
        
        self.ds_name = corruption_category_name
        self.ds = list(tfds.load(f"cifar10_corrupted/{corruption_category_name}", split=real_split, shuffle_files=True)[0])[-100:]
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            raise NotImplementedError

        item = self.ds[idx]
        img = Image.fromarray(np.array(item["image"]))
        label = torch.tensor(item["label"].numpy())

        if self.transform:
            img = self.transform(img)

        return [img, label]


class ImageTransform(dict):
    def __init__(self, num_classes=10):
        super().__init__({
            # 'train': self.build_train_transform(),
            'val': self.build_val_transform(num_classes)
        })

    def build_val_transform(self, num_classes):
        if True:
            if num_classes == 10:
                norm_t = transforms.Normalize((0.4194, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
            elif num_classes == 100:
                norm_t = transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            return transforms.Compose([
                transforms.ToTensor(),
                norm_t
            ])


def get_args():
    """
    Inspired by code at https://github.com/mr-eggplant/SAR/main.py
    """
    
    parser = argparse.ArgumentParser(description='Experimental Settings')

    # dataloader
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size for testing.')
    parser.add_argument('--do_shuffle', default=True, type=bool, help='whether to shuffle (True) or maintain existing order (False)')
    parser.add_argument('--data_size', default=1000, type=int, help='how many instances to allow, per dataset')
    parser.add_argument('--class_num', default='10', type=str, help='only valid for cifar10/cifar100')

    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # experimenta settings
    parser.add_argument('--precision', default='fp32', type=str, choices=['int8', 'fp32'], help='model precision. do not set to int8 unless using our method.')
    parser.add_argument('--update_cadence', default=1, type=int, help='update every n data instances')
    parser.add_argument('--repeats', default=1, type=int, help='number of different images to try if one_time is True')
    parser.add_argument('--seed', default=42, type=int, help='random seed setting')
    parser.add_argument('--tta', default=None, type=str, help='Form of adaptation to use', choices=['none', 'eata', 'ours', 'tema', 'tent', 'sar', 'cotta', "quantnone", "quantours"])
    parser.add_argument('--model', default='resnet18', type=str, help='Models to choose from. For birds, only vggish is supported.', choices=['vggish', 'mobilenetv2', 'resnet18'])
    parser.add_argument('--exp_type', default='gradual', type=str, help='episodic, mix_shifts, label_shifts')
    parser.add_argument('--stop_layer', default=1000, type=int, help="for layer ablation experiments")

    parser.add_argument('--use_cuda', action='store_true', help="Whether or not to use CUDA")
    parser.add_argument('--warmup', default=0, type=int, help="How many data instances to use for warmup")
    parser.add_argument('--save', action='store_true', help="save results?")
    return parser.parse_args()


def prep_model(model, adaptation_method, args=None, tau=0.9, averager=0.9):
    if adaptation_method == 'ours' or adaptation_method == 'quantours':
        model = BalancedBatchNorm.retrofit_model(model, tau=tau, avgr=averager, stop_layer=args['stop_layer'])
        model.eval()
        # mem=calculate_model_full_memory(model, "test", (1,3,32,32), None, 1, args["precision"])
    elif adaptation_method == 'tema':
        model = Norm(cfg=ttbn_cfg, model=model, num_classes=args['class_num'])
        mem = 0
        # mem = calculate_model_full_memory(model, "test", (1,3,32,32), None, 1, args["precision"])
    elif adaptation_method == 'tent':
        model = tent.configure_model(model)
        params, _ = tent.collect_params(model)
        optimizer = torch.optim.Adam(params, 
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0)
        model = tent.Tent(model, optimizer)
        # mem = calculate_model_full_memory(model, "train", (1,3,32,32),optimizer, args["batch_size"], args["precision"])
    elif adaptation_method == 'eata':
        # compute fisher informatrix
        print(args["class_num"])
        if args["class_num"] == "10":
            fisher_dataset = torchvision.datasets.CIFAR10("data/cifar10", train=False,
                                          transform=ImageTransform(num_classes=10)['val'], download=True)
        elif args["class_num"] == "100":
            fisher_dataset = torchvision.datasets.CIFAR100("data/cifar100", train=False, transform=ImageTransform(num_classes=100)['val'], download=True)
        fisher_loader = torch.utils.data.DataLoader(
            fisher_dataset,
            shuffle=True,
            batch_size=64,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        fisher_loader_limit = 300

        model = eata.configure_model(model)
        params, param_names = eata.collect_params(model)
        # No momentum (just SGD)
        try:
            fishers = np.load(f"{args['model']}_eata_fishers_cifar{args['class_num']}.npy", allow_pickle=True)
        except:
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
                if iter_ > fisher_loader_limit:
                    break
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    model = model.cuda()
                    targets = targets.cuda(non_blocking=True)
                outputs = model(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == fisher_loader_limit:
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            del ewc_optimizer
            np.save(f"{args['model']}_eata_fishers_cifar{args['class_num']}.npy", fishers, allow_pickle=True)

        optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
        model = eata.EATA(model, optimizer, fishers, fisher_alpha=2000, e_margin=0.4*math.log(int(args['class_num'])), d_margin=0.05)
        # mem = calculate_model_full_memory(model, "train", (1,3,32,32), optimizer, args["batch_size"], args["precision"])
    elif adaptation_method == 'sar':
        print("SAR does not work well with BN and batch size < 10")
        model = sar.configure_model(model)
        params, _ = sar.collect_params(model)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=0.001, momentum=0.9)
        model = sar.SAR(model, optimizer, margin_e0=0.5*math.log(10))
        mem = calculate_model_full_memory(model, "train", (1,3,32,32), optimizer, args["batch_size"], args["precision"])
    elif adaptation_method == 'cotta':
        model = cotta.configure_model(model)
        params, _ = cotta.collect_params(model)
        base_optimizer = torch.optim.SGD(params, 
                                         lr=0.001, 
                                         momentum=0.9, 
                                         dampening=0., 
                                         weight_decay=0.,
                                         nesterov=True)
        model = cotta.CoTTA(model, 
                            base_optimizer,
                            steps=1, 
                            episodic=False, 
                            mt_alpha=0.999, 
                            rst_m=0.01,
                            ap=0.92)
        # mem = f"{calculate_model_full_memory(model, 'train', (1,3,32,32), base_optimizer, args['batch_size'], args['precision'])} but add another model"
    elif adaptation_method == 'none' or adaptation_method == 'quantnone':
        mem = 0
    else:
        raise Exception(f"{adaptation_method} not implemented.")
    # print(mem)
    return model



def generate_dataset(args):
    """
    Sets up datasets for different types of experiments.
    Gradual is the same setup as in RealisticTTA, but with randomly shuffled corruption types.
    Mixed shifts is the same setup as in SAR.
    """
    # args = vars(args)
    # random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    
    corruption_sequence = deepcopy(BENCHMARK_CORRUPTIONS)
    if args['exp_type'] == 'label_shifts':
        raise Exception("TODO: Label shifts >:(")
    elif args['exp_type'] == 'mix_shifts' or args['exp_type'] == 'search' or args['exp_type'] == 'layer':
        shift_sequence = [1,2,3,4,5]
        random.shuffle(corruption_sequence)
    elif args['exp_type'] == 'gradual':
        shift_sequence = [1,2,3,4,5,5,4,3,2,1]
        random.shuffle(corruption_sequence)
    elif args['exp_type'] == 'episodic':
        shift_sequence = [args['level']]
        corruption_sequence = [args['corruption']]
    else:
        raise Exception("Unsupported experiment type.")
    file_name = f"{args['exp_type']}_cifar{args['class_num']}_dset.pth"
    # file_name = f"mix_shifts_cifar{args['class_num']}_dset.pth"
    try:
        combined_dataset = torch.load(file_name, map_location=torch.device("cpu"))
        print("Loaded dset from memory")
    except Exception as e:
        datasets = []
        print(shift_sequence, corruption_sequence)
        for corruption_name in corruption_sequence:
            if False:# corruption_name == 'none':
                if args["class_num"] == "10":
                    dset = torchvision.datasets.CIFAR10("data/cifar10", train=False, transform=ImageTransform(num_classes=10)['val'], download=True)
                elif args["class_num"] == "100":
                    dset = torchvision.datasets.CIFAR100("data/cifar100", train=False, transform=ImageTransform(num_classes=100)['val'], download=True)
                datasets.append(dset)
            else:
                for shift in shift_sequence:
                    print(args['class_num'], corruption_name, shift, args['data_size'])
                    if args["class_num"] == "10":
                        dset = CIFAR10CorruptDataset(f"{corruption_name}_{shift}", split=args["seed"], transform=ImageTransform(num_classes=10)["val"])
                    elif args["class_num"] == "100":
                        c100_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
                        ])
                        dset = CorruptCIFAR("", corruption_name + "_" + str(shift), train=False, download=False, split=args['data_size'], transform=c100_transform, seed=args["seed"])
                    datasets.append(dset)
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        torch.save(combined_dataset, file_name)
    return combined_dataset, corruption_sequence


def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    temp_size = os.path.getsize("tmp.pt")
    os.remove('tmp.pt')
    return temp_size # bytes


def calculate_model_full_memory(model, mode, input_size, optimizer, batch_size, precision):
    if precision == "fp32": bits = 32
    else: bits=8
    memory_cost, size_dict = profile_memory_cost(model, input_size=input_size, activation_bits=bits, trainable_param_bits=bits, frozen_param_bits=bits, batch_size=batch_size)
    return memory_cost

