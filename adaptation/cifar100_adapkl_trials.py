"""
For purposes of conducting quantized tta on CIFAR100-C
"""
import os
import torchvision
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import itertools
import json
import math
from torchvision import transforms
import random
import torch
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from tqdm import tqdm

DEVICE = "cpu"

import sys
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional



cifar100_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
}

# https://github.com/chenyaofo/pytorch-cifar-models courtesy of
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)

        return x


def _resnet(
    arch: str,
    layers: List[int],
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(BasicBlock, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

# courtesy of ends https://github.com/chenyaofo/pytorch-cifar-models

def cifar100_resnet20(*args, **kwargs) -> CifarResNet: pass

class ImageTransform(dict):
    def __init__(self):
        super().__init__({
            # 'train': self.build_train_transform(),
            'val': self.build_val_transform()
        })

    def build_val_transform(self):
        if True:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
            ])

from typing import Tuple
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
        self.data.append(np.load(file_path)[(self.corr_num-1)*10000:self.corr_num*10000])
        self.targets.extend(labels)

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


def update_statistics_hook(model_quantized, do_update=False, kl_divs=[]):
    # add hook to force update running stats for q_batchnorm
    import functools
    def _calculate_stats(self, x, y, i, module, scale=1, zero=0, do_update=False):
      x = x[0].int_repr().type(torch.float32) # x is the activation
      # x=x[0]
      # reconstitute mean, variance from existing weights
      # need to make sure it is on the same scale as the existing parameters
      # new_running_mean = x.mean([0,2,3])
      # new_running_var = x.var([0,2,3], unbiased=False)
      new_running_mean = scale * quant_func.add(x.mean([0,2,3]), -zero)
      new_running_var = x.var([0,2,3], unbiased=False) * scale**2
      if not do_update:
        # scale momentum from 0.75 to 0.95
        source_dist = torch.distributions.MultivariateNormal(module.running_mean, (module.running_var+0.00001)*torch.eye(module.running_var.shape[0]))
        target_dist = torch.distributions.MultivariateNormal(new_running_mean, (new_running_var+0.00001)*torch.eye(new_running_var.shape[0]))
        sim = (0.5*torch.distributions.kl_divergence(source_dist, target_dist) + 0.5*torch.distributions.kl_divergence(target_dist, source_dist))
        # sim = torch.distributions.kl_divergence(source_dist, target_dist)
        # sim_mean = torch.nn.functional.cosine_similarity(
                # module.running_mean, new_running_mean, dim=0)
        # sim_var = torch.nn.functional.cosine_similarity(
                # module.running_var, new_running_var, dim=0)
        kl_divs.append(sim)
        #mean_momentum = 0.9 + 0.1*(1-max(sim_mean, 0))
        #var_momentum = 0.9 + 0.1*(1-max(sim_var, 0))
        # mean_momentum = 0.98
        # var_momentum = 0.98
        # module.running_mean = mean_momentum*module.running_mean + (1-mean_momentum)*new_running_mean
        # module.running_var = var_momentum*module.running_var + (1-var_momentum)*new_running_var + var_momentum*(1-var_momentum)*(module.running_mean - new_running_mean)**2
      if do_update and len(kl_divs) > 0:
        var_momentum = 0.9 + 0.1*(kl_divs[i])
        mean_momentum = 0.9 + 0.1*(kl_divs[i])
        module.running_mean = mean_momentum*module.running_mean + (1-mean_momentum)*new_running_mean
        module.running_var = var_momentum*module.running_var + (1-var_momentum)*new_running_var + var_momentum*(1-var_momentum)*(module.running_mean - new_running_mean)**2

    all_hooks = []
    seen_count = 0
    prev_scale, prev_zero = 1, 0
    quant_func = torch.ao.nn.quantized.FloatFunctional()
    print(len(kl_divs), seen_count, do_update)
    if len(kl_divs) > 0 and not do_update:
        kl_divs = []
    for n, mod in model_quantized.named_modules():
      if isinstance(mod, torch.ao.nn.quantized.Conv2d):
        # record scale from convolutional layer
        prev_scale = mod.scale
        prev_zero = mod.zero_point
      if isinstance(mod, torch.nn.BatchNorm2d) or isinstance(mod, torch.ao.nn.quantized.BatchNorm2d):
        # add hooks
        all_hooks.append(mod.register_forward_hook(
            functools.partial(_calculate_stats, i=seen_count, module=mod, scale=prev_scale, zero=prev_zero, do_update=do_update)))
        seen_count += 1
    return all_hooks, kl_divs


def scale_to_mean_std(numbers):
    mean = np.mean(numbers)
    std = math.sqrt(sum((x - mean) ** 2 for x in numbers) / len(numbers))
    scaled_numbers = [(x - mean) / std for x in numbers]
    return scaled_numbers


def fix_model_settings(model):
    for module in model.modules():
        if isinstance(module, torch.ao.nn.quantized.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = None
            module.num_batches_tracked = torch.tensor(0)

    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def check_accuracy(MODEL_TO_TEST, dataset, corruption_name, batch_size, scale, trial_num, with_adaptation=False):
    # CHECK ACCURACY WITHOUT ADAPTATION
    count_right, count_seen = 0, 0

    if with_adaptation:
        MODEL_TO_TEST = fix_model_settings(MODEL_TO_TEST)
        hooks, kl_divs = update_statistics_hook(MODEL_TO_TEST)
        print(kl_divs)
        data_loader = torch.utils.data.DataLoader(
            dataset["val"],
            shuffle=True,
            batch_size=batch_size,
            sampler=None,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        images, labels = next(iter(data_loader))
        # MODEL_TO_TEST.train()
        try:
            with torch.no_grad():
                outputs = MODEL_TO_TEST(images)
                MODEL_TO_TEST.eval()
        finally:
            for h in hooks:
                h.remove()
            del hooks
        
        # MODEL_TO_TEST.train()
        kl_divs_normed = [(max(min(div, 1), -1) +1)/2 for div in scale_to_mean_std(kl_divs)]
        hooks, _ = update_statistics_hook(MODEL_TO_TEST, do_update=True, kl_divs=kl_divs_normed)
        MODEL_TO_TEST = fix_model_settings(MODEL_TO_TEST)
        try:
            with torch.no_grad():
                outputs = MODEL_TO_TEST(images)
                MODEL_TO_TEST.eval()
        finally:
            for h in hooks:
                h.remove()
            del hooks
    
    data_loader = torch.utils.data.DataLoader(
        dataset["val"],
        shuffle=True,
        batch_size=32,
        sampler=None,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    MODEL_TO_TEST = fix_model_settings(MODEL_TO_TEST)
    
    outputs_to_save = []
    labels_to_save = []
    MODEL_TO_TEST.eval()
    num_batches=125
    with tqdm(total=num_batches, desc="Validate") as t:
        with torch.no_grad():
            for idx, (images, labels) in enumerate(data_loader):
                if idx > num_batches:
                    break
                if torch.cuda.is_available and DEVICE == "cuda":
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = MODEL_TO_TEST(images)
                count_seen += labels.shape[0]
                for i, o in enumerate(outputs):
                    if labels[i] == o.argmax():
                        count_right += 1
                t.set_postfix({'accuracy': 100 * count_right / count_seen, 'count_seen': count_seen})
                t.update()

                outputs_to_save.append(outputs.detach().cpu())
                labels_to_save.append(labels.cpu())
    
    loss_name = "c100kl90" if with_adaptation else "baseline"
    np.save(f"cifar100/{corruption_name}_{'resnet18'}_{scale}_{loss_name}_b{batch_size}_{trial_num}.npy", \
            {"out": outputs_to_save, "labels": labels_to_save, "acc": count_right/count_seen,}# "kl": kl_divs}
    )

def main(corruption_name, batch_size, scale, trial_num):
    resnet18_pt = _resnet("resnet20", layers=[3]*3, model_urls=cifar100_pretrained_weight_urls, pretrained=True, num_classes=100)

    # prepare model
    resnet18_pt.eval()
    if scale == "int8":
        backend = "fbgemm"
        resnet18_pt.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        resnet18_quant = torch.quantization.prepare(resnet18_pt, inplace=False)
        torch.quantization.prepare(resnet18_quant, inplace=True)
        torch.quantization.convert(resnet18_quant, inplace=True)
        resnet18_quant.load_state_dict(torch.load("rn20cifar100_quant_no_fuse.pth"))
    else:
        resnet18_quant = resnet18_pt
    if DEVICE == "cuda" and torch.cuda.is_available():
        resnet18_quant.to(DEVICE)

    # prepare data
    print(corruption_name)
    print()
    if "none" in corruption_name:
        dataset = {
            'val': torchvision.datasets.CIFAR100("data/cifar100", train=False,
                                          transform=ImageTransform()['val'], download=True),
        }
    else:
        c100_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            ])
        dataset = {
            "val": CorruptCIFAR("", corruption_name, train=False, download=False, transform=c100_transform)
        }

    data_loader = torch.utils.data.DataLoader(
        dataset["val"],
        shuffle=True,
        batch_size=batch_size,
        sampler=None,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # initial baseline
    check_accuracy(resnet18_quant, dataset, corruption_name, batch_size, scale, trial_num, with_adaptation=False)

    # with running stats update
    check_accuracy(resnet18_quant, dataset, corruption_name, batch_size, scale, trial_num, with_adaptation=True)

if __name__ == "__main__":
    BENCHMARK_CORRUPTIONS = [
        'gaussian_noise',
        'shot_noise',
        'impulse_noise',
        'defocus_blur',
        'motion_blur',
        'zoom_blur',
        'glass_blur',
        'snow',
        'frost',
        'fog',
        'brightness',
        'contrast',
        'elastic_transform',
        'pixelate',
        'jpeg_compression',
        'gaussian_blur',
        'saturate',
        'spatter',
        'speckle_noise',
        'none'
    ]

    for corruption_name in ["none"]:# BENCHMARK_CORRUPTIONS: #["gaussian_noise", "snow", "zoom_blur", "contrast"]:#  BENCHMARK_CORRUPTIONS:
        for i in range(1, 6):
            print(f"{corruption_name}_{i} started...")
            for j in range(0, 5):
                main(f"{corruption_name}_{i}", 1, "int8", j)
            print(f"{corruption_name}_{i} done.")

