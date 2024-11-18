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
from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance_nd

DEVICE = "cpu"


class QuantizedBalancedBatchNorm(torch.nn.Module):
    def __init__(self, weight_dict, layer_num, scale=1, zero_point=0, tau=0.9, avgr=0.9, do_adapt=True, do_reset=True):
        super(QuantizedBalancedBatchNorm, sel).__init__()
        self.scale = scale
        self.zero_point = zero_point
        
        self.tau, self.avgr = tau, avgr
        self.gamma_weight = torch.mm(weight_dict['gamma'], weight_dict['weight'])
        self.beta = weight_dict['beta']
        self.gamma = weight_dict['gamma']
        self.b = weight_dict['bias']


    def adapt(self, x):
        with torch.no_grad():
            
            # if self.layer_num < self.stop_layer:
            #     return (self.batch_norm.running_mean, self.batch_norm.running_var)

            x = x[0].int_repr().type(torch.float32)
            new_running_mean = self.scale * self.qfunc.add(x.mean([1,2]), -self.zero_point)
            new_running_var = x.var([1,2], unbiased=False) * self.scale**2

            avgr = self.avgr
            tau = self.tau
            new_running_mean = avgr * self.batch_norm.running_mean + (1-avgr) * new_running_mean
            new_running_var = avgr * self.batch_norm.running_var + (1-avgr) * new_running_var
            covar = ((1/(self.batch_norm.running_var+0.000001)))*torch.eye(self.batch_norm.running_var.shape[0])
            mean_diff = new_running_mean - self.batch_norm.running_mean
            temp = torch.matmul(mean_diff, torch.matmul(covar, mean_diff))
            temp = tau*(1-torch.exp(-(temp)))
            running_mean = (temp) * self.batch_norm.running_mean + (1-temp) * new_running_mean
            running_var = (temp) * self.batch_norm.running_var + (1-temp) * new_running_var
            return (running_mean, running_var)

    def forward(self, x):
        mean, var = self.adapt(x)
        eps = 1e-5
        var_denom = torch.rsqrt(eps+var)
        wt = torch.diag(self.gamma_weight * var_denom)
        bias = self.beta + (self.b - mean) * self.gamma * var_denom
        return wt @ x + bias

    @staticmethod
    def find_bns(parent, idx=0, tau=0.9, avgr=0.9, weight_dict=None):
        replace_mods = []
        prev_scale = 1
        prev_zero = 0
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, torch.ao.nn.quantized.Conv2d) or isinstance(child, torch.nn.Conv2d):
                module = QuantizedBalancedBatchNorm(weight_dict, idx, scale=child.scale, zero_point=child.zero_point,tau=0.9,avgr=0.9)
                idx += 1
                replace_mods.append((parent, child, module))
            else:
                replace_mods.extend(BalancedBatchNorm.find_bns(child, idx, tau=tau, avgr=avgr, weight_dict=weight_dict))

        return replace_mods
    
    @staticmethod
    def retrofit_model(model, tau=0.9, avgr=0.1, stop_layer=1000):
        updated_mods = BalancedBatchNorm.find_bns(model, tau=tau, avgr=avgr)
        model.requires_grad_(False)
        idx = 0
        for (parent, name, child) in updated_mods:
            child.layer_num = idx
            child.track_running_stats = False
            # if stop_layer <= 52:
            #     child.set_stop_layer(stop_layer)
            setattr(parent, name, child)
            idx += 1
        return model


class BalancedBatchNorm(torch.nn.Module):
    def __init__(self, layer, layer_num, scale=1, zero_point=0, tau=0.9, do_adapt=True, do_reset=True, avgr=0.9):
        super(BalancedBatchNorm, self).__init__()
        self.scale = scale
        self.zero_point = zero_point
        if not (isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.ao.nn.quantized.BatchNorm2d)):
            raise Exception(f"Layers of type {layer.dtype} are not supported. Must be BatchNorm2d.")
        
        self.batch_norm = layer
        self.qfunc = torch.nn.quantized.FloatFunctional()
        self.tau = tau
        self.original_weights = (deepcopy(layer.running_mean), deepcopy(layer.running_var))
        self.do_adapt = do_adapt
        self.do_reset = do_reset
        self.tau = 0.9
        self.avgr = 0.9
        self.tau_batch = torch.Tensor([0.1,0.3,0.5,0.7,0.9])
        # if DEVICE == "cuda" and torch.cuda.is_available():
        #     print("moving to tau")
        #     self.tau_batch = self.tau_batch.cuda()
        self.layer_num = layer_num
        self.stop_layer = -1
    
    @torch.no_grad()
    def reset_weights(self):
        self.batch_norm.running_mean = self.original_weights[0]
        self.batch_norm.running_var = self.original_weights[1]
    

    def set_stop_layer(self, lyr_num):
        print(lyr_num)
        self.stop_layer = lyr_num


    def toggle_adaptation(self, do_adapt):
        self.do_adapt = do_adapt
        self.do_reset = do_adapt

    
    def set_tau(self, tau):
        self.tau_batch = torch.Tensor([tau])


    @torch.no_grad()
    def adapt(self, x):
        with torch.no_grad():
            
            # if self.layer_num < self.stop_layer:
            #     return (self.batch_norm.running_mean, self.batch_norm.running_var)

            if x.dtype != torch.float32:
                x = x[0].int_repr().type(torch.float32)
                new_running_mean = self.scale * self.qfunc.add(x.mean([1,2]), -self.zero_point)
                new_running_var = x.var([1,2], unbiased=False) * self.scale**2
            else:
                x = x[0]
                new_running_mean = x.mean([1,2])
                new_running_var = x.var([1,2], unbiased=False)

            avgr = self.avgr
            tau = self.tau
            new_running_mean = avgr * self.batch_norm.running_mean + (1-avgr) * new_running_mean
            new_running_var = avgr * self.batch_norm.running_var + (1-avgr) * new_running_var
            covar = ((1/(self.batch_norm.running_var+0.000001)))*torch.eye(self.batch_norm.running_var.shape[0])
            mean_diff = new_running_mean - self.batch_norm.running_mean
            temp = torch.matmul(mean_diff, torch.matmul(covar, mean_diff))
            # temp = torch.matmul((new_running_mean - self.batch_norm.running_mean), torch.linalg.inv((0.000001+self.batch_norm.running_var)*torch.eye(self.batch_norm.running_var.shape[0])))
            # temp = torch.matmul(temp, (new_running_mean - self.batch_norm.running_mean))
            temp = tau*(1-torch.exp(-(temp)))
            running_mean = (temp) * self.batch_norm.running_mean + (1-temp) * new_running_mean
            running_var = (temp) * self.batch_norm.running_var + (1-temp) * new_running_var #+ temp*(1-temp)*(new_running_mean-self.batch_norm.running_mean)**2
            # temp_var = 0.8 # torch.nn.functional.cosine_similarity(new_running_var.reshape(1,-1), self.batch_norm.running_var.reshape(1,-1))
            # temp_mean =0.8 #  torch.nn.functional.cosine_similarity(new_running_mean.reshape(1,-1), self.batch_norm.running_mean.reshape(1,-1))
            # print(temp_mean)
            # running_mean = (temp_mean) * self.batch_norm.running_mean + (1-temp_mean) * new_running_mean
            # running_var = (temp_var) * self.batch_norm.running_var + (1-temp_var) * new_running_var #+ temp*(1-temp)*(new_running_mean-self.batch_norm.running_mean)**2
            return (running_mean, running_var)


    @torch.no_grad()
    def forward(self, x, reset=True, adapt=True):
        with torch.no_grad():
            if adapt: # adaptation pass
                possible_statistics = self.adapt(x)
                outputs = []
                scales = []
                zero_points = []

                self.batch_norm.running_mean = possible_statistics[0]
                self.batch_norm.running_var = possible_statistics[1]
                
            x = self.batch_norm(x)
            self.reset_weights()
            return x
    
    @staticmethod
    def find_bns(parent, idx=0, tau=0.9, avgr=0.9):
        replace_mods = []
        prev_scale = 1
        prev_zero = 0
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, torch.nn.BatchNorm2d) or isinstance(child, torch.ao.nn.quantized.BatchNorm2d):
                module = BalancedBatchNorm(layer=child, layer_num=idx, scale=prev_scale, zero_point=prev_zero, tau=0.1, avgr=0.9)
                replace_mods.append((parent, name, module))
                idx += 1
            elif isinstance(child, torch.ao.nn.quantized.Conv2d):
                prev_scale = child.scale
                prev_zero = child.zero_point
            else:
                replace_mods.extend(BalancedBatchNorm.find_bns(child, idx, tau=tau, avgr=avgr))

        return replace_mods
    
    @staticmethod
    def retrofit_model(model, tau=0.9, avgr=0.1, stop_layer=1000):
        updated_mods = BalancedBatchNorm.find_bns(model, tau=tau, avgr=avgr)
        model.requires_grad_(False)
        idx = 0
        for (parent, name, child) in updated_mods:
            child.layer_num = idx
            child.track_running_stats = False
            if stop_layer <= 52:
                child.set_stop_layer(stop_layer)
            setattr(parent, name, child)
            idx += 1
        return model



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

class CIFAR10CorruptDataset(Dataset):
    """Corrupted CIFAR10C dataset."""

    def __init__(self, corruption_category_name, split=['test'], transform=None):
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
            real_split = split
        
        self.ds_name = corruption_category_name
        self.ds = list(tfds.load(f"cifar10_corrupted/{corruption_category_name}", split=real_split, shuffle_files=True)[0])[:1000]
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

    def get_class_label_indices(self, class_idx):
        label_indices = []
        for idx, item in enumerate(self.ds):
            if item["label"].numpy() == class_idx:
                label_indices.append(idx)
        
        return label_indices



def check_accuracy(MODEL_TO_TEST, dataset, corruption_name, batch_size, scale, trial_num, with_adaptation=False, adapt_idx=0):
    # CHECK ACCURACY WITHOUT ADAPTATION
    count_right, count_seen = 0, 0

    data_loader = torch.utils.data.DataLoader(
        dataset["val"],
        shuffle=True,
        batch_size=batch_size,
        sampler=None,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    if with_adaptation:
        MODEL_TO_TEST = BalancedBatchNorm.retrofit_model(deepcopy(MODEL_TO_TEST))

    outputs_to_save = []
    labels_to_save = []
    confidences = []
    MODEL_TO_TEST.eval()
    num_batches=len(data_loader)
    double_count = 0
    with tqdm(total=num_batches, desc="Validate") as t:
        with torch.no_grad():
            for idx, (images, labels) in enumerate(data_loader):
                if idx == 5:
                    break
                if torch.cuda.is_available and DEVICE == "cuda":
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = MODEL_TO_TEST(images)
                confidences.append(torch.softmax(outputs, dim=1).max(1)[0].mean())
                
                if confidences[-1] < 0:
                    double_count += 1
                    for mod in MODEL_TO_TEST.modules():
                        if isinstance(mod, BalancedBatchNorm):
                            mod.set_tau(max(0.3, confidences[-1]))
                    new_outputs = MODEL_TO_TEST(images)
                    new_conf = torch.softmax(new_outputs, dim=1).max(1)[0].mean() 
                    confidences[-1] = new_conf
                    outputs = new_outputs

                count_seen += labels.shape[0]
                for i, o in enumerate(outputs):
                    if labels[i] == o.argmax():
                        count_right += 1
                t.set_postfix({'accuracy': 100 * count_right / count_seen, 'count_seen': count_seen})
                t.update()
    
    return count_right / count_seen

def main(corruption_name, batch_size, scale, trial_num=0):
    resnet18_pt = resnet18(pretrained=True)
    # resnet18_pt = mobilenet_v2(pretrained=True)

    # prepare model
    resnet18_pt.eval()
    if scale == "int8":
        backend = "fbgemm"
        resnet18_pt.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        resnet18_quant = torch.quantization.prepare(resnet18_pt, inplace=False)
        torch.quantization.prepare(resnet18_quant, inplace=True)
        torch.quantization.convert(resnet18_quant, inplace=True)
        resnet18_quant.load_state_dict(torch.load("resnet18_quant_no_fuse.pth"))
        # resnet18_quant.load_state_dict(torch.load("mobinetv2_quant_no_fuse.pth"))
    else:
        resnet18_quant = resnet18_pt
    if DEVICE == "cuda" and torch.cuda.is_available():
        resnet18_quant.to(DEVICE)

    # prepare data
    print(corruption_name)
    print()
    if "none" in corruption_name:
        dataset = {
            'val': torchvision.datasets.CIFAR10("data/cifar10", train=False,
                                          transform=ImageTransform()['val'], download=True),
        }
    else:
        dataset = {
            "val": CIFAR10CorruptDataset(corruption_name, split="val",
                                                transform=ImageTransform()["val"]),
        }

    np.random.seed(trial_num)
    torch.manual_seed(trial_num)

    data_loader = torch.utils.data.DataLoader(
        dataset["val"],
        batch_size=batch_size,
        sampler=None,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # initial baseline
    check_accuracy(resnet18_quant, dataset, corruption_name, batch_size, scale, trial_num, with_adaptation=False)

    # with running stats update
    for ii in range(1):
        acc = check_accuracy(resnet18_quant, dataset, corruption_name, batch_size, scale, trial_num, with_adaptation=True, adapt_idx=ii)
    return acc

if __name__ == "__main__":
    BENCHMARK_CORRUPTIONS = [
        'none',
        # 'gaussian_noise',
        # 'shot_noise',
        # 'impulse_noise',
        # 'defocus_blur',
        # 'frosted_glass_blur',
        # 'motion_blur',
        # 'zoom_blur',
        # 'snow',
        # 'frost',
        # 'fog',
        # 'brightness',
        # 'contrast',
        # 'elastic',
        # 'pixelate',
        # 'jpeg_compression',
        # 'gaussian_blur',
        # 'saturate',
        # 'spatter',
        # 'speckle_noise',
    ]

    accs = []
    for corruption_name in BENCHMARK_CORRUPTIONS: #["gaussian_noise", "snow", "zoom_blur", "contrast"]:#  BENCHMARK_CORRUPTIONS:
        if corruption_name == 'none':
            main("none_1", 1, "int8", 1)
        else:
            for i in [1,3,5]:
                print(f"{corruption_name}_{i} started...")
                for j in range(0, 1):
                    accs.append(main(f"{corruption_name}_{i}", 1, "int8", j))
                print(f"{corruption_name}_{i} done.")
    print(accs)
