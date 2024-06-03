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
from tinyml.tinytl.tinytl.utils import profile_memory_cost
from tqdm import tqdm
import sys
import time
import tracemalloc
from memory_profiler import profile
import psutil
import multiprocessing as mp
import requests

DEVICE = "cpu"

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
]

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
        self.ds = list(tfds.load(f"cifar10_corrupted/{corruption_category_name}", split=real_split, shuffle_files=True)[0])[:100]
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


all_dsets = {}
for BC in BENCHMARK_CORRUPTIONS:
    for i in range(1, 6):
        all_dsets[f"{BC}_{i}"] = CIFAR10CorruptDataset(f"{BC}_{i}", split="val",
                                            transform=ImageTransform()["val"])


def update_statistics_hook(model_quantized, do_update=False, kl_divs=[], tau=0.9):
    # add hook to force update running stats for q_batchnorm
    import functools
    def _calculate_stats(self, x, y, i, module, scale=1, zero=0, do_update=False):
      if sys.argv[3] == "int8":
        x = x[0].int_repr().type(torch.float32) # x is the activation
        new_running_mean = scale * quant_func.add(x.mean([0,2,3]), -zero)
        new_running_var = x.var([0,2,3], unbiased=False) * scale**2
      else:
        x = x[0]
        new_running_mean = x.mean([0,2,3])
        new_running_var = x.var([0,2,3], unbiased=False)

      if not do_update:
        source_dist = torch.distributions.MultivariateNormal(module.running_mean, (module.running_var+0.00001)*torch.eye(module.running_var.shape[0]))
        target_dist = torch.distributions.MultivariateNormal(new_running_mean, (new_running_var+0.00001)*torch.eye(new_running_var.shape[0]))
        sim = (0.5*torch.distributions.kl_divergence(source_dist, target_dist) + 0.5*torch.distributions.kl_divergence(target_dist, source_dist))
        kl_divs.append(sim)
      
      if do_update and len(kl_divs) > 0:
        var_momentum = tau + (1-tau)*(kl_divs[i])
        mean_momentum = tau + (1-tau)*(kl_divs[i])
        module.running_mean = mean_momentum*module.running_mean + (1-mean_momentum)*new_running_mean
        module.running_var = var_momentum*module.running_var + (1-var_momentum)*new_running_var + var_momentum*(1-var_momentum)*(module.running_mean - new_running_mean)**2

    all_hooks = []
    seen_count = 0
    prev_scale, prev_zero = 1, 0
    quant_func = torch.ao.nn.quantized.FloatFunctional()

    if len(kl_divs) > 0 and not do_update:
        kl_divs = []
    for n, mod in model_quantized.named_modules():
      if isinstance(mod, torch.ao.nn.quantized.Conv2d):
        prev_scale = mod.scale
        prev_zero = mod.zero_point
      if isinstance(mod, torch.nn.BatchNorm2d) or isinstance(mod, torch.ao.nn.quantized.BatchNorm2d):
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

import pickle
from copy import deepcopy
def check_accuracy(MODEL_TO_TEST, dataset, batch_size, scale, trial_num, with_adaptation=False, update_freq=10):
    count_right, count_seen = 0, 0
    # update_freq = int(sys.argv[2])
    dataset, dset_names = dataset
     
    original_model = torch.load("resnet18_quant_no_fuse.pth")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        sampler=None,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    MODEL_TO_TEST = fix_model_settings(MODEL_TO_TEST)
    
    outputs_to_save = []
    labels_to_save = []
    MODEL_TO_TEST.eval()
    num_batches=len(data_loader)

    times = []
    print(time.time())
    q = mp.Queue()
    monitor_proc = mp.Process(target=monitor_memory, args=(q,))
    with tqdm(total=num_batches, desc="Validate") as t:
        with torch.no_grad():
            monitor_proc.start()
            for idx, (images, labels) in enumerate(data_loader):

                if idx % update_freq == 0 and with_adaptation:
                    # completely reset model
                    #if sys.argv[3] == "int8":
                        # MODEL_TO_TEST=deepcopy(original_model)
                    #    MODEL_TO_TEST.load_state_dict(original_model)
                        
                    #else:
                    #    MODEL_TO_TEST = resnet18(pretrained=True) 
                    # MODEL_TO_TEST = fix_model_settings(MODEL_TO_TEST)
                    hooks, kl_divs = update_statistics_hook(MODEL_TO_TEST)
                    try:
                        with torch.no_grad():
                            outputs = MODEL_TO_TEST(images)
                            MODEL_TO_TEST.eval()
                    finally:
                        for h in hooks:
                            h.remove()
                        del hooks
                    
                    # MODEL_TO_TEST.train()
#                    conf = torch.softmax(outputs, dim=1).max(1)[0].mean()
#                    if conf > 0.825:
#                        tau=1
#                    else:
#                        tau=0.9
                    tau=0.9
                    kl_divs_normed = [(max(min(div, 1), -1) +1)/2 for div in scale_to_mean_std(kl_divs)]
                    
                    hooks, _ = update_statistics_hook(MODEL_TO_TEST, do_update=True, kl_divs=kl_divs_normed, tau=tau)
                    MODEL_TO_TEST = fix_model_settings(MODEL_TO_TEST)
                    try:
                        with torch.no_grad():
                            outputs = MODEL_TO_TEST(images)
                            MODEL_TO_TEST.eval()
                    finally:
                        for h in hooks:
                            h.remove()
                        del hooks
                
                # if torch.cuda.is_available and DEVICE == "cuda":
                #     images = images.cuda()
                #     labels = labels.cuda()
                
                outputs = MODEL_TO_TEST(images)
                times.append(time.time())
                count_seen += labels.shape[0]
                for i, o in enumerate(outputs):
                    if labels[i] == o.argmax():
                        count_right += 1
                t.set_postfix({'accuracy': 100 * count_right / count_seen, 'count_seen': count_seen})
                t.update()

                # outputs_to_save.append(outputs.detach().cpu())
                # labels_to_save.append(labels.cpu())
    if with_adaptation:
        q.put(update_freq)
    else:
        q.put("na")
    # np.save(f"22may/sharp_{sys.argv[2]}_{trial_num}_times.npy", times)
    print(np.mean(np.diff(times)), np.std(np.diff(times)))
    # print(f"22may/{'resnet18'}_{scale}_{update_freq}_b{batch_size}_u{sys.argv[2]}_n{100}_{trial_num}.npy")
    # np.save(f"22may/{'resnet18'}_{scale}_{sys.argv[1]}_b{batch_size}_u{sys.argv[2]}_n{10}_{trial_num}.npy", \
    #    {"out": outputs_to_save, "labels": labels_to_save, "acc": count_right/count_seen, "dset_names": dset_names}
    #)

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

file_path = "mobinetv2_quant_no_fuse.pth"
# file_path = "resnet18_quant_no_fuse.pth" 
def main(dataset, batch_size, scale, dset_names):
    # dataset, dset_names = generate_dataset()
    resnet18_pt = mobilenet_v2(pretrained=True) 
    # resnet18_pt = resnet18(pretrained=True)
    print_model_size(resnet18_pt)
    # prepare model
    resnet18_pt.eval()
    memory_cost, detailed_info = profile_memory_cost(resnet18_pt, (1,3,32,32), False, activation_bits=32, trainable_param_bits=32, frozen_param_bits=32, batch_size=1)
    print(memory_cost)
    if scale == "int8":
        backend = "fbgemm"
        resnet18_pt.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        resnet18_quant = torch.quantization.prepare(resnet18_pt, inplace=False)
        torch.quantization.prepare(resnet18_quant, inplace=True)
        torch.quantization.convert(resnet18_quant, inplace=True)
        resnet18_quant.load_state_dict(torch.load(file_path))
        torch.jit.save(torch.jit.script(resnet18_quant), "resnet18.pth")
    else:
        resnet18_quant = resnet18_pt
    if DEVICE == "cuda" and torch.cuda.is_available():
        resnet18_quant.to(DEVICE)
    print_model_size(resnet18_quant)
    memory_cost, detailed_info = profile_memory_cost(resnet18_quant, (1,3,32,32), False, activation_bits=8, trainable_param_bits=8, frozen_param_bits=8, batch_size=1)
    print(memory_cost)
    # initial baseline
    # check_accuracy(resnet18_quant, (dataset, dset_names), batch_size, scale, 0, with_adaptation=False)
    
    # with running stats update
    # check_accuracy(resnet18_quant, (dataset, dset_names), batch_size, scale, 1, with_adaptation=True, update_freq=1)
    # check_accuracy(resnet18_quant, (dataset, dset_names), batch_size, scale, 1, with_adaptation=True)
    check_accuracy(resnet18_quant, (dataset, dset_names), batch_size, scale, 1, with_adaptation=True, update_freq=100)
    

def generate_dataset():
    dset_type = sys.argv[1]
    dataset = []
    dset_names = []
    if dset_type == "gradual":
        for BC in sorted(BENCHMARK_CORRUPTIONS):
            for i in range(1, 6):
                dataset.append(all_dsets[f"{BC}_{i}"])
                dset_names.append(f"{BC}_{i}")
            for i in reversed(range(1, 5)):
                dataset.append(all_dsets[f"{BC}_{i}"])
                dset_names.append(f"{BC}_{i}")
    elif dset_type == "sharp":
        random.seed(1)
        for BC in random.sample(BENCHMARK_CORRUPTIONS, len(BENCHMARK_CORRUPTIONS)):
            level = random.randint(3, 5)
            dataset.append(all_dsets[f"{BC}_{level}"])
            dset_names.append(f"{BC}_{level}")
    elif dset_type == "detection":
        random.seed(2)
        for BC in ["gaussian_noise"]:
            level = random.randint(3, 5)
            dataset.append(all_dsets[f"{BC}_{level}"])
            dset_names.append(f"{BC}_{level}")
    
    combined_dataset = torch.utils.data.ConcatDataset(dataset)
    return combined_dataset, dset_names

def monitor_memory(q):
    # baseline_memory = psutil.virtual_memory().total - psutil.virtual_memory().available
    # mem_available = 0
    times = []
    power = []
    while q.empty():
    #    mem_available= max(mem_available, psutil.virtual_memory().total - psutil.virtual_memory().available)
        r = requests.get("http://192.168.1.230/cm?cmnd=Status%208")
        r = r.json()
        power.append(r["StatusSNS"]["ENERGY"]["Power"]) 
        times.append(time.time())
        time.sleep(0.005)
    freq = q.get()
    # print(mem_available - baseline_memory)
    np.save(f"baseline_pt/resnet_fp32_power_{freq}.npy", {"t": times, "p": power})
    print("Avg", np.mean(power),"Max", np.max(power))
    return

if __name__ == "__main__":

    combined_dataset, dset_names = generate_dataset()
    print(dset_names)
    main(combined_dataset, 1, sys.argv[3], dset_names)
    print("finished", sys.argv[0])
    


