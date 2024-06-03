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
from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2
from tqdm import tqdm

DEVICE = "cpu"

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
        self.ds = list(tfds.load(f"cifar10_corrupted/{corruption_category_name}", split=real_split, shuffle_files=True)[0])
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


def update_statistics_hook(model_quantized, do_update=False, kl_divs=[], tau=0.9):
    # add hook to force update running stats for q_batchnorm
    import functools
    def _calculate_stats(self, x, y, i, module, scale=1, zero=0, do_update=False):
      # x = x[0].int_repr().type(torch.float32) # x is the activation
      x=x[0]
      # reconstitute mean, variance from existing weights
      # need to make sure it is on the same scale as the existing parameters
      new_running_mean = x.mean([0,2,3])
      new_running_var = x.var([0,2,3], unbiased=False)
      # new_running_mean = scale * quant_func.add(x.mean([0,2,3]), -zero)
      # new_running_var = x.var([0,2,3], unbiased=False) * scale**2
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
        var_momentum = tau + (1-tau)*(kl_divs[i])
        mean_momentum = tau + (1-tau)*(kl_divs[i])
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
        #conf = torch.softmax(outputs, dim=1).max(1)[0].mean()
        #if conf > 0.85:
        #    tau=0.9
        #else:
        #    tau=0.7
        tau = 0.9
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
    
    loss_name = "c10kl90" if with_adaptation else "baseline"
    try:
        len(kl_divs)
    except:
        kl_divs = []
    np.save(f"baseline_pt/{corruption_name}_{'mbnet'}_{scale}_{loss_name}_b{batch_size}_{trial_num}.npy", \
            {"out": outputs_to_save, "labels": labels_to_save, "acc": count_right/count_seen, "kl_divs": kl_divs}
    )

def main(corruption_name, batch_size, scale, trial_num=0):
    mbn_pt = mobilenet_v2(pretrained=True)

    # prepare model
    mbn_pt.eval()
    if scale == "int8":
        backend = "fbgemm"
        mbn_pt.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        mbn_quant = torch.quantization.prepare(mbn_pt, inplace=False)
        torch.quantization.prepare(mbn_quant, inplace=True)
        torch.quantization.convert(mbn_quant, inplace=True)
        mbn_quant.load_state_dict(torch.load("mobinetv2_quant_no_fuse.pth"))
    else:
        mbn_quant = mbn_pt
    if DEVICE == "cuda" and torch.cuda.is_available():
        mbn_quant.to(DEVICE)

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
        shuffle=True,
        batch_size=batch_size,
        sampler=None,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # initial baseline
    check_accuracy(mbn_quant, dataset, corruption_name, batch_size, scale, trial_num, with_adaptation=False)

    # with running stats update
    check_accuracy(mbn_quant, dataset, corruption_name, batch_size, scale, trial_num, with_adaptation=True)

if __name__ == "__main__":
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
        'none'
    ]

    for corruption_name in BENCHMARK_CORRUPTIONS: #["gaussian_noise", "snow", "zoom_blur", "contrast"]:#  BENCHMARK_CORRUPTIONS:
        for i in range(1, 6):
            print(f"{corruption_name}_{i} started...")
            for j in range(0, 5):
                main(f"{corruption_name}_{i}", 1, "fp32", j)
            print(f"{corruption_name}_{i} done.")
