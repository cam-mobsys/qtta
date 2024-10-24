import os, sys, time

import torch
import torchvision

from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import *

from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch._export import capture_pre_autograd_graph

class BatchNormRecorder(torch.nn.Module):
    def __init__(self, layer):
        super(BatchNormRecorder, self).__init__()
        self.layer = layer
        self.mean = []
        self.var = []

    def forward(self, x):
        x = self.layer(x)
        self.mean.append(x.mean())
        self.var.append(x.var())
        return x

    def get_avg(self):
        return (np.mean(self.mean), np.std(self.mean), np.mean(self.var), np.std(self.var))
    

    @staticmethod
    def find_bns(parent):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, torch.nn.BatchNorm2d) or isinstance(child, torch.ao.nn.quantized.BatchNorm2d):
                module = BatchNormRecorder(layer=child)
                replace_mods.append((parent, name, module))
                # layer_count += 1
            else:
                replace_mods.extend(BatchNormRecorder.find_bns(child))

        return replace_mods

    @staticmethod
    def retrofit_model(model):
        mods_to_replace = BatchNormRecorder.find_bns(model)
        for (parent, name, child) in mods_to_replace:
            setattr(parent, name, child)
        return model


def check_quantours(model_name):
    model = resnet18(pretrained=True, num_classes=num_classes)
    model = prep_model(model, "quantours")
    return model


def do_pt2e(model_name):
    num_classes = 10
    if 'resnet' in model_name:
        model = resnet18(pretrained=True, num_classes=num_classes)
    elif 'mobi' in model_name:
        model = mobilenet_v2(pretrained=True, num_classes=num_classes)
    model.eval()
    
    dataset = torchvision.datasets.CIFAR10("data/cifar10", train=False,
                                          transform=ImageTransform(num_classes=num_classes)['val'], download=True)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=64, sampler=None, num_workers=0, pin_memory=True, drop_last=False)

    img, lbl = next(iter(data_loader))
    exported_model = capture_pre_autograd_graph(model, img)
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)

    # calibrate
    with torch.no_grad():
        for idx, (img, lbl) in enumerate(data_loader):
            if idx == 5:
                break
            model(img)

    model = convert_pt2e(prepared_model)
    loop(prepared_model, img)


def loop(model, input_):
    # model = torch.jit.load("resnet18_cifar10_quant.pt")
    with torch.no_grad():
        for i in range(10):
            start = time.time()
            output = model(input_)
            print(f'{time.time() - start:.3f}s')
    return output



def do_quantize(model_name):
    # set up fp32 model
    num_classes = 10
    batch_size = 1
    if 'resnet' in model_name:
        model = resnet18(pretrained=True, num_classes=num_classes)
    elif 'mobi' in model_name:
        model = mobilenet_v2(pretrained=True, num_classes=num_classes)

    # generate data
    dataset = torchvision.datasets.CIFAR10("data/cifar10", train=False,
                                          transform=ImageTransform(num_classes=num_classes)['val'], download=True)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, sampler=None, num_workers=0, pin_memory=True, drop_last=False)

    # calibrate
    with torch.no_grad():
        for idx, (img, lbl) in enumerate(data_loader):
            break
    # model = prep_model(model, "quantours")
    model.cpu().eval()
    loop(model, img)
    backend = "qnnpack"
    modules_to_fuse = []
    for n, m in model.named_modules():
        if 'conv' in n:
            modules_to_fuse.append([n, n.replace("conv", "bn")])
    torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    torch.quantization.prepare(model, inplace=True)
    # model = BatchNormRecorder.retrofit_model(model) 
    
    # calibrate
    with torch.no_grad():
        for idx, (img, lbl) in enumerate(data_loader):
            if idx == 100:
                break
            model(img)
    print(f"Processed {idx} samples")

    torch.quantization.convert(model, inplace=True)
    print(model)

    loop(model, img)

    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if "scale" in k or "zero_point" in k:
            if len(state_dict[k].shape) < 1:
                state_dict[k] = state_dict[k].expand(1)
        new_key = k.split(".")
        if new_key[-2] == "batch_norm" or new_key[-2] == "qfunc":
            del new_key[-2]
        state_dict[".".join(new_key)] = state_dict.pop(k)
        
    torch.save(state_dict, "resnet18_cifar10_qnn_fuse.pt")
    # torch.jit.save(torch.jit.script(model), "resnet18_cifar10_quant.pt")
    # mus, sigmas = [], []
    # for mod in model.modules():
    #     if isinstance(mod, BatchNormRecorder):
    #         mus.append(mod.get_avg()[0])
    #         sigmas.append(mod.get_avg()[2])
    # print(mus, sigmas)

if __name__ == '__main__':
    # do_pt2e("resnet18")
    do_quantize("resnet18")
