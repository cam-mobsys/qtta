import os, sys, time, random, math, itertools 
import json, requests, psutil, pickle
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from copy import deepcopy
from memory_profiler import profile
from scipy.stats import entropy, wasserstein_distance, wasserstein_distance_nd
from scipy.spatial.distance import jensenshannon

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.distributions as D
import torch.autograd.profiler as profiler

from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2
from tinyml.tinytl.tinytl.utils import profile_memory_cost
from utils import *

DEVICE = "cpu"
class BalancedBatchNorm(torch.nn.Module):
    def __init__(self, layer, scale=1, zero_point=0, tau=0.9, do_adapt=True, do_reset=True, update_cadence=1):
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
        self.update_cadence = update_cadence
        self.counter = 0
        # self.tau_batch = torch.Tensor([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        self.tau_batch = torch.Tensor([0.80])
        if DEVICE == "cuda" and torch.cuda.is_available():
            self.tau_batch = self.tau_batch.cuda()

    
    @torch.no_grad()
    def reset_weights(self):
        self.batch_norm.running_mean = self.original_weights[0]
        self.batch_norm.running_var = self.original_weights[1]
    

    def toggle_adaptation(self, do_adapt):
        self.do_adapt = do_adapt
        self.do_reset = do_adapt


    @torch.no_grad()
    def adapt(self, x):
        with torch.no_grad():
            if x.dtype != torch.float32:
                x = x[0].int_repr().type(torch.float32)
                new_running_mean = self.scale * self.qfunc.add(x.mean([1,2]), -self.zero_point)
                new_running_var = x.var([1,2], unbiased=False) * self.scale**2
            else:
                x = x[0]
                new_running_mean = x.mean([1,2])
                new_running_var = x.var([1,2], unbiased=False)

            # make batch with different taus
            alpha = 0.9
            # temp_running_mean = new_running_# alpha*self.batch_norm.running_mean + (1-alpha)*new_running_mean
            # temp_running_var = # alpha*self.batch_norm.running_var + (1-alpha)*new_running_var + alpha*(1-alpha)*(new_running_mean-self.batch_norm.running_mean)**2
            mean_sim = (1+torch.nn.functional.cosine_similarity(new_running_mean.reshape(1, -1), self.batch_norm.running_mean.reshape(1, -1)))/2
            var_sim = (1+torch.nn.functional.cosine_similarity(new_running_var.reshape(1, -1), self.batch_norm.running_var.reshape(1, -1)))/2
            # mean_sim =
            # var_sim = 0.5
            mean_balancing_constant = alpha + (1-alpha) * (1-mean_sim)
            var_balancing_constant = alpha + (1-alpha) * (1-var_sim)
            
            # more similar means more dependence on target statistics
            # more different means more dependence on source statistics
            running_mean = torch.matmul(mean_balancing_constant[:, None], self.batch_norm.running_mean[None]) + torch.matmul((1-mean_balancing_constant)[:, None],new_running_mean[None])
            running_var = torch.matmul(var_balancing_constant[:,None], self.batch_norm.running_var[None]) + torch.matmul((1-var_balancing_constant)[:, None], new_running_var[None])
            return (running_mean, running_var)


    @torch.no_grad()
    def forward(self, x):
        with torch.no_grad():
            self.counter += 1
            """
            if self.counter % self.update_cadence == 0:
                self.do_reset = True
                self.do_adapt = True
                self.counter = 0
            if self.do_reset:
                self.reset_weights()
            """
            if self.do_adapt: # adaptation pass
                possible_statistics = self.adapt(x)
                self.batch_norm.running_mean = possible_statistics[0][0]
                self.batch_norm.running_var = possible_statistics[1][0]
            if self.do_reset:
                self.reset_weights()
            x = self.batch_norm(x)

            return x

    """
    @torch.no_grad()
    def adapt(self, x):
        with torch.no_grad():
            if x.dtype != torch.float32:
                x = x[0].int_repr().type(torch.float32)
                new_running_mean = self.scale * self.qfunc.add(x.mean([1,2]), -self.zero_point)
                new_running_var = x.var([1,2], unbiased=False) * self.scale**2
            else:
                x = x[0]
                new_running_mean = x.mean([1,2])
                new_running_var = x.var([1,2], unbiased=False)

            # make batch with different taus
            mean_sim = torch.nn.functional.cosine_similarity(new_running_mean.reshape(1, -1), self.batch_norm.running_mean.reshape(1, -1))
            var_sim = torch.nn.functional.cosine_similarity(new_running_var.reshape(1, -1), self.batch_norm.running_var.reshape(1, -1))

            mean_balancing_constant = self.tau + (1-self.tau) * (1-mean_sim)
            var_balancing_constant = self.tau + (1-self.tau) * (1-var_sim)
            
            # more similar means more dependence on target statistics
            # more different means more dependence on source statistics
            running_mean = mean_balancing_constant*self.batch_norm.running_mean + (1-mean_balancing_constant)*new_running_mean
            running_var = var_balancing_constant*self.batch_norm.running_var + (1-var_balancing_constant)*new_running_var + (1-var_balancing_constant)*var_balancing_constant*(self.batch_norm.running_mean - new_running_mean)
            # running_mean = torch.matmul(mean_balancing_constant[:, None], self.batch_norm.running_mean[None]) + torch.matmul((1-mean_balancing_constant)[:, None],new_running_mean[None])
            # running_var = torch.matmul(var_balancing_constant[:,None], self.batch_norm.running_var[None]) + torch.matmul((1-var_balancing_constant)[:, None], new_running_var[None]) + (1-var_balancing_constant)*var_balancing_constant*(self.batch_norm.running_mean[None]-new_running_mean[None])
            return (running_mean, running_var)


   
    @torch.no_grad()
    def forward(self, x, reset=False, adapt=True):
        with torch.no_grad():
            self.counter += 1
            if self.counter % self.update_cadence == 0:
                self.do_reset = True
                self.do_adapt = True
                self.counter = 0
            if self.do_reset:
                self.reset_weights()
            if self.do_adapt: # adaptation pass
                possible_statistics = self.adapt(x)
                self.batch_norm.running_mean = possible_statistics[0]
                self.batch_norm.running_var = possible_statistics[1]
                x = self.batch_norm(x)
                outputs = []
                for mean, var in zip(possible_statistics[0], possible_statistics[1]):
                    self.batch_norm.running_mean = mean
                    self.batch_norm.running_var = var
                    o = self.batch_norm(x)
                    outputs.append(o)
                x = outputs[-1]
                # diff = 0.5*(outputs[1].dequantize() - outputs[0].dequantize())/outputs[1].dequantize() + 0.5*(outputs[0].dequantize() - outputs[1].dequantize())/outputs[0].dequantize()
                # print(np.linalg.norm(diff))
                # print(self.batch_norm.bias / self.batch_norm.running_var)
            else:
                self.batch_norm(x)
 
            return x
     
    """

    @staticmethod
    def find_bns(parent, tau, avgr):
        replace_mods = []
        prev_scale = 1
        prev_zero = 0
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, torch.nn.BatchNorm2d) or isinstance(child, torch.ao.nn.quantized.BatchNorm2d):
                module = BalancedBatchNorm(layer=child, scale=prev_scale, zero_point=prev_zero, tau=tau, avgr=avgr)
                replace_mods.append((parent, name, module))
                # layer_count += 1
            elif isinstance(child, torch.ao.nn.quantized.Conv2d):
                prev_scale = child.scale
                prev_zero = child.zero_point
            else:
                replace_mods.extend(BalancedBatchNorm.find_bns(child, tau, avgr))

        return replace_mods
    
    @staticmethod
    def retrofit_model(model, tau, avgr):
        mods_to_replace = BalancedBatchNorm.find_bns(model)
        print("replacing", len(mods_to_replace), "layers")
        for (parent, name, child) in mods_to_replace:
            setattr(parent, name, child)
        return model

def check_accuracy(MODEL_TO_TEST, dataset, args, trial_num=0, with_adaptation=False, tau=0.9, averager=0.9):
    count_right, count_seen = 0, 0
    if args["exp_type"] == "mix_shifts":
        do_shuffle = True
    elif args["exp_type"] == "gradual":
        do_shuffle = False
    else:
        do_shuffle = True #args["do_shuffle"]

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=do_shuffle,
        batch_size=args["batch_size"],
        sampler=None,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    if with_adaptation and args["tta"]:
        MODEL_TO_TEST = prep_model(MODEL_TO_TEST, args["tta"], args, 0.9, 0.9)
        if torch.cuda.is_available and DEVICE == 'cuda':
            MODEL_TO_TEST = MODEL_TO_TEST.cuda()
        
    outputs_to_save = []
    labels_to_save = []
    time_steps = []
    # MODEL_TO_TEST.eval()
    num_batches = len(data_loader)
    count_right, count_seen = 0, 0
    classes = {}
    for i in range(int(args['class_num'])):
        classes[i] = 0
    
    file_name = f"{args['model']}_{args['class_num']}_{args['exp_type']}.txt"
    with tqdm(total=num_batches, desc="Validate") as t:
        with torch.no_grad():
            for idx, (images, labels) in enumerate(data_loader):
                if args['exp_type'] != 'episodic' and idx > len(data_loader):
                    break
                elif args['exp_type'] == 'episodic' and idx > args['data_size']//args["batch_size"]:
                    break
                if torch.cuda.is_available and DEVICE == "cuda":
                    images = images.cuda()
                    labels = labels.cuda()
                    # count_right, count_seen = 0, 0

                start_time = time.time()
                outputs = MODEL_TO_TEST(images)
                time_steps.append(time.time() - start_time)

                if idx*args["batch_size"] >= args["warmup"]:
                    count_seen += labels.shape[0]
                    for i, o in enumerate(outputs):
                        # print(o.argmax(), labels[i])
                        if labels[i] == o.argmax():
                            count_right += 1
                        classes[int(labels[i].detach())] += 1
                    t.set_postfix({'accuracy': 100. * count_right / count_seen, 'count_seen': count_seen})
                t.update()
                
                if idx == 999 and args['tta']=='tema' and args['save']:
                    with open(file_name, "a") as f:
                        if args['exp_type'] == 'episodic':
                            f.write(f"{args['tta']}1k,{args['batch_size']},{corruption_name},{i},{count_right/count_seen}\n")
                        else:
                            f.write(f"{args['tta']}1k,{args['batch_size']},{count_right/count_seen}\n")
                    count_right, count_seen = 0, 0
                # outputs_to_save.append(outputs.detach().cpu())
                # labels_to_save.append(labels.cpu())
            with profiler.profile(with_stack=True, profile_memory=True) as prof:
                MODEL_TO_TEST(images)
            print(prof.key_averages(group_by_stack_n=10).table(sort_by='self_cpu_time_total', row_limit=10))

    print(classes)
    return count_right / count_seen, time_steps


if __name__ == "__main__":
    args = get_args()
    print(args)
    args = vars(args)

    if args["model"] == "resnet18":
        if args["class_num"] == "10":
            file_path = "resnet18_quant_no_fuse.pth"
            # file_path = "resnet18_test_quant.pth"
            resnet18_pt = resnet18(pretrained=True, num_classes=10)
        elif args["class_num"] == "100":
            file_path = "resnet18_cifar100_quant.pth"
            resnet18_pt = resnet18(pretrained=True, num_classes=100)
        else:
            raise Exception("Model/class mismatch")
    elif args["model"] == "mobilenetv2":
        if args["class_num"] == "10":
            file_path = "mobinetv2_quant_no_fuse.pth"
            resnet18_pt = mobilenet_v2(pretrained=True, num_classes=10)
        elif args["class_num"] == "100":
            file_path = "mobinetv2_cifar100_quant.pth"
            resnet18_pt = mobilenet_v2(pretrained=True, num_classes=100)
    else:
        raise Exception(f"Model {args['model']} not supported.")
    
    # dataset, dset_names = generate_dataset()
    print_model_size(resnet18_pt)
    # prepare model
    resnet18_pt.eval()
    # memory_cost, _ = profile_memory_cost(resnet18_pt, (1,3,32,32), False, activation_bits=32, trainable_param_bits=32, frozen_param_bits=32, batch_size=args["batch_size"])
    # print("fp32", memory_cost)
    # if args['tta'] == 'quantours': 
    #     resnet18_pt = prep_model(resnet18_pt, "quantours")
    #     file_path = "resnet18_test_quant.pth"
    if args["precision"] == "int8" or 'quant' in args['tta']:
        backend = "qnnpack"
        modules_to_fuse = ['conv1', 'bn1']
        resnet18_pt = torch.ao.quantization.fuse_modules(resnet18_pt, modules_to_fuse)
        resnet18_pt.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        resnet18_quant = torch.quantization.prepare(resnet18_pt, inplace=False)
        torch.quantization.convert(resnet18_quant, inplace=True)
        resnet18_quant.load_state_dict(torch.load(file_path))
        # resnet18_quant = torch.jit.load(f"{args['model']}_cifar10_quant.pt")
        print("loaded state dict")
        DEVICE = "cpu"
        # memory_cost, _ = profile_memory_cost(resnet18_quant, (1,3,32,32), False, activation_bits=8, trainable_param_bits=8, frozen_param_bits=8, batch_size=args["batch_size"])
        print_model_size(resnet18_quant)
        # print("int8", memory_cost)
        # torch.jit.save(torch.jit.script(resnet18_quant), "resnet18.pth")
    else:
        if args["use_cuda"] == True:
            DEVICE = "cuda"
        else:
            DEVICE = "cpu"
        resnet18_quant = resnet18_pt
    
    if DEVICE == "cuda" and torch.cuda.is_available():
        print(DEVICE)
        resnet18_quant.to(DEVICE)
    
    # total = 0
    # num_repeats = 3
    # for ii in range(num_repeats):
    #     total += check_accuracy(resnet18_quant, dataset["val"], batch_size, scale, 1, with_adaptation=True, update_index=ii)
    # print("Final average:", total / num_repeats)
    if args["exp_type"] in ['timing', 'mix_shifts', 'gradual', 'search', 'layer']:
        if args["exp_type"] == 'layer':
            print("NOTICE: ONLY DOING PARTIAL ADAP, PLEASE CHANGE new_runs.py")
        # file_name = f"{args['model']}_{args['class_num']}_{args['precision']}_{args['exp_type']}_{args['tta']}_{args['batch_size']}.txt"
        file_name = f"{args['model']}_{args['class_num']}_{args['exp_type']}.txt"
        dataset, dataset_sequence = generate_dataset(args)
        # check_accuracy(deepcopy(resnet18_quant), dataset, args, with_adaptation=False)
        tta_mode = args["tta"]
        args["tta"] = tta_mode

        if DEVICE == "cuda" and torch.cuda.is_available():
            mod = deepcopy(resnet18_quant).cuda()
        else:
            mod = deepcopy(resnet18_quant)

        if args["exp_type"] == "search":
            for tau in [0,0.1,0.3,0.5,0.7,0.9,1.0]:
                for averager in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]:# [0, 0.1, 0.5, 0.9, 1.0]:
                    acc, _ = check_accuracy(mod, dataset, args, with_adaptation=True, tau=tau, averager=averager)
                    mod = deepcopy(resnet18_quant)
                    print(acc, averager, tau)
                    with open(f"{args['model']}_search.txt", "a") as f:
                        f.write(f"{tau},{averager},{acc}\n")
        else:
            acc, timestamps = check_accuracy(mod, dataset, args, with_adaptation=True)
            print("Time:", np.mean(timestamps), np.std(timestamps))
            if args["save"]:
                with open(file_name, "a") as f:
                    if args['exp_type'] == 'layer':
                        f.write(f"{args['tta']},{args['stop_layer']},{acc},{np.mean(timestamps)}, {np.std(timestamps)}\n")
                    else:
                        f.write(f"{args['tta']},{args['batch_size']},{acc},{np.mean(timestamps)}, {np.std(timestamps)}\n")
    elif args['exp_type'] == 'episodic':
        file_name = f"{args['model']}_{args['class_num']}_episodic.txt"
        for corruption_name in BENCHMARK_CORRUPTIONS:
            for i in range(5, 6):
                if "none" in corruption_name:
                    dataset = torchvision.datasets.CIFAR10("data/cifar10", train=False,
                                                    transform=ImageTransform()['val'], download=True)
                else:
                    dataset = CIFAR10CorruptDataset(f"{corruption_name}_{i}", split=args["data_size"],
                                                    transform=ImageTransform()["val"])
    
                print(f"{corruption_name}_{i}")
                if DEVICE == "cuda":
                    model = deepcopy(resnet18_quant).cuda()
                else:
                    model = deepcopy(resnet18_quant)
                acc, _ = check_accuracy(model, dataset, args, with_adaptation=True)
                if args["save"]:
                    with open(file_name, "a") as f:
                        f.write(f"{args['tta']},{args['batch_size']},{corruption_name},{i},{acc}\n")

    print("finished", sys.argv[0])
    




