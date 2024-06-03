import os
from tqdm import tqdm
import numpy as np
import torch
# torch._C._cuda_init()
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, log_loss
import sys, json, os, itertools
import torchvision
from scipy.stats import entropy
import tensorflow as tf
from mcunet.mcunet.model_zoo import build_model
from tt.algorithm.quantize.custom_quantized_format import build_quantized_network_from_cfg
from tt.algorithm.quantize.quantize_helper import create_scaled_head, create_quantized_head
from tt.algorithm.core.utils import dist
from tt.algorithm.core.model import build_mcu_model
from tt.algorithm.core.utils.config import configs, load_config_from_file, update_config_from_args, update_config_from_unknown_args
from tt.algorithm.core.dataset import build_dataset
from tt.algorithm.core.optimizer import build_optimizer
from tt.algorithm.core.trainer.cls_trainer import ClassificationTrainer
from tt.algorithm.core.builder.lr_scheduler import build_lr_scheduler
import importlib
from tt.algorithm.core.utils.partial_backward import apply_backward_config, parsed_backward_config
from tt.algorithm.core.utils.basic import DistributedMetric, accuracy, AverageMeter

device="cuda"


def entropy_min_loss(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)

def get_dataset(corruption_type):
    configs.data_provider.dataset = f"cifar10_corrupted/{corruption_type}"
    dataset = build_dataset()
    data_loader = dict()
    for split in dataset:
        use_shuffle = None
        if torch.cuda.is_available():
            sampler = torch.utils.data.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            seed=configs.manual_seed,
            shuffle=(split == 'train'))
        else:
            sampler = None
            use_shuffle = not configs.evaluate

        data_loader[split] = torch.utils.data.DataLoader(
            dataset[split],
            shuffle=use_shuffle,
            batch_size=configs.data_provider.base_batch_size,
            sampler=sampler,
            num_workers=configs.data_provider.n_worker,
            pin_memory=True,
            drop_last=(split == 'train'),
        )
    return data_loader

def finetune(model, data_loader, repeat_steps=1):
    # criterion = torch.nn.CrossEntropyLoss()
    configs.run_config.bs256_lr = 0.05
    # configs.run_config.warmup_lr = 0.007
    criterion = entropy_min_loss
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer, len(data_loader['val']))
    # params = []
    # for n, m in model.named_modules():
    # if n in ["4"] or "1.13" in n:
    # if "13" in n or n == "4":
    # params += list(m.parameters())

    # optimizer = torch.optim.SGD(params, lr=0.005)
    losses = []
    results = []
    eval_results = []
    total_labels = []
    seen, right = 0, 0
    prev_seen, prev_right = 0, 0
    with tqdm(total=4000, desc="") as t:
        for idx, (images, labels) in enumerate(data_loader["val"]):
            if idx > 4000:
                break
            if torch.cuda.is_available() and device == "cuda":
                images = images.cuda()
                labels = labels.cuda()

            # make sure it's not using BN (though there's no BN so hm)
            with torch.no_grad():
                model.eval()
                # for n, m in model.named_parameters():
                #     if "bn" in n:
                #         m.requires_grad = False
                #         m.track_running_stats = False
                output = model(images)
                for i,lab in enumerate(labels):
                    print(output[i].argmax())
                    if lab == output[i].argmax():
                        prev_right += 1
                    prev_seen += 1
                prev_acc = prev_right / prev_seen
                # eval_results.append(right/images.shape[0])
                eval_results.append(output.cpu())

            model.train()
            for inner_step in range(repeat_steps):
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output).mean(0)
                # loss = criterion(output, labels)
                loss.backward()
                losses.append(loss.detach().cpu())

                if hasattr(optimizer, 'pre_step'):  # for SGDScale optimizer
                    optimizer.pre_step(model)
                optimizer.step()
                if hasattr(optimizer, 'post_step'):  # for SGDScaleInt optimizer
                    optimizer.post_step(model)
                # torch.save(model.state_dict(), f"chkpts/{configs.data_provider.dataset.split('/')[1]}_entmin_{idx}.pth") 
            
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                output = model(images)
                results.append(output.detach().cpu())
                for i,lab in enumerate(labels):
                    if lab == output[i].argmax():
                        right += 1
                    seen += 1
                total_labels.append(labels.detach().cpu())
            # model.load_state_dict(checkpoint["state_dict"])
            model,_,_ = build_model(net_id="mbv2-w0.35", pretrained=True)
            model.to(device)
            t.set_postfix({"seen":seen, "acc": 100 * right / seen, "diff": (right / seen) - prev_acc})
            t.update()
    return losses, results, eval_results, total_labels

if __name__ == "__main__":
    print("starting main...")
    torch.cuda.set_per_process_memory_fraction(0.3, device=0)
    model, image_size, description = build_model(net_id="mbv2-w0.35", pretrained=True)
    configs.net_config.net_name = "mbv2-w0.35"
    configs.data_provider.root = f"/content/dataset/cifar10c/"
    configs.data_provider.n_worker = 0
    configs.net_config.pretrained=True
    configs.run_config.bs256_lr = 0.001
    configs.run_config.warmup_lr = 0.005
    configs.run_config.optimizer_name = "sgd_scale"
    configs.backward_config.enable_backward_config = 1
    configs.data_provider.image_size = 128 # this must be 128 or the results are bad
    configs.data_provider.num_classes = 10
    configs.run_config.num_epochs = 1
    configs.data_provider.base_batch_size = 16
    configs.backward_config.manual_weight_idx = [45, 46, 47, 48, 49, 50]
    configs.backward_config.n_bias_update = len(configs.backward_config.manual_weight_idx)
    configs.backward_config['weight_update_ratio'] = [None, None, None, None, None, None]
    configs.backward_config.n_weight_update = len(configs.backward_config.manual_weight_idx)
    configs.run_config.eval_per_epochs = 1
    configs.run_config.do_tent = True

    #test_mod = torch.load('tt/algorithm/assets/mcu_models/mbv2-w0.35.pkl','rb')
    # tt_net = build_quantized_network_from_cfg(test_mod)
    # tt_net = create_scaled_head(tt_net, norm_feat=False)
    # checkpoint = torch.load("mbv2w035_chkpt.pth")
    # tt_net.load_state_dict(checkpoint["state_dict"])
    # tt_net.to(device)
    # print("...loaded model wireframe...")

    # if configs.backward_config.enable_backward_config:
        # configs.backward_config = parsed_backward_config(configs.backward_config, tt_net)
    #     apply_backward_config(tt_net, configs.backward_config)
        # switch to training mode => nothing to turn on if int8
    # print("...loaded model...")
    model.to(device)
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

    accs, ents = [], []
    pre_accs, pre_ents = [], []
    for c, corruption in enumerate(BENCHMARK_CORRUPTIONS): 
        for i in range(1, 6):
            print(f"{corruption}_{i}")
            data_loader = get_dataset(f"{corruption}_{i}")
            for trial_num in range(5):
                losses, results, eval_results, labels = finetune(model, data_loader, repeat_steps=1)
                # for er, r, l in zip(eval_results, results, labels):
                #    if 
                np.save(f"ft/{corruption}-{i}_int8_entmin_b{configs.data_provider.base_batch_size}_{trial_num}.npy",\
                    {'pre_outputs': eval_results, 'losses': losses, 'outputs': results, "labels": labels})

