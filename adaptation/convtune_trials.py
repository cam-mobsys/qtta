"""
Fine-tuning the convolutional layers with entropy-based loss on an FP32 model as
proof-of-concept that entropy minimization is insufficient for convolutional weights.
"""
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
device="cuda"
import importlib

# pytorch version
def calibration(y, p_mean, num_bins=10):
    """Compute the calibration. -- https://github.com/google-research/google-research/blob/master/uncertainties/sources/postprocessing/metrics.py
    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263
    Args:
        y: ground truth class label, size (?, 1)
        p_mean: numpy array, size (?, num_classes)
                containing the mean output predicted probabilities
        num_bins: number of bins
    Returns:
        cal: a dictionary
             {reliability_diag: realibility diagram
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
             }
    """
    # Compute for every test sample x, the predicted class.
    class_pred = p_mean.argmax(dim=1)
    # and the confidence (probability) associated with it.
    conf = torch.max(p_mean, dim=1)[0]

    # Storage
    acc_tab = torch.zeros(num_bins)  # empirical (true) confidence
    mean_conf = torch.zeros(num_bins)  # predicted confidence
    nb_items_bin = torch.zeros(num_bins)  # number of items in the bins
    tau_tab = torch.linspace(0, 1, num_bins+1)  # confidence bins

    for i in range(num_bins):  # iterate over the bins

        # select the items where the predicted max probability falls in the bin
        sec = (tau_tab[i] < conf) & (conf <= tau_tab[i + 1])
        nb_items_bin[i] = sec.sum()  # Number of items in the bin
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # average of the predicted max probabilities
        mean_conf[i] = conf[sec].mean() if nb_items_bin[i] > 0 else float("nan")
        # compute the empirical confidence
        acc_tab[i] = (class_pred_sec == y_sec).sum()/len(y_sec) if nb_items_bin[i] > 0 else float("nan")

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Reliability diagram
    reliability_diag = (mean_conf, acc_tab)

    weights = nb_items_bin.type(torch.float64) / nb_items_bin.sum()
    if weights.sum() == 0.0:
        print(nb_items_bin.type(torch.float64))
        weights = torch.ones(nb_items_bin.type(torch.float64)) / num_bins

    # Expected Calibration Error
    try:
        ece = ((mean_conf - acc_tab).abs() * weights).sum() / weights.sum()
        # Maximum Calibration Error
        mce = (mean_conf - acc_tab).abs().max()
    except ZeroDivisionError as e:
        # ece = 0.0
        # mce = 0.0
        raise e
    # Saving
    cal = {'reliability_diag': reliability_diag,
           'ece': ece,
           'mce': mce}
    return cal

from tt.algorithm.core.utils.partial_backward import apply_backward_config
from tt.algorithm.core.utils.basic import DistributedMetric, accuracy, AverageMeter

def train_one_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler):
  model.train()
  if torch.cuda.is_available() and device == "cuda":
    data_loader['train'].sampler.set_epoch(epoch) # this is to shuffle when dist/multiGPU is used.
    train_loss = DistributedMetric('train_loss')
    train_top1 = DistributedMetric('train_top1')
    tqdm_disable = True if dist.rank() > 0 else False
  else:
      # when cpu is used, data_loader itself set the shuffle argument to shuffle the data per epoch
    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    tqdm_disable = False

  cals, losses, val_losses, val_cals = [], [], [], []
  ## freeze part of the model
  for n, m in model.named_parameters():
    # if "bn" in n:
    #   m.requires_grad = False
    #   m.track_running_stats = False
    if "13" in n or n[:2] == "4.":
      m.requires_grad = True
      m.track_running_stats = True
    else:
      m.requires_grad = False
      m.track_running_stats = False

  with tqdm(total=len(data_loader['train']),desc='Train Epoch #{}'.format(epoch + 1)) as t:
    for mb_idx, (images, labels) in enumerate(data_loader['train']):
      if torch.cuda.is_available() and device == "cuda":
        images, labels = images.cuda(), labels.cuda()

      optimizer.zero_grad()

      output = model(images)
      print(output.shape)
      try:
        loss = criterion(output, labels)
      except:
        loss = criterion(output)
      # cals.append(calibration(labels, torch.nn.functional.softmax(output, dim=1))["ece"])
      losses.append(loss.detach().cpu())

      # if configs.run_config.do_tent:
      # # if True: # testing
      #   # TODO CXD: Need a more time-efficient way to do model.modules().iterate => parse
      #   conv_layer_params = []
      #   for name, module in model.named_modules():
      #     if ".conv" in name or name[:2] == "4.":
      #       conv_layer_params+=list(module.parameters())
      #   loss.backward(inputs=conv_layer_params[-2:])
      # else:
      #   loss.backward()
      loss.backward()

      if hasattr(optimizer, 'pre_step'):  # for SGDScale optimizer
          optimizer.pre_step(model)
      optimizer.step()
      if hasattr(optimizer, 'post_step'):  # for SGDScaleInt optimizer
          optimizer.post_step(model)

      # after one step
      train_loss.update(loss, images.shape[0])
      acc1 = accuracy(output, labels, topk=(1,))[0]
      train_top1.update(acc1.item(), images.shape[0])
      t.set_postfix({
          'loss': train_loss.avg.item(),
          'top1': train_top1.avg.item(),
          'batch_size': images.shape[0],
          'img_size': images.shape[2],
          'lr': optimizer.param_groups[0]['lr'],
      })
      t.update()
      # after step (NOTICE that lr changes every step instead of epoch)
      lr_scheduler.step()

      # adding validation to track calibration
      if mb_idx % 10 == 0:
        with torch.no_grad():
          val_imgs, val_labels = next(iter(data_loader["val"]))
          if torch.cuda.is_available() and device == "cuda":
            val_imgs, val_labels = val_imgs.cuda(), val_labels.cuda()
          val_output = model(val_imgs)
          try:
            val_loss = criterion(val_output, val_labels)
          except:
            val_loss = criterion(val_output)
          # val_cals.append(calibration(val_labels.cuda(), torch.nn.functional.softmax(val_output, dim=1))["ece"])
          val_losses.append(val_loss.detach().cpu())

  return {
      'train/top1': train_top1.avg.item(),
      'vals': val_losses,
      'vcals': val_cals,
      'cals': cals,
      'losses': losses
  }, model, criterion, optimizer, lr_scheduler

def entropy_min_loss(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)

def validate_model(trial_num, model, val_set, optimizer, batch_size, scale="fp32", do_save=False, loss_name="crossent", wt=None):
  acc = 0
  seen = 0
  model.train()
  # turn on/off batch norm => off for normal CIFAR10-C evaluations
  parms = []
  for i, (n, m) in enumerate(model.named_parameters()):
    if ("bn" in n or ".conv." in n) and i > 134:
      m.requires_grad = True
      parms.append(m)
    else:
      m.require_grad = False
      #m.track_running_stats = True
  optimizer = torch.optim.Adam(parms)
  # # Sanity Check
  # for name, module in model.named_modules():
  #   if "bn" in name and hasattr(module, 'training') and module.training:
  #     print('{} is training {}'.format(name, module.training))
  #   if "bn" in name and module.track_running_stats:
  #     print(f"{name} is tracking stats")
  pre_acc , pre_count= 0, 0
  outputs = []
  pre_outputs = []
  max_mem_used = 0
  entropies = []
  cals = []
  losses = []
  labels_total = []
  #criterion = torch.nn.CrossEntropyLoss()
  criterion = entropy_min_loss
  # model.cpu()
  with tqdm(total=4000, desc="Validate") as t:
    if True: #with torch.no_grad():
      for idx, (images, labels) in enumerate(val_set):
        if idx >= 4000:
          break
        if torch.cuda.is_available() and device == "cuda": 
          images, labels = images.cuda(), labels.cuda()
        model.eval()
        output = model(images)
        for idx, logit in enumerate(output):
            if logit.argmax() == labels[idx]:
                pre_acc += 1
            pre_count += 1
        #outputs.append(output.cpu())
        labels_total.append(labels.cpu())
        #try:
        pre_outputs.append(output.detach().cpu())
        #loss = criterion(output, labels)
        #except:
        if idx % 5 == 0:
            model.train()
            loss = criterion(output).mean(0)
            loss.backward()
            optimizer.step()
        #try:
         # cal = calibration(labels.cpu(), torch.nn.functional.softmax(output.cpu(), dim=1))
          #cals.append(cal["ece"])
        #except Exception as e:
         # pass
          # print(labels.shape, images.shape)
        
            losses.append(loss.detach().cpu())
        
            model.eval()
            with torch.no_grad():
                output = model(images)
            model.load_state_dict(wt)
        outputs.append(output.detach().cpu())
        for idx, logit in enumerate(output):
          if logit.argmax() == labels[idx]:
            acc += 1
          seen += 1
          # entropies.append(entropy(torch.softmax(logit.detach().cpu(), dim=0)))
          t.set_postfix({'pre_acc': pre_acc*100/pre_count, 'accuracy': 100 * acc / seen, 'entropy': np.mean(entropies), 'ece': np.mean(cals)})
          t.update()
          max_mem_used = max(max_mem_used, torch.cuda.max_memory_allocated("cuda"))


  do_save = True

  if do_save:
    np.save(f"ft/{configs.data_provider.dataset.split('/')[-1]}_{scale}_{loss_name}_b{batch_size}_{trial_num}.npy",\
            {'acc': acc/seen, 'losses': losses,"labels":labels_total, "pre_outputs": pre_outputs,  'outputs': outputs, "mem": max_mem_used})
    # np.save(f"{configs.data_provider.dataset.split('/')[-1]}_{scale}_{loss_name}_b{batch_size}_{trial_num}.npy",\
    #         {'acc': 100*acc/seen, 'ece': cals, 'mem': max_mem_used, "entropy": entropies})
  return {'acc': 100*acc/seen, 'losses': losses, "cals": cals, "entropy": np.mean(entropies)}

def test_cifar10c(corruption_name, model, batch_sizes=[4], loss_name="enttrain5", wt=None):
  configs.data_provider.dataset = f"cifar10_corrupted/{corruption_name}"
  configs.data_provider.root = f"/content/dataset/cifar10c/"
  configs.run_dir = f"/content/runs/cifar10c_{corruption_name}/{configs.run_config.optimizer_name}/"
  configs.data_provider.n_worker = 0
  criterion = torch.nn.CrossEntropyLoss()
  print("...building optimizer...")
  optimizer = build_optimizer(model)
  print("building dset...")
  dataset = build_dataset()
  lr_scheduler = build_lr_scheduler(optimizer, len(dataset['train']))

  for batch_size in batch_sizes:
    data_loader = dict()
    for split in dataset:
      use_shuffle = None
      if torch.cuda.is_available() and device == "cuda":
        sampler = torch.utils.data.DistributedSampler(
          dataset[split],
          num_replicas=dist.size(),
          rank=dist.rank(),
          # seed=configs.manual_seed,
          shuffle=(split == 'train'))
      else:
        sampler = None
        use_shuffle = not configs.evaluate

      data_loader[split] = torch.utils.data.DataLoader(
          dataset[split],
          shuffle=use_shuffle,
          batch_size=batch_size,
          sampler=sampler,
          num_workers=configs.data_provider.n_worker,
          pin_memory=True,
          drop_last=(split == 'train'),
      )

    ground_truth = []
    predicted = []
    entropies = []
    acc, seen = 0, 0

    if torch.cuda.is_available():
      model.cuda()
    
    print(batch_size, loss_name, corruption_name)
    for i in range(1):
      validate_model(i, model, data_loader["val"], optimizer=optimizer, batch_size=batch_size, do_save=True, loss_name=loss_name, wt=wt)

if __name__ == "__main__":
  print("starting main...")
  model, image_size, description = build_model(net_id="mbv2-w0.35", pretrained=True)
  print("...loaded model wireframe...")

  from mcunet.mcunet.tinynas.nn.modules import LinearLayer
  droupout_rate = 0.2
  n_classes = 10
  ll = LinearLayer(448, n_classes, dropout_rate=droupout_rate)
  model.classifier = ll
  wts = torch.load("/home/cynthia/mcunet_cifar.pth")
  model.load_state_dict(wts)
  device = "cuda"
  print(torch.cuda.is_available())
  msize = 0
  for param in model.parameters():
    msize += param.numel()
  print(msize)
  model.to(device)
  print("model allocated")

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

  for corruption in BENCHMARK_CORRUPTIONS: 
    for i in range(1, 6):
      print(f"{corruption}_{i}")
      test_cifar10c(f"{corruption}_{i}", model, [1], wt=wts)
