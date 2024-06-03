import numpy as np
import pandas
import torch
import math
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import entropy
import glob

# THIS SCRIPT IS FOR ANALYSIS OF FP32 Entropy loss training data and power consumption.

def main():
    batch_size = sys.argv[1]
    print(sys.argv[2])
    files = sorted(glob.glob(sys.argv[2]))
    if len(files) <= 2:
        raise Exception("not the right glob, mi amia")
    
    pre_ents, post_ents = [], []
    pre_accs, post_accs = [], []
    for idx, f in enumerate(files):
        results = np.load(f, allow_pickle=True)[()]
        labels = results["labels"]
        right, seen = 0, 0
        for o, l in zip(results["pre_outputs"], labels):
            for o_, l_ in zip(o, l):
                if o_.argmax() == l_:
                    right += 1
                seen += 1
        pre_accs.append(right/seen)
        smax = torch.softmax(torch.concatenate(results["pre_outputs"]), dim=1)
        pre_ents.append(np.mean(entropy(smax, axis=1)))

        right, seen = 0, 0
        for o, l in zip(results["outputs"], labels):
            for o_, l_ in zip(o, l):
                if o_.argmax() == l_:
                    right += 1
                seen += 1
        post_accs.append(right/seen)
        smax = torch.softmax(torch.concatenate(results["outputs"]), dim=1)
        post_ents.append(np.mean(entropy(smax, axis=1)))
    print(f"Accuracy: {np.mean(pre_accs)}, {np.std(pre_accs)}, {np.mean(post_accs)}, {np.std(post_accs)}")
    print(f"Entropy: {np.mean(pre_ents)}, {np.std(pre_ents)}, {np.mean(post_ents)}, {np.std(post_ents)}")

    print("real std acc", np.std((np.array(post_accs) - np.array(pre_accs))/np.array(pre_accs)))
    print("real std ent", np.std((np.array(post_ents) - np.array(pre_ents))/np.array(pre_ents)))

    print("real diff acc", np.mean((np.array(post_accs) - np.array(pre_accs))/np.array(pre_accs)))
    print("real diff ent", np.mean((np.array(post_ents) - np.array(pre_ents))/np.array(pre_ents)))


def integrate_power(filename):
    temp = np.load(filename, allow_pickle=True)[()]
    t_del = np.diff(temp["t"])
    print(t_del)
    wh = [t*p/3600 for t, p in zip(t_del, temp["p"][1:])]
    print(filename, np.sum(wh), np.mean(wh), np.max(wh))



if __name__ == "__main__":
    if sys.argv[1] == "enttrain":
        main()
    else:
        integrate_power(sys.argv[1])
