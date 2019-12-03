# adapted from https://github.com/locuslab/convex_adversarial/blob/master/examples/har.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import argparse

import src.problems as pblm
from src.trainer import *
from src.attacks import pgd,fgs

from src.adaptive_robust_training import train_instance_adaptive_robust
from src.adaptive_epsilon import adaptive_epsilon

def replace_10_with_0(y): 
    return y % 10

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--starting_epsilon", type=float, default=None)
    parser.add_argument('--prefix', default="svhn_experiment")
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--verbose', type=int, default='1')
    parser.add_argument('--alpha_grad', action='store_true')
    parser.add_argument('--scatter_grad', action='store_true')
    parser.add_argument('--l1_proj', type=int, default=None)

    args = parser.parse_args()

    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    train = datasets.SVHN("./svhn_data", split='train', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    test = datasets.SVHN("./svhn_data", split='test', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, pin_memory=True)
    # key difference is here:
    instance_adaptive_loader = adaptive_epsilon(train_loader, args.epsilon, args.batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # new svhn
    model = pblm.svhn_model().cuda()

    opt = optim.Adam(model.parameters(), lr=args.lr)

    for t in range(args.epochs):
        train_instance_adaptive_robust(instance_adaptive_loader, model, opt, t, train_log, 
            args.verbose, 
            args.alpha_grad, args.scatter_grad) # l1_proj
        evaluate_robust(test_loader, model, args.epsilon, t, test_log, args.verbose)
        torch.save(model.state_dict(), args.prefix + "_model.pth")

    print("PGD ATTACK:")
    pgd(test_loader, model, args.epsilon, verbose=False, robust=False)
    print("FGS ATTACK:")
    fgs(test_loader, model, args.epsilon, verbose=False, robust=False)