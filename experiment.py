# adapted from https://github.com/locuslab/convex_adversarial/blob/master/examples/fashion_mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import argparse

import convex_adversarial.examples.problems as pblm
from convex_adversarial.examples.trainer import *

from src.adaptive_robust_training import train_instance_adaptive_robust
from src.adaptive_epsilon import adaptive_epsilon

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--starting_epsilon", type=float, default=None)
    parser.add_argument('--prefix', default="fashion_mnist_experiment")
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--verbose', type=int, default='1')
    parser.add_argument('--alpha_grad', action='store_true')
    parser.add_argument('--scatter_grad', action='store_true')
    parser.add_argument('--l1_proj', type=int, default=None)
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--vgg', action='store_true')
    args = parser.parse_args()


    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    fashion_train = datasets.FashionMNIST("./fashion_mnist", train=True,
       download=True, transform=transforms.ToTensor())
    fashion_test = datasets.FashionMNIST("./fashion_mnist", train=False,
       download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(fashion_train, batch_size=1, shuffle=True, pin_memory=True)
    # key difference is here:
    instance_adaptive_loader = adaptive_epsilon(train_loader, args.epsilon, args.batch_size)
    test_loader = torch.utils.data.DataLoader(fashion_test, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.large: 
        model = pblm.mnist_model_large().cuda()
    elif args.vgg: 
        model = pblm.mnist_model_vgg().cuda()
    else: 
        model = pblm.mnist_model().cuda()

    opt = optim.Adam(model.parameters(), lr=args.lr)

    for t in range(args.epochs):
        train_instance_adaptive_robust(instance_adaptive_loader, model, opt, t, train_log, 
            args.verbose, 
            args.alpha_grad, args.scatter_grad, l1_proj=args.l1_proj)
        evaluate_robust(test_loader, model, args.epsilon, t, test_log, args.verbose)
        torch.save(model.state_dict(), args.prefix + "_model.pth")