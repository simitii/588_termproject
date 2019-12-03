# adapted from https://github.com/locuslab/convex_adversarial/blob/master/examples/trainer.py

import torch
import torch.nn as nn
from torch.autograd import Variable
from convex_adversarial.convex_adversarial import robust_loss, robust_loss_parallel
import torch.optim as optim

import numpy as np
import time
import gc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_instance_adaptive_robust(instance_adaptive_loader, model, opt, epoch, log, verbose, 
                real_time=False, clip_grad=None, **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()
    epsilons = AverageMeter()

    model.train()

    end = time.time()
    for i, ([X],[y], [epsilon]) in enumerate(instance_adaptive_loader):
        X,y, epsilon = X.cuda(), y.cuda().long(), epsilon.item()
        if y.dim() == 2: 
            y = y.squeeze(1)
        data_time.update(time.time() - end)

        with torch.no_grad(): 
            out = model(Variable(X))
            ce = nn.CrossEntropyLoss()(out, Variable(y))
            err = (out.max(1)[1] != y).float().sum()  / X.size(0)


        robust_ce, robust_err = robust_loss(model, epsilon, 
                                             Variable(X), Variable(y), 
                                             **kwargs)
        opt.zero_grad()
        robust_ce.backward()


        if clip_grad: 
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        opt.step()

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        robust_losses.update(robust_ce.detach().item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))
        epsilons.update(epsilon, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, epsilon, robust_ce.detach().item(), 
                robust_err, ce.item(), err.item(), file=log)

        if verbose and (i % verbose == 0 or real_time): 
            endline = '\n' if i % verbose == 0 else '\r'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Epsilon {epsilon.val:.3f} ({epsilon.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i, len(instance_adaptive_loader), batch_time=batch_time,
                   data_time=data_time, epsilon=epsilons, loss=losses, errors=errors, 
                   rloss = robust_losses, rerrors = robust_errors), end=endline)
        log.flush()

        del X, y, robust_ce, out, ce, err, robust_err

    print('')
    torch.cuda.empty_cache()