import os
import torch.nn as nn
import argparse
import lipt.nets as nets
import lipt.datasets as datasets
import lipt.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep
import torchvision.datasets as tdatasets
import numpy as np
import math
import copy

from torch import tensor, long
import matplotlib.pyplot as plt

def save_indices(args, subset, exp):

    if (args.dataset == "ImageNet" and args.easy_epochs == 100) or (args.dataset == "CIFAR10" and args.easy_epochs == 200):
        curri_or_not = "nocl"
    else: 
        curri_or_not = "cl"
        
    simple_selection_name = ''
    if "Wa" in args.selection: # "GraNdProbWa", "GlisterWa":
        if args.selection == "GraNdProbWa":
            simple_selection_name = "grand"
        else:
            simple_selection_name = args.selection[:-2].lower()
    
        pretrain_config = ''
        averaging_config = '{}{}{}'.format(args.wa_pre_epochs, args.selection_coeff, args.selection_epochs)
        if args.rand_augment:
            augment_string = ''
            augment_string += 'o' if args.augment_first else 'x'
            augment_string += 'o' if args.augment_second else 'x'
            augment_string += 'o' if args.augment_third else 'x'
    
            augment_config = '-{}{}{}_{}'.format(args.aug_n, args.aug_m, args.augment_coeff, augment_string)
        else:
            augment_config = ''
        if args.rand_pruning:
            pruning_config = '-{}_{}'.format(args.sparse_init, args.density)
        else:
            pruning_config = ''
    else:
        simple_selection_name = args.selection.lower()
        pretrain_config = '{}ep'.format(args.selection_epochs)
        averaging_config = ''
        augment_config = ''
        pruning_config = '' 
    
    np.save('{}/{}/{}-{}/{}-{}{}{}{}-{}p-{}.npy'.format(
                args.indices_path,
                simple_selection_name,
                args.dataset.lower(), args.model.lower(),
                curri_or_not,  
                pretrain_config, averaging_config, augment_config, pruning_config, 
                int(args.fraction*100), 
                int(args.seed) * int(exp+1)), 
                subset["indices"]) 


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    add_general_args(parser)
    add_selection_args(parser)
    add_sparse_args(parser)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    start_exp = 0
    start_epoch = 0

    avg_best_acc = 0.0
    best_prec1 = 0.0

    for exp in range(start_exp, args.num_exp):
        print('\n================== Exp %d ==================\n' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_ex: ",
              args.num_exp, ", epochs: ", args.epochs, ", fraction: ", args.fraction, ", seed: ", int(args.seed)*int(exp+1),
              ", lr: ", args.lr, ", device: ", args.device, "\n", sep="")

        print("model, dataset", args.model, args.dataset)
        print("selection method:", args.selection)
        print("baseline - pretraining epochs:", args.selection_epochs)
        print("proposed - avg args:", args.wa_num_models, args.wa_pre_epochs, args.selection_coeff, args.selection_epochs)
        print("proposed - aug args:", args.rand_augment, 
                                      args.aug_n, args.aug_m, args.augment_first, args.augment_second, args.augment_third)
        print("proposed - pruning args:", args.rand_pruning, args.sparse_init, args.density, args.prune_first, args.prune_second)
        print("dynamic enabled:", args.dynamic, args.reselect)
        print("seed:", int(args.seed)*int(exp+1))
      
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
            (args.data_path)
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

        torch.random.manual_seed(int(args.seed)*int(exp+1))

        if args.dataset == "ImageNet":
            easydir = os.path.join(args.data_path, 'train')
            valdir = os.path.join(args.data_path, 'val')

            easynormalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            imagenet_default_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        easynormalize,
                    ])

            imagenet_random_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandAugment(num_ops=args.aug_n, magnitude=args.aug_m),
                        transforms.ToTensor(),
                        easynormalize,
                    ])

            if args.rand_augment and args.augment_first:
                easy_dataset = tdatasets.ImageFolder(easydir, imagenet_random_transform)
            else:
                easy_dataset = tdatasets.ImageFolder(easydir, imagenet_default_transform) 

            val_dataset = tdatasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    easynormalize,
                ])) 

            dst_train = easy_dataset
            dst_train.targets = tensor(dst_train.targets, dtype=long)
            dst_test = val_dataset

        selection_args = dict(epochs=args.selection_epochs,
                              selection_method=args.uncertainty,
                              balance=args.balance,
                              greedy=args.submodular_greedy,
                              function=args.submodular)
        method = methods.__dict__[args.selection](dst_train, args, args.fraction, 
                                                  int(args.seed)*int(exp+1), dst_test=dst_test, **selection_args)

        subset = method.select()

        if args.save_indices:
            save_indices(args, subset, exp)

        dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])

        train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                       num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)

        model = args.model

        network = nets.__dict__[model](channel, num_classes, im_size).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
            quit()
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu[0])
            network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
        elif torch.cuda.device_count() > 1:
            network = nets.nets_utils.MyDataParallel(network).cuda()

        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

        # Optimizer
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                             weight_decay=args.weight_decay, nesterov=args.nesterov)

        # LR scheduler
        if args.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
                                                                   eta_min=args.min_lr)
        elif args.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
                                                        gamma=args.gamma)
        else:
            scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
        scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

        rec = init_recorder()

        best_prec1 = 0.0

        _ttime = 0.0
        for epoch in range(start_epoch, args.epochs):
            if args.dynamic and epoch > 0 and epoch % args.reselect == 0:
                train_loader = reselect(args, exp, network, dst_train, dst_test, subset,
                                criterion, optimizer, scheduler, train_loader, epoch)
                continue # WARNING this may disturb updating prec1

            # train for one epoch
            endt = time.time()
            train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)
            _ttime = time.time() - endt

            # evaluate on validation set
            if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                prec1 = test(test_loader, network, criterion, epoch, args, rec)
                if prec1 > best_prec1:
                    best_prec1 = prec1

        print('_ttime:', _ttime)

        print('| Best accuracy: ', best_prec1, ", on model " + "\n\n")
        final_acc = test(test_loader, network, criterion, args.epochs, args, rec)
        print('| Final accuracy: ', final_acc, ", on model " + "\n\n")
        start_epoch = 0
        sleep(2)

        avg_best_acc += best_prec1


    print('| Avg best accuracy:', avg_best_acc / args.num_exp)

if __name__ == '__main__':
    main()
