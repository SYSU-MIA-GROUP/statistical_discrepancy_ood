from __future__ import print_function

import os
import time
from tqdm import tqdm
import numpy as np
import torch

from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model


def forward_fun(args):
    def forward_features(inputs, model):
        if args.model_arch in {'mobilenet'}:
            feat = model.intermediate_forward(inputs)
        elif args.model_arch.find('resnet') > -1:
            feat = model.forward_features(inputs)
        else:
            logit = model(inputs)
        return feat

    return forward_features



args = get_args()
forward_features = forward_fun(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def statistic_ind(args, mode_args):
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    method = args.method # element_mean
    method_args = args.method_args
    name = args.name  # resnet50

    in_save_dir = os.path.join(base_dir, name, method, in_dataset)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_loader_in(args, split='train')
    loaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.num_classes
    method_args['num_classes'] = num_classes
    model = get_model(args, num_classes, load_ckpt=True)
    time.time()

    if True:
        # f1 = open(os.path.join(in_save_dir, "in_features.txt"), 'w')
        ########################################In-distribution###########################################
        print("Processing in-distribution images")
        n = len(loaderIn.dataset)
        count = 0
        feat_list = []
        label_list = []
        for j, data in enumerate(loaderIn):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images.float()

            with torch.no_grad():
                features = forward_features(inputs, model)
                features = features.detach().cpu().numpy()
            feat_list.append(features)
            label_list.append(labels.detach().cpu().numpy())
            count += curr_batch_size
            # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, n, time.time()-t0))
            time.time()
        total_feat = np.concatenate(feat_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)

        np.save(os.path.join(in_save_dir, 'in_features_train'), total_feat)
        np.save(os.path.join(in_save_dir, 'train_labels'), label_list)
    return


def statistic_ind_val(args, mode_args):
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    method = args.method  # element_mean
    method_args = args.method_args
    name = args.name  # resnet50

    in_save_dir = os.path.join(base_dir, name, method, in_dataset)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_loader_in(args, split='val')
    loaderIn, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    method_args['num_classes'] = num_classes
    model = get_model(args, num_classes, load_ckpt=True)
    time.time()

    if True:
        # f1 = open(os.path.join(in_save_dir, "in_features.txt"), 'w')
        ########################################In-distribution###########################################
        print("Processing in-distribution test images")
        n = len(loaderIn.dataset)
        print(f'the length of dataset is {n}')
        count = 0
        feat_list = []
        for j, data in tqdm(enumerate(loaderIn)):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images.float()

            with torch.no_grad():
                features = forward_features(inputs, model)
                features = features.detach().cpu().numpy()
                feat_list.append(features)

            count += curr_batch_size
            # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, n, time.time()-t0))
            time.time()
        features_all = np.concatenate(feat_list, axis=0)
        np.save(os.path.join(in_save_dir, 'in_features_test'), features_all)
    return


def statistic_out(args, mode_args):
    base_dir = args.base_dir  
    ood_datasets = args.out_datasets
    method = args.method  # element_mean
    method_args = args.method_args
    name = args.name  # resnet50
    in_dataset = args.in_dataset

    for ood_dataset in ood_datasets:
        out_save_dir = os.path.join(base_dir, name, method, in_dataset, ood_dataset)
        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        loader_out_dict = get_loader_out(args, (None, ood_dataset), split='val')
        loaderOut, num_classes = loader_out_dict.val_ood_loader, 1000
        method_args['num_classes'] = num_classes
        model = get_model(args, num_classes, load_ckpt=True)
        time.time()

        if True:
            # f1 = open(os.path.join(in_save_dir, "in_features.txt"), 'w')
            ########################################In-distribution###########################################
            print("Processing out-distribution images")
            n = len(loaderOut.dataset)
            count = 0
            feat_list = []
            for j, data in enumerate(loaderOut):
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                curr_batch_size = images.shape[0]

                inputs = images.float()

                with torch.no_grad():
                    features = forward_features(inputs, model)
                    features = features.detach().cpu().numpy()
                feat_list.append(features)
                count += curr_batch_size
                # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, n, time.time()-t0))
                time.time()
            total_feat = np.concatenate(feat_list, axis=0)
            np.save(os.path.join(out_save_dir, 'out_features'), total_feat)
    return


if __name__ == '__main__':
    args.method_args = dict()
    mode_args = dict()
    statistic_ind(args, mode_args)
    statistic_ind_val(args, mode_args)
    statistic_out(args, mode_args)
    print('echo finished statistic')
