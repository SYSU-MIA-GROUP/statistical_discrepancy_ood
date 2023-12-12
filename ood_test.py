import os
import matplotlib.colors
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import mmcv
from os.path import dirname
from models.resnet import resnet50
from numpy.linalg import norm, pinv
import torch.backends.cudnn as cudnn
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.covariance import EmpiricalCovariance
from scipy.special import logsumexp
from sklearn import metrics
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import FastICA

device = torch.device('cuda:1')
args = get_args()


def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)
    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1
    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)  
    return num_fp, thresh


def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh


def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def benchmark():
    recall = 0.95
    result = []
    id_data = "imagenet"
    num_classes = 1000
    args.model_arch = 'resnet50'
    file_path = f"../features/{args.model_arch}/" # your path to save the feature vector

    if os.path.isfile(os.path.join(file_path, id_data, 'weight.npy')):
        w = np.load(os.path.join(file_path, id_data, 'weight.npy'))
    else:
        model = get_model(args, num_classes, load_ckpt=True)
        w = model.fc.weight.cpu().detach().numpy()
        np.save(os.path.join(file_path, id_data, 'weight.npy'), w)
    if os.path.isfile(os.path.join(file_path, id_data, 'bias.npy')):
        b = np.load(os.path.join(file_path, id_data, 'bias.npy'))
    else:
        model = get_model(args, num_classes, load_ckpt=True)
        b = model.fc.bias.cpu().detach().numpy()
        np.save(os.path.join(file_path, id_data, 'bias.npy'), b)

    feature_id_train = np.load(f"{file_path}/{id_data}/in_features_train.npy")

    theshold = np.quantile(feature_id_train, 0.92)

    feature_id_val = np.load(f"{file_path}/{id_data}/in_features_val.npy")


    OOD_data_list = ['inat', 'sun50', 'places50', 'dtd']
    feature_oods = {name: np.load(f"{file_path}/{id_data}/{name}/out_features.npy") for name in OOD_data_list}

    method = 'element-mean'
    logit_id_val = feature_id_val.clip(min=None, max=theshold) @ w.T + b

    logit_oods = {name: feat.clip(min=None, max=1.0) @ w.T + b for name, feat in feature_oods.items()}

    score_id = logsumexp(logit_id_val, axis=-1) * (1 + np.mean(feature_id_val, axis=-1))
    # ------------------------ compute for ood data -------------------------
    for name, feature_ood, logit_ood in zip(OOD_data_list, feature_oods.values(),logit_oods.values()):
        score_ood = logsumexp(logit_ood, axis=-1) * (1 + np.mean(feature_ood, axis=-1))
        auc_ood, aupr_ood = auc(score_id, score_ood)[0:2]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood, aupr=aupr_ood))
        print(f'{method} - {name} FPR95: {fpr_ood:.2%}, AUROC: {auc_ood:.2%}')

    df = pd.DataFrame(result)
    print(f'{method} - Average FPR95 {df.fpr.mean():.2%}, AUROC {df.auroc.mean():.2%}')


if __name__ == '__main__':
    benchmark()