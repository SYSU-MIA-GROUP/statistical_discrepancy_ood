import argparse
import warnings
import torch
warnings.filterwarnings('ignore')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in_dataset', default="imagenet", type=str, help='CIFAR-10/100 | imagenet')
    parser.add_argument('--out_datasets', default=['inat', 'sun50', 'places50', 'dtd'], type=list,
                        help="['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd']")
    parser.add_argument('--name', default="resnet50", type=str, help='neural network name and training set')
    parser.add_argument('--model_arch', default='resnet50', type=str, help='model architecture [resnet50]')
    parser.add_argument('--threshold', default=1.0, type=float, help='sparsity level')
    parser.add_argument('--method', default='energy', type=str, help='odin mahalanobis CE_with_Logst')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lam', type=float, default=1.70)
    parser.add_argument('--feature_mean', type=torch.Tensor, default=None)
    parser.add_argument('--feature_std', type=torch.Tensor, default=None)
    parser.add_argument('--trans', type=torch.Tensor, default=None)
    parser.add_argument('--bats_with_cov', action='store_true')
    parser.add_argument('--shrink_num', type=int, default=256)
    parser.add_argument('--bats', action='store_true')
    parser.add_argument('--forward_react_feature', action='store_true')
    parser.add_argument('--forward_bats_feature', action='store_true')
    parser.add_argument('--cov_r', action='store_true')
    parser.add_argument('--frequency', action='store_true')
    parser.add_argument('--entropy', action='store_true')
    parser.add_argument('--saved_models', type=str, default='./saved_models/')
    parser.add_argument('--loss', type=str, default='celoss', help='celoss, bcewithlogits')
    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--sigma', type=float, default=1e-3)
    parser.add_argument('--use_ratio', action='store_true')
    parser.add_argument('--use_re', action='store_true')
    parser.add_argument('--forward_block_one', action='store_true')
    parser.add_argument('--forward_block_two', action='store_true')
    parser.add_argument('--forward_block_three', action='store_true')
    parser.add_argument('--forward_feat_map', action='store_true')
    parser.add_argument('--base_dir', default='output/ood_scores', type=str, help='result directory')
    parser.add_argument('--cfg', default='models/vit-base-p16-384.py')
    parser.add_argument('--checkpoint',
                        default='checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth',
                        help='Path to checkpoint for vit model')
    parser.set_defaults(argument=True)
    args = parser.parse_args()
    return args