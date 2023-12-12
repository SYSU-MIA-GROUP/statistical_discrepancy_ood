import os
import torch
from mmcls.apis import init_model
import mmcv
import torchvision


def get_model(args, num_classes, load_ckpt=True):
    if args.in_dataset == 'imagenet':
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'ViT':
            cfg = mmcv.Config.fromfile(args.cfg)
            model = init_model(cfg, args.checkpoint, 0)
        elif args.model_arch == 'vit-l-16':
            model = torchvision.models.vit_l_16(weights="ViT_L_16_Weights.DEFAULT")

    elif args.in_dataset == 'ImageNet100':
        if args.model_arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=False)
            checkpoint = torch.load("/data/xiaoyuan/react/checkpoints/resnet50_imagenet100_standard.pt")
            model.load_state_dict(checkpoint)
        elif args.model_arch == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=num_classes, pretrained=False)
            checkpoint = torch.load("/data/xiaoyuan/react/checkpoints/mobilenet_imagenet100_standard.pt")
            model.load_state_dict(checkpoint)
    else:
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50_cifar
            model = resnet50_cifar(num_classes=num_classes, method=args.method)
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)

        if load_ckpt:
            checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name, epochs=args.epochs))
            model.load_state_dict(checkpoint['state_dict'])

    model.cuda()
    model.eval()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    return model


if __name__ == '__main__':
    print('小丑召唤')
