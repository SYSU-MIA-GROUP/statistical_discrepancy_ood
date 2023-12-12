#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'
python feature_compute.py --gpu '0' --model_arch 'resnet50' --name 'resnet50' --method 'element_mean' --base_dir 'features'