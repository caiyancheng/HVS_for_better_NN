#!/bin/bash

# 激活conda环境（假设环境名为myenv）
# 注意：激活conda环境要用source并指定conda.sh路径
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hvs
cd /auto/homes/yc613/Py_codes/HVS_for_better_NN/

python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 5
python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 10
python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 20
python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 50

python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 5
python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 10
python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 20
python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 50

python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 5
python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 10
python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 20
python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 50