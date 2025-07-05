#!/bin/bash

# 激活conda环境（假设环境名为myenv）
# 注意：激活conda环境要用source并指定conda.sh路径
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hvs
cd /auto/homes/yc613/Py_codes/HVS_for_better_NN/

python Train_Laplacian_NN/test_attack_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 5 --eps_value 0.1
python Train_Laplacian_NN/test_attack_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 5 --eps_value 0.02
python Train_Laplacian_NN/test_attack_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 5 --eps_value 0.1
python Train_Laplacian_NN/test_attack_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 5 --eps_value 0.02


#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 5 # 75.08%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 50 # 75.08%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 5 # 74.45%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 50 # 74.45%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 5 --peak_luminance 100 --diagonal_size_inches 5 # 74.05%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 5 --peak_luminance 100 --diagonal_size_inches 50 # 74.05%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 5 # 74.58%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 50 # 74.58%

#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 5 #74.22%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 10 #73.77%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 20 #74.63%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 30 #74.83%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 40 #74.65%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 50 #74.69%
#
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 5 #74.72%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 10 #74.39%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 20 #73.82%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 30 #NAN
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 40 #NAN
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 50 #NAN
#
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 5 #73.90%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 10 #75.02%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 20 #74.64%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 30 #74.82%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 40 #74.48%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 50 #75.09%

###这种强行驱除的方式太不可控，不太行
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 5 #70.49%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 10 #57.38%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 20 #73.23%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 30 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 40 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 50 #
#
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 5 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 10 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 20 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 30 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 40 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 50 #
#
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 5 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 10 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 20 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 30 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 40 #
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_2.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 50 #

#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 5 #74.38%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 10 #74.05%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 20 #74.76%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 500 --diagonal_size_inches 50 #74.02%
#
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 5 #74.56%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 10 #74.35%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 20 #74.65%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 5 --peak_luminance 500 --diagonal_size_inches 50 #NAN
#
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 5 #74.74%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 10 #74.17%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 20 #74.12%
#python Train_Laplacian_NN/train_resnet_cifar100_no_first_downsample_dkl_contrast_lpyr_CSF_masking_transducer.py --pyr_levels 4 --peak_luminance 100 --diagonal_size_inches 50 #73.61%