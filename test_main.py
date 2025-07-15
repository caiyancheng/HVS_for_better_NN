import os
import torch
import torch.nn as nn
from dataset_load import dataset_load
from model_zoo import model_create
from color_space_transform import Color_space_transform
from lpyr_dec import laplacian_pyramid_simple, laplacian_pyramid_simple_contrast
from set_random_seed import set_seed
import math

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
set_seed(66)


def compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters):
    ar = resolution[0] / resolution[1]
    height_mm = math.sqrt((diagonal_size_inches * 25.4) ** 2 / (1 + ar ** 2))
    display_size_m = (ar * height_mm / 1000, height_mm / 1000)
    pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
    display_ppd = 1 / pix_deg
    return display_ppd


def test_model(model, testloader, device, color_trans):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            if color_trans is not None:
                inputs = color_trans(inputs)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


if __name__ == '__main__':
    train_dataset_name_list = ['Tiny-ImageNet'] #['CIFAR-100']
    model_name_list = ['resnet18', 'resnet18-lpyr', 'resnet18-clpyr', 'resnet18-clpyr-CSF', 'resnet18-clpyr-CM-transducer']
    color_space_name_list = ['sRGB', 'RGB_linear', 'XYZ_linear', 'DKL_linear']
    # peak_luminance_list = [100]#, 200, 500]
    # diagonal_size_inches_list = [5, 10]#, 20, 50]
    # resolution = [32, 32]
    peak_luminance_list = [100, 200, 500]
    diagonal_size_inches_list = [10, 20, 50]
    resolution = [64, 64]
    viewing_distance_meters = 1

    for dataset_name in train_dataset_name_list:
        for model_name in model_name_list:
            if model_name == 'resnet18':
                diagonal_iter = [diagonal_size_inches_list[0]]
                color_space_iter = color_space_name_list
            else:
                diagonal_iter = diagonal_size_inches_list
                color_space_iter = ['DKL_linear']

            for color_space_name in color_space_iter:
                if color_space_name == 'sRGB':
                    luminance_iter = [peak_luminance_list[0]]
                else:
                    luminance_iter = peak_luminance_list

                for diagonal_size_inches in diagonal_iter:
                    for peak_luminance in luminance_iter:
                        print(f"Testing: Model={model_name}, Dataset={dataset_name}, Color={color_space_name}, "
                              f"PeakL={peak_luminance}, Diagonal={diagonal_size_inches}")

                        display_ppd = compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters)
                        set_seed(66)

                        # 数据集和颜色变换
                        _, testloader = dataset_load(dataset_name=dataset_name)
                        color_trans = Color_space_transform(color_space_name=color_space_name,
                                                            peak_luminance=peak_luminance)

                        # 创建模型并加载参数
                        model = model_create(model_name=model_name, dataset_name=dataset_name)
                        model.to(device)

                        if model_name.endswith('-clpyr') or model_name.endswith('-clpyr-CSF') or model_name.endswith(
                                '-clpyr-CM-transducer'):
                            lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, device,
                                                                     contrast='weber_g1')
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)
                        if model_name.endswith('-lpyr'):
                            lpyr = laplacian_pyramid_simple(resolution[1], resolution[0], display_ppd, device)
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)

                        # 权重路径
                        model_path = (f'../HVS_for_better_NN_pth_2/'
                                      f'best_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}_'
                                      f'diag{diagonal_size_inches}.pth')
                        if os.path.exists(model_path):
                            model.load_state_dict(torch.load(model_path, map_location=device))
                            acc = test_model(model, testloader, device, color_trans)
                            print(f"✅ Accuracy: {acc:.2f}%")
                        else:
                            print(f"❌ Model weights not found at {model_path}")
