import os
import torch
import torch.nn as nn
from dataset_load import dataset_load
from model_zoo import model_create
from set_random_seed import set_seed
from color_space_transform import Color_space_transform
from lpyr_dec import laplacian_pyramid_simple, laplacian_pyramid_simple_contrast
import math

set_seed(66)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
    print(f"✅ Test Accuracy: {acc:.2f}%")
    return acc

if __name__ == '__main__':
    dataset_name = 'CIFAR-100'
    model_name_list = ['resnet18', 'resnet18-lpyr', 'resnet18-clpyr', 'resnet18-clpyr-CSF', 'resnet18-clpyr-CM-transducer']
    color_space_name = 'DKL_linear'
    peak_luminance_list = [100, 200, 500]
    diagonal_size_inches_list = [5, 10, 20, 50]
    resolution = [32, 32]
    viewing_distance_meters = 1

    for model_name in model_name_list:
        for diagonal_size_inches in diagonal_size_inches_list:
            for peak_luminance in peak_luminance_list:
                display_ppd = compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters)
                set_seed(66)
                print(f"🧪 Testing: {model_name}, L={peak_luminance}, diag={diagonal_size_inches} inches")

                _, testloader = dataset_load(dataset_name=dataset_name)
                color_trans = Color_space_transform(color_space_name=color_space_name,
                                                    peak_luminance=peak_luminance)
                model = model_create(model_name=model_name, dataset_name=dataset_name)
                model.to(device)

                if model_name.endswith('-clpyr') or model_name.endswith('-clpyr-CSF') or model_name.endswith('-clpyr-CM-transducer'):
                    lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, device, contrast='weber_g1')
                    model.set_lpyr(lpyr=lpyr, pyr_levels=4)
                elif model_name.endswith('-lpyr'):
                    lpyr = laplacian_pyramid_simple(resolution[1], resolution[0], display_ppd, device)
                    model.set_lpyr(lpyr=lpyr, pyr_levels=4)

                # 加载训练好的模型参数
                weight_path = (f'../HVS_for_better_NN_pth_2/'
                               f'best_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}_'
                               f'diag{diagonal_size_inches}.pth')
                if os.path.exists(weight_path):
                    model.load_state_dict(torch.load(weight_path, map_location=device))
                    test_model(model, testloader, device, color_trans)
                else:
                    print(f"❌ Model weights not found: {weight_path}")
