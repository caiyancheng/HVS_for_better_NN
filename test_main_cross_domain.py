import os
import torch
import torch.nn as nn
from dataset_load import dataset_load
from model_zoo import model_create
from set_random_seed import set_seed
from color_space_transform import Color_space_transform
from tqdm import tqdm
import itertools
from lpyr_dec import *
import math

set_seed(66)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss()

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
        for inputs, targets in tqdm(testloader, desc='Testing'):
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
    dataset_name = 'CIFAR-100-C'
    resolution = [32, 32]
    viewing_distance_meters = 1.0
    diagonal_size_inches_list = [5, 10, 20, 50]

    corruption_type_list = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    severity_list = [1, 2, 3, 4, 5]

    model_name_list = ['resnet18-lpyr', 'resnet18-clpyr', 'resnet18-clpyr-CSF', 'resnet18-clpyr-CM-transducer']
    color_space_name_list = ['DKL_linear']
    peak_luminance_list = [100, 200, 500]

    for model_name, color_space_name, peak_luminance, corruption_type, severity, diagonal_size_inches in itertools.product(
        model_name_list, color_space_name_list, peak_luminance_list, corruption_type_list, severity_list, diagonal_size_inches_list):

        print(f"\n🔍 Testing: Model={model_name}, Dataset={dataset_name}, Corruption={corruption_type}, "
              f"Severity={severity}, Color={color_space_name}, Peak L={peak_luminance}, Diagonal={diagonal_size_inches}\"")

        testloader = dataset_load(dataset_name=dataset_name, type='test',
                                  corruption_type=corruption_type, severity=severity)

        color_trans = Color_space_transform(color_space_name=color_space_name, peak_luminance=peak_luminance)

        model = model_create(model_name=model_name, dataset_name='CIFAR-100')
        model.to(device)

        display_ppd = compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters)

        if model_name.endswith('-clpyr') or model_name.endswith('-clpyr-CSF') or model_name.endswith('-clpyr-CM-transducer'):
            lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, device, contrast='weber_g1')
            model.set_lpyr(lpyr=lpyr, pyr_levels=4)
        if model_name.endswith('-lpyr'):
            lpyr = laplacian_pyramid_simple(resolution[1], resolution[0], display_ppd, device)
            model.set_lpyr(lpyr=lpyr, pyr_levels=4)

        model_path = f'../HVS_for_better_NN_pth_2/best_{model_name}_CIFAR-100_{color_space_name}_pl{peak_luminance}_diag{diagonal_size_inches}.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"❌ Model checkpoint not found: {model_path}")
            continue

        acc = test_model(model, testloader, device, color_trans)
        print(f"🎯 Test Accuracy on {corruption_type} (severity {severity}): {acc:.2f}%")
