import os
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import itertools
import pandas as pd  # ✅ 新增
from dataset_load import dataset_load
from model_zoo import model_create
from set_random_seed import set_seed
from color_space_transform import Color_space_transform
from lpyr_dec import *
from datetime import datetime

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
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            if color_trans is not None:
                inputs = color_trans(inputs)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

if __name__ == '__main__':
    save_csv = False
    dataset_name = 'CIFAR-100-C'
    base_model_dataset = 'CIFAR-100'
    resolution = [32, 32]
    viewing_distance_meters = 1.0
    diagonal_size_inches_list = [5, 10]#, 20, 50]

    corruption_type_list = ['gaussian_noise',  'fog', 'jpeg_compression']
    # corruption_type_list = [
    #     'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    #     'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
    #     'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    # ]
    severity_list = [5]

    model_name_list = ['resnet18', 'resnet18-lpyr', 'resnet18-clpyr', 'resnet18-clpyr-CSF', 'resnet18-clpyr-CM-transducer']
    color_space_name_list = ['sRGB', 'RGB_linear', 'XYZ_linear', 'DKL_linear']
    peak_luminance_list = [100]#, 200, 500]

    log_dir = '../HVS_for_better_NN_logs_cross_domain_test/'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f'cross_domain_test_log_{timestamp}.txt')
    log_file = open(log_file_path, 'w')
    if save_csv:
        # ✅ 存储所有结果的字典
        results_dict = {}

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

            for peak_luminance, diagonal_size_inches in itertools.product(luminance_iter, diagonal_iter):
                display_ppd = compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters)
                model = model_create(model_name=model_name, dataset_name=base_model_dataset)
                model.to(device)

                if model_name.endswith('-clpyr') or model_name.endswith('-clpyr-CSF') or model_name.endswith('-clpyr-CM-transducer'):
                    lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, device, contrast='weber_g1')
                    model.set_lpyr(lpyr=lpyr, pyr_levels=4)
                elif model_name.endswith('-lpyr'):
                    lpyr = laplacian_pyramid_simple(resolution[1], resolution[0], display_ppd, device)
                    model.set_lpyr(lpyr=lpyr, pyr_levels=4)

                model_path = (f'../HVS_for_better_NN_pth_2/'
                              f'best_{model_name}_{base_model_dataset}_{color_space_name}_pl{peak_luminance}_'
                              f'diag{diagonal_size_inches}.pth')

                if not os.path.exists(model_path):
                    print(f"❌ Model file not found: {model_path}")
                    log_file.write(f"[Skip] {model_path} not found\n")
                    continue

                model.load_state_dict(torch.load(model_path))

                if save_csv:
                    row_key = f'{model_name}_{color_space_name}_pl{peak_luminance}'
                    results_dict[row_key] = {}

                for corruption_type, severity in itertools.product(corruption_type_list, severity_list):
                    print(f"\n🔍 Testing: Model={model_name}, Dataset={dataset_name}, Corruption={corruption_type}, "
                          f"Severity={severity}, Color={color_space_name}, Peak L={peak_luminance}, Diagonal={diagonal_size_inches}\"")

                    testloader = dataset_load(dataset_name=dataset_name, type='test',
                                              corruption_type=corruption_type, severity=severity)

                    color_trans = Color_space_transform(color_space_name=color_space_name,
                                                        peak_luminance=peak_luminance)

                    acc = test_model(model, testloader, device, color_trans)

                    log_line = (f"[Result] Model={model_name}, Corruption={corruption_type}, Severity={severity}, "
                                f"Color={color_space_name}, PeakL={peak_luminance}, Diag={diagonal_size_inches}, "
                                f"Accuracy={acc:.2f}%\n")

                    print(log_line)
                    log_file.write(log_line)
                    log_file.flush()

                    # ✅ 存入结果表
                    if save_csv:
                        results_dict[row_key][corruption_type] = acc
                        csv_output_path = os.path.join(log_dir, f'corruption_results_{timestamp}_middle.csv')
                        df = pd.DataFrame.from_dict(results_dict, orient='index')
                        df = df[corruption_type_list]  # 确保列顺序一致
                        df.to_csv(csv_output_path)
                        print(f"\n✅ Results saved to {csv_output_path}")

    log_file.close()

    if save_csv:
        # ✅ 将结果写入CSV表格
        csv_output_path = os.path.join(log_dir, f'corruption_results_{timestamp}_final.csv')
        df = pd.DataFrame.from_dict(results_dict, orient='index')
        df = df[corruption_type_list]  # 确保列顺序一致
        df.to_csv(csv_output_path)
        print(f"\n✅ Results saved to {csv_output_path}")
