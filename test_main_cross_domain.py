import os
import torch
import torch.nn as nn
from dataset_load import dataset_load
from model_zoo import model_create
from set_random_seed import set_seed
from color_space_transform import Color_space_transform
from tqdm import tqdm
import itertools

set_seed(66)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss()

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
    corruption_type_list = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    severity_list = [1, 2, 3, 4, 5]  # 多个严重等级

    model_name_list = ['resnet18']
    color_space_name_list = ['sRGB', 'RGB_linear', 'XYZ_linear', 'DKL_linear']
    peak_luminance_list = [100, 200, 500]

    for model_name, color_space_name, peak_luminance, corruption_type, severity in itertools.product(
        model_name_list, color_space_name_list, peak_luminance_list, corruption_type_list, severity_list):

        print(f"🔍 Testing: Dataset={dataset_name}, Corruption={corruption_type}, Severity={severity}, "
              f"Model={model_name}, Color={color_space_name}, Peak L={peak_luminance}")

        testloader = dataset_load(dataset_name=dataset_name, type='test',
                                  corruption_type=corruption_type, severity=severity)

        color_trans = Color_space_transform(color_space_name=color_space_name, peak_luminance=peak_luminance)

        model = model_create(model_name=model_name, dataset_name='CIFAR-100')
        model.to(device)

        model_path = f'../HVS_for_better_NN_pth_2/best_{model_name}_CIFAR-100_{color_space_name}_pl{peak_luminance}.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"❌ Model checkpoint not found: {model_path}")
            continue

        acc = test_model(model, testloader, device, color_trans)
        print(f"🎯 Test Accuracy on {corruption_type} (severity {severity}): {acc:.2f}%\n")
