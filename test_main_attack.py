import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset_load import dataset_load
from model_zoo import model_create
from set_random_seed import set_seed
from color_space_transform import Color_space_transform
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from lpyr_dec import laplacian_pyramid_simple, laplacian_pyramid_simple_contrast
import math
import itertools

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

def prepare_data_for_attack(testloader, max_batches=10):
    x_list, y_list = [], []
    for i, (inputs, targets) in enumerate(testloader):
        x_list.append(inputs)
        y_list.append(targets)
        if i + 1 >= max_batches:
            break
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return x.numpy(), y.numpy(), x

def test_attack(model, classifier, x_orig_tensor, y_orig, color_trans, device, eps=0.02):
    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        eps_step=eps * 0.1,
        max_iter=32,
        verbose=False
    )
    x_adv_np = attack.generate(x=x_orig_tensor.numpy())
    x_adv_tensor = torch.tensor(x_adv_np, dtype=torch.float32)

    if color_trans:
        x_adv_cs = color_trans(x_adv_tensor).to(device)
    else:
        x_adv_cs = x_adv_tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits_adv = model(x_adv_cs)

    pred_adv = logits_adv.argmax(dim=1).cpu().numpy()
    acc_adv = np.mean(pred_adv == y_orig)
    return acc_adv

if __name__ == '__main__':
    dataset_name_list = ['CIFAR-100']
    model_name_list = ['resnet18', 'resnet18-lpyr', 'resnet18-clpyr', 'resnet18-clpyr-CSF', 'resnet18-clpyr-CM-transducer']
    color_space_name_list = ['sRGB', 'RGB_linear', 'XYZ_linear', 'DKL_linear']
    peak_luminance_list = [100] #, 200, 500]
    diagonal_size_inches_list = [10] #[5, 10, 20, 50]
    resolution = [32, 32]
    viewing_distance_meters = 1
    eps_list = [0.02]#[0.01, 0.02, 0.05]

    for dataset_name in dataset_name_list:
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
                        display_ppd = compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters)
                        testloader = dataset_load(dataset_name=dataset_name, type='test')
                        color_trans = Color_space_transform(color_space_name=color_space_name, peak_luminance=peak_luminance)
                        model = model_create(model_name=model_name, dataset_name=dataset_name)
                        model.to(device)

                        # 设置金字塔结构
                        if model_name.endswith('-clpyr') or model_name.endswith('-clpyr-CSF') or model_name.endswith('-clpyr-CM-transducer'):
                            lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, device, contrast='weber_g1')
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)
                        if model_name.endswith('-lpyr'):
                            lpyr = laplacian_pyramid_simple(resolution[1], resolution[0], display_ppd, device)
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)

                        model_path = (f'../HVS_for_better_NN_pth_2/'
                                      f'best_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}_'
                                      f'diag{diagonal_size_inches}.pth')
                        if not os.path.exists(model_path):
                            print(f"❌ Model checkpoint not found: {model_path}")
                            continue
                        model.load_state_dict(torch.load(model_path, map_location=device))
                        print(f"✅ Loaded model from {model_path}")

                        x_np, y_np, x_tensor = prepare_data_for_attack(testloader, max_batches=10)
                        classifier = PyTorchClassifier(
                            model=model,
                            loss=criterion,
                            clip_values=(0.0, 1.0),
                            input_shape=(3, 32, 32),
                            nb_classes=100
                        )

                        for eps in eps_list:
                            acc = test_attack(model, classifier, x_tensor, y_np, color_trans, device, eps=eps)
                            print(f"⚠️ PGD Attack eps={eps:.3f} → Accuracy: {acc*100:.2f}%")
