import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_load import dataset_load
from model_zoo import model_create
from set_random_seed import set_seed
from color_space_transform import Color_space_transform
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import numpy as np
import itertools

set_seed(66)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss()

def prepare_data_for_attack(testloader, max_batches=10):
    """从testloader中取出部分数据用于攻击，返回numpy格式和tensor格式"""
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
    """执行攻击，转化颜色空间，评估攻击后准确率"""
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
    test_dataset_name_list = ['CIFAR-100']
    model_name_list = ['resnet18']
    color_space_name_list = ['sRGB', 'RGB_linear', 'XYZ_linear', 'DKL_linear']
    peak_luminance_list = [100, 200, 500]
    eps_list = [0.01, 0.02, 0.05]  # 多个eps值

    for dataset_name, model_name, color_space_name, peak_luminance in itertools.product(
        test_dataset_name_list, model_name_list, color_space_name_list, peak_luminance_list):

        print(f"🔍 Attack Test: Dataset={dataset_name}, Model={model_name}, Color={color_space_name}, Peak L={peak_luminance}")

        testloader = dataset_load(dataset_name=dataset_name, type='test', corruption_type=None)

        color_trans = Color_space_transform(color_space_name=color_space_name, peak_luminance=peak_luminance)

        model = model_create(model_name=model_name, dataset_name=dataset_name)
        model.to(device)

        model_path = f'../HVS_for_better_NN_pth_2/best_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"❌ Model checkpoint not found: {model_path}")
            continue

        x_np, y_np, x_tensor = prepare_data_for_attack(testloader, max_batches=10)

        classifier = PyTorchClassifier(
            model=model,
            clip_values=(0.0, 1.0),
            loss=criterion,
            optimizer=None,
            input_shape=(3, 32, 32),
            nb_classes=100,
            device_type='gpu' if torch.cuda.is_available() else 'cpu'
        )

        # 先计算 clean accuracy
        model.eval()
        with torch.no_grad():
            if color_trans:
                x_orig_cs = color_trans(x_tensor).to(device)
            else:
                x_orig_cs = x_tensor.to(device)
            logits_clean = model(x_orig_cs)
        pred_clean = logits_clean.argmax(dim=1).cpu().numpy()
        acc_clean = np.mean(pred_clean == y_np)
        print(f"🎯 Clean Accuracy (subset): {acc_clean * 100:.2f}%")

        # 对不同eps计算攻击准确率
        for eps in eps_list:
            acc_adv = test_attack(model, classifier, x_tensor, y_np, color_trans, device, eps=eps)
            print(f"⚠️ PGD Adversarial Accuracy (subset): {acc_adv * 100:.2f}%, Eps={eps}\n")
