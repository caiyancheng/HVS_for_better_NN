import os
import numpy as np
import torch.optim as optim
import torch.nn as nn
from dataset_load import *
import itertools
from model_zoo import model_create
from set_random_seed import set_seed
set_seed(66)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from color_space_transform import Color_space_transform
from tqdm import tqdm
import math
from lpyr_dec import *
from torchsummary import summary
import io
import sys

criterion = nn.CrossEntropyLoss()
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()

def compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters):
    ar = resolution[0] / resolution[1]
    height_mm = math.sqrt((diagonal_size_inches * 25.4) ** 2 / (1 + ar ** 2))
    display_size_m = (ar * height_mm / 1000, height_mm / 1000)
    pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
    display_ppd = 1 / pix_deg
    return display_ppd

def train_one_epoch(model, trainloader, optimizer, criterion, device, epoch, color_trans):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if color_trans is not None:
            inputs = color_trans(inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch}] Training Loss: {running_loss / len(trainloader):.3f}")

def test_one_epoch(model, testloader, device, epoch, color_trans):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if color_trans is not None:
                inputs = color_trans(inputs)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"[Epoch {epoch}] Test Accuracy: {acc:.2f}%")
    return acc

def train_model(model, trainloader, testloader, optimizer, scheduler, criterion, device, save_path, color_trans, log_file_path, max_epochs=100):
    best_acc = 0.0
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'w') as log_file:
        buf = io.StringIO()
        tee = Tee(sys.stdout, buf)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            summary(model, input_size=(3, 32, 32))
        finally:
            sys.stdout = old_stdout
        log_file.write(buf.getvalue())
        log_file.write('\n')
        log_file.write(
            f"# Model: {model_name}, Dataset: {dataset_name}, Color: {color_space_name}, Peak L: {peak_luminance}\n")
        for epoch in tqdm(range(1, max_epochs + 1)):
            train_one_epoch(model, trainloader, optimizer, criterion, device, epoch, color_trans)
            acc = test_one_epoch(model, testloader, device, epoch, color_trans)
            log_file.write(f"[Epoch {epoch}] Test Accuracy: {acc:.2f}%\n")
            log_file.flush()
            scheduler.step()

            # 保存最好的模型
            if acc > best_acc:
                best_acc = acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"✅ Saved best model with accuracy {best_acc:.2f}%")
                log_file.write(f"Saved best model with accuracy {best_acc:.2f}%\n")

if __name__ == '__main__':
    train_dataset_name_list = ['CIFAR-100']
    # model_name_list = ['resnet18', 'resnet18-lpyr', 'resnet18-clpyr', 'resnet18-clpyr-CSF', 'resnet18-clpyr-CM-transducer']
    model_name_list = ['resnet18', 'resnet18-lpyr', 'resnet18-clpyr', 'resnet18-clpyr-CSF', 'resnet18-clpyr-CM-transducer']
    color_space_name_list = ['sRGB', 'RGB_linear', 'XYZ_linear', 'DKL_linear']
    peak_luminance_list = [100, 200, 500]
    diagonal_size_inches_list = [5, 10, 20, 50]
    resolution = [32, 32]
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
                        display_ppd = compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters)
                        set_seed(66)
                        print(f"Dataset: {dataset_name}, Model: {model_name}, Color Space: {color_space_name}, "
                              f"Peak Luminance: {peak_luminance}, Diagonal: {diagonal_size_inches} inches")

                        trainloader, testloader = dataset_load(dataset_name=dataset_name)
                        color_trans = Color_space_transform(color_space_name=color_space_name,
                                                            peak_luminance=peak_luminance)
                        model = model_create(model_name=model_name, dataset_name=dataset_name)
                        model.to(device)

                        if model_name.endswith('-clpyr') or model_name.endswith('-clpyr-CSF') or model_name.endswith('-clpyr-CM-transducer'):
                            lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, device, contrast='weber_g1')
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)
                        if model_name.endswith('-lpyr'):
                            lpyr = laplacian_pyramid_simple(resolution[1], resolution[0], display_ppd, device)
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)

                        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

                        save_path = (f'../HVS_for_better_NN_pth_2/'
                                     f'best_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}_'
                                     f'diag{diagonal_size_inches}.pth')
                        log_path = (f'../HVS_for_better_NN_logs/'
                                    f'log_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}_'
                                    f'diag{diagonal_size_inches}.txt')
                        # 如果日志文件存在，说明训练已经完成，跳过
                        if os.path.exists(log_path):
                            print(f"🚫 Skipping: already trained — {log_path}")
                            continue

                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)

                        try:
                            train_model(
                                model=model,
                                trainloader=trainloader,
                                testloader=testloader,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                criterion=criterion,
                                device=device,
                                save_path=save_path,
                                color_trans=color_trans,
                                log_file_path=log_path,
                                max_epochs=100
                            )
                        except Exception as e:
                            print(f"Error occurred: {e}")



