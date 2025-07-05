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

criterion = nn.CrossEntropyLoss()

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
    model_name_list = ['resnet18']
    color_space_name_list = ['sRGB', 'RGB_linear', 'XYZ_linear', 'DKL_linear']
    peak_luminance_list = [100, 200, 500]

    for dataset_name, model_name, color_space_name, peak_luminance in itertools.product(train_dataset_name_list, model_name_list,
                                                                        color_space_name_list, peak_luminance_list):
        print(f"Dataset: {dataset_name}, Model: {model_name}, Color Space: {color_space_name}, Peak Luminance: {peak_luminance}")
        trainloader, testloader = dataset_load(dataset_name=dataset_name)
        color_trans = Color_space_transform(color_space_name=color_space_name, peak_luminance=peak_luminance)
        model = model_create(model_name=model_name, dataset_name=dataset_name)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        save_path = f'../HVS_for_better_NN_pth_2/best_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}.pth'
        log_path = f'../HVS_for_better_NN_logs/log_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}.txt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
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




