import os
import sys
import io
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset_load import *
from model_zoo import model_create
from set_random_seed import set_seed
from color_space_transform import Color_space_transform
from lpyr_dec import *
from tqdm import tqdm
from torchsummary import summary

set_seed(66)

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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

def train_model(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 分布式采样
    trainloader = args['trainloader']
    testloader = args['testloader']

    train_sampler = DistributedSampler(trainloader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(testloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)

    trainloader_ddp = DataLoader(trainloader.dataset, batch_size=trainloader.batch_size, sampler=train_sampler,
                                num_workers=trainloader.num_workers, pin_memory=True)
    testloader_ddp = DataLoader(testloader.dataset, batch_size=testloader.batch_size, sampler=test_sampler,
                               num_workers=testloader.num_workers, pin_memory=True)

    model = args['model']
    model.to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = args['optimizer']
    scheduler = args['scheduler']
    criterion = args['criterion']
    color_trans = args['color_trans']
    save_path = args['save_path']
    log_file_path = args['log_file_path']
    resolution = args['resolution']
    max_epochs = args['max_epochs']
    model_name = args['model_name']
    dataset_name = args['dataset_name']
    color_space_name = args['color_space_name']
    peak_luminance = args['peak_luminance']

    if rank == 0:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'w') as log_file:
            buf = io.StringIO()
            tee = Tee(sys.stdout, buf)
            old_stdout = sys.stdout
            sys.stdout = tee
            try:
                summary(model.module if isinstance(model, DDP) else model, input_size=(3, resolution[0], resolution[1]))
            finally:
                sys.stdout = old_stdout
            log_file.write(buf.getvalue())
            log_file.write('\n')
            log_file.write(
                f"# Model: {model_name}, Dataset: {dataset_name}, Color: {color_space_name}, Peak L: {peak_luminance}\n")

    best_acc = 0.0
    for epoch in range(1, max_epochs + 1):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, trainloader_ddp, optimizer, criterion, device, epoch, color_trans)
        acc = test_one_epoch(model, testloader_ddp, device, epoch, color_trans)

        if rank == 0:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"[Epoch {epoch}] Test Accuracy: {acc:.2f}%\n")
                log_file.flush()
            scheduler.step()

            # 保存最好的模型只由rank0负责
            if acc > best_acc:
                best_acc = acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.module.state_dict(), save_path)
                print(f"✅ Saved best model with accuracy {best_acc:.2f}%")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"Saved best model with accuracy {best_acc:.2f}%\n")

    cleanup()

if __name__ == '__main__':
    train_dataset_name_list = ['Tiny-ImageNet'] #'CIFAR-100']#,
    model_name_list = ['resnet18', 'resnet18-lpyr', 'resnet18-lpyr-2', 'resnet18-clpyr', 'resnet18-clpyr-CSF', 'resnet18-clpyr-CM-transducer']
    color_space_name_list = ['sRGB', 'RGB_linear', 'XYZ_linear', 'DKL_linear']
    peak_luminance_list = [100, 200, 500]
    diagonal_size_inches_list = [10, 20, 50] #5
    resolution = [64, 64]
    viewing_distance_meters = 1
    # batch_size = 128 * 8

    gpu_ids = [0, 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
    world_size = len(gpu_ids)

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

                        if dataset_name == 'Tiny-ImageNet':
                            trainloader = dataset_load(dataset_name=dataset_name, type='train')
                            testloader = dataset_load(dataset_name=dataset_name, type='test')
                        else:
                            trainloader = dataset_load(dataset_name=dataset_name, type='train')
                            testloader = dataset_load(dataset_name=dataset_name, type='test')

                        color_trans = Color_space_transform(color_space_name=color_space_name,
                                                            peak_luminance=peak_luminance)
                        model = model_create(model_name=model_name, dataset_name=dataset_name)

                        model.to(torch.device('cpu'))  # 先放cpu避免多进程CUDA冲突

                        if model_name.endswith('-clpyr') or model_name.endswith('-clpyr-CSF') or model_name.endswith('-clpyr-CM-transducer'):
                            lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, torch.device('cpu'), contrast='weber_g1')
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)
                        if model_name.endswith('-lpyr') or model_name.endswith('-lpyr-2'):
                            lpyr = laplacian_pyramid_simple(resolution[1], resolution[0], display_ppd, torch.device('cpu'))
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)

                        # 优化器、scheduler 在spawn的train里创建，为避免cuda问题，这里先不创建

                        save_path = (f'../HVS_for_better_NN_pth_2/'
                                     f'best_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}_'
                                     f'diag{diagonal_size_inches}.pth')
                        log_path = (f'../HVS_for_better_NN_logs/'
                                    f'log_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}_'
                                    f'diag{diagonal_size_inches}.txt')

                        if os.path.exists(log_path):
                            print(f"🚫 Skipping: already trained — {log_path}")
                            continue

                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)

                        # 需要创建优化器和scheduler给train函数
                        # 先简单用SGD，后面你可以根据需要修改
                        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

                        args = {
                            'trainloader': trainloader,
                            'testloader': testloader,
                            'model': model,
                            'optimizer': optimizer,
                            'scheduler': scheduler,
                            'criterion': criterion,
                            'color_trans': color_trans,
                            'save_path': save_path,
                            'log_file_path': log_path,
                            'resolution': resolution,
                            'max_epochs': 100,
                            'model_name': model_name,
                            'dataset_name': dataset_name,
                            'color_space_name': color_space_name,
                            'peak_luminance': peak_luminance,
                        }

                        try:
                            mp.spawn(train_model,
                                     args=(world_size, args),
                                     nprocs=world_size,
                                     join=True)
                        except Exception as e:
                            print(f"Error occurred: {e}")
