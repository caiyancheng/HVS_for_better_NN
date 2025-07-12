import os
import math
import io
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsummary import summary
from tqdm import tqdm

from dataset_load import dataset_load
from model_zoo import model_create
from set_random_seed import set_seed
from color_space_transform import Color_space_transform
from lpyr_dec import laplacian_pyramid_simple, laplacian_pyramid_simple_contrast

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
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        if color_trans:
            inputs = color_trans(inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(trainloader)

def test_one_epoch(model, testloader, device, epoch, color_trans):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            if color_trans:
                inputs = color_trans(inputs)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, gpus_to_use):
    set_seed(66)
    setup(rank, world_size)

    local_gpu = gpus_to_use[rank]
    torch.cuda.set_device(local_gpu)
    device = torch.device(f"cuda:{local_gpu}")

    criterion = nn.CrossEntropyLoss()

    train_dataset_name_list = ['Tiny-ImageNet']
    model_name_list = ['resnet18', 'resnet18-lpyr', 'resnet18-lpyr-2', 'resnet18-clpyr', 'resnet18-clpyr-CSF', 'resnet18-clpyr-CM-transducer']
    color_space_name_list = ['sRGB', 'RGB_linear', 'XYZ_linear', 'DKL_linear']
    peak_luminance_list = [100, 200, 500]
    diagonal_size_inches_list = [10, 20, 50]
    resolution = [64, 64]
    viewing_distance_meters = 1

    for dataset_name in train_dataset_name_list:
        for model_name in model_name_list:
            diagonal_iter = [diagonal_size_inches_list[0]] if model_name == 'resnet18' else diagonal_size_inches_list
            color_space_iter = color_space_name_list if model_name == 'resnet18' else ['DKL_linear']

            for color_space_name in color_space_iter:
                luminance_iter = [peak_luminance_list[0]] if color_space_name == 'sRGB' else peak_luminance_list

                for diagonal_size_inches in diagonal_iter:
                    for peak_luminance in luminance_iter:
                        display_ppd = compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters)

                        if rank == 0:
                            print(f"Dataset: {dataset_name}, Model: {model_name}, Color Space: {color_space_name}, "
                                  f"Peak Luminance: {peak_luminance}, Diagonal: {diagonal_size_inches} inches")

                        trainset = dataset_load(dataset_name=dataset_name, type='train')
                        testset = dataset_load(dataset_name=dataset_name, type='test')
                        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
                        test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False)

                        trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=128, num_workers=4, pin_memory=True)
                        testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=128, num_workers=4, pin_memory=True)

                        color_trans = Color_space_transform(color_space_name=color_space_name, peak_luminance=peak_luminance)
                        model = model_create(model_name=model_name, dataset_name=dataset_name).to(device)

                        if 'clpyr' in model_name:
                            lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, device, contrast='weber_g1')
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)
                        elif 'lpyr' in model_name:
                            lpyr = laplacian_pyramid_simple(resolution[1], resolution[0], display_ppd, device)
                            model.set_lpyr(lpyr=lpyr, pyr_levels=4)

                        model = DDP(model, device_ids=[local_gpu])
                        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

                        save_path = f'../HVS_for_better_NN_pth_2/best_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}_diag{diagonal_size_inches}.pth'
                        log_path = f'../HVS_for_better_NN_logs/log_{model_name}_{dataset_name}_{color_space_name}_pl{peak_luminance}_diag{diagonal_size_inches}.txt'

                        if rank == 0 and os.path.exists(log_path):
                            print(f"🚫 Skipping: already trained — {log_path}")
                            continue

                        best_acc = 0.0
                        if rank == 0:
                            os.makedirs(os.path.dirname(log_path), exist_ok=True)
                            log_file = open(log_path, 'w')
                            buf = io.StringIO()
                            tee = Tee(sys.stdout, buf)
                            old_stdout = sys.stdout
                            sys.stdout = tee
                            try:
                                summary(model.module, input_size=(3, resolution[0], resolution[1]))
                            finally:
                                sys.stdout = old_stdout
                            log_file.write(buf.getvalue() + '\n')
                            log_file.write(f"# Model: {model_name}, Dataset: {dataset_name}, Color: {color_space_name}, Peak L: {peak_luminance}\n")

                        for epoch in range(1, 101):
                            train_sampler.set_epoch(epoch)
                            train_loss = train_one_epoch(model, trainloader, optimizer, criterion, device, epoch, color_trans)
                            acc = test_one_epoch(model, testloader, device, epoch, color_trans)

                            if rank == 0:
                                log_file.write(f"[Epoch {epoch}] Train Loss: {train_loss:.3f}, Test Acc: {acc:.2f}%\n")
                                log_file.flush()
                                print(f"[Epoch {epoch}] Train Loss: {train_loss:.3f}, Test Acc: {acc:.2f}%")
                                if acc > best_acc:
                                    best_acc = acc
                                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                    torch.save(model.module.state_dict(), save_path)
                                    print(f"✅ Saved best model with accuracy {best_acc:.2f}%")
                                    log_file.write(f"Saved best model with accuracy {best_acc:.2f}%\n")
                            scheduler.step()

                        if rank == 0:
                            log_file.close()

    cleanup()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    gpus_to_use = [0, 1]  # 使用可见的 GPU 0 和 2 实际映射为 0, 1
    world_size = len(gpus_to_use)
    mp.spawn(train_ddp, args=(world_size, gpus_to_use), nprocs=world_size, join=True)
