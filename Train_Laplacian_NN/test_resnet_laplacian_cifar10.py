import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import os
from model_zoo import CustomResNet18  # 确保 model_zoo.py 和此文件在同一目录，或正确导入路径
from torchsummary import summary

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

root = r'../'
data_root = os.path.join(root, r'Datasets/CIFAR10/data')

# 定义测试集数据预处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# 加载测试集
testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

def test_best_model():
    # 创建模型并加载权重
    model = CustomResNet18(num_classes=10).to(device)
    checkpoint_path = os.path.join(root, r'HVS_for_better_NN_pth/best_resnet18_laplacian_cifar10.pth')

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint {checkpoint_path} not found.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    summary(model, input_size=(3, 32, 32))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc="Testing best model"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f"🎯 Accuracy of the best model on the CIFAR-10 test set: {acc:.2f}%")

if __name__ == '__main__':
    test_best_model()
