import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18
import os

# 路径配置
data_root = r'E:\Datasets\CIFAR10\data'
model_path = './best_resnet18_cifar10.pth'

# 数据预处理（测试用）
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# 测试数据加载
testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义模型
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 类别数
model = model.to(device)

# 加载模型权重
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")
else:
    raise FileNotFoundError(f"❌ Cannot find model file at {model_path}")

# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"📊 Test Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    test()
