from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn


# 下载到本地 from https://huggingface.co/microsoft/resnet-50/tree/main
local_model_path = "C:/Users/23784/Desktop/北大大三上/人工智能系统实践1/module 0/resnet"
image_processor = AutoImageProcessor.from_pretrained(local_model_path)
model = ResNetForImageClassification.from_pretrained(local_model_path)

# 添加一个全连接层，将输出维度从 1000 维压缩为 10 维
class CustomResNet(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1000, 10)  

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        logits = self.fc(outputs.logits)  
        return logits

# 创建自定义的 ResNet 模型
custom_model = CustomResNet(model)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5],std=[0.23])
])


mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

# 计算准确率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model.to(device)
custom_model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        inputs = {"pixel_values": images}
        logits = custom_model(**inputs)
        
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

