import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import copy
from torchvision.models import ResNet18_Weights
import logging
from torch.optim.lr_scheduler import StepLR

# 配置日志输出到文件
logging.basicConfig(
    filename='training_log_experiment1.txt',  # 日志文件名
    level=logging.INFO,  # 记录级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    filemode='w'  # 文件模式 'w' 表示每次运行时覆盖日志文件
)

# 确定设备 (CPU 或 GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集1-6的路径
dataset_paths = [
    'D:/project/dataset/1',
    'D:/project/dataset/2',
    'D:/project/dataset/3',
    'D:/project/dataset/4',
    'D:/project/dataset/5'
]

# 数据增强：图像预处理和数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.RandomRotation(20),  # 随机旋转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载单个数据集
def load_dataset(data_dir):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    return dataloaders, dataset_sizes

# 加载数据集1-6的总和，形成数据集7
def load_combined_dataset():
    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []

    for path in dataset_paths:
        dataloaders, dataset_sizes = load_dataset(path)
        all_train_datasets.append(dataloaders['train'].dataset)
        all_val_datasets.append(dataloaders['val'].dataset)
        all_test_datasets.append(dataloaders['test'].dataset)

    # 合并数据集1-5
    combined_train_dataset = ConcatDataset(all_train_datasets)
    combined_val_dataset = ConcatDataset(all_val_datasets)
    combined_test_dataset = ConcatDataset(all_test_datasets)

    # 创建数据集7的数据加载器
    dataloaders = {
        'train': DataLoader(combined_train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(combined_val_dataset, batch_size=32, shuffle=False, num_workers=4),
        'test': DataLoader(combined_test_dataset, batch_size=32, shuffle=False, num_workers=4)
    }

    dataset_sizes = {
        'train': len(combined_train_dataset),
        'val': len(combined_val_dataset),
        'test': len(combined_test_dataset)
    }

    return dataloaders, dataset_sizes

# 模型训练函数
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每一轮训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 验证模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                scheduler.step()  # 更新学习率
            else:
                val_loss_history.append(epoch_loss)

            # 保存验证集上表现最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val Acc: {best_acc:.4f}')
    logging.info(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history

# 测试模型并计算准确率、精确率、召回率、F1分数
def evaluate_model(model, dataloader, dataset_size, dataset_name=""):
    model.eval()  # 测试模式
    running_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} Precision: {precision:.4f}")
    print(f"{dataset_name} Recall: {recall:.4f}")
    print(f"{dataset_name} F1 Score: {f1:.4f}")

    logging.info(f"{dataset_name} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return accuracy, precision, recall, f1

# 绘制训练过程的 loss 曲线
def plot_loss_curve(train_loss, val_loss, num_epochs):
    plt.figure()
    plt.plot(range(num_epochs), train_loss, label='Training Loss')
    plt.plot(range(num_epochs), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig('loss_curve_experiment1.png')
    plt.show()

# 主函数，训练并评估模型
def train_and_evaluate_on_combined_dataset():
    # 加载合并后的数据集1-6
    dataloaders, dataset_sizes = load_combined_dataset()

    # 从第一个数据集获取类名（适用于所有数据集）
    class_names = dataloaders['train'].dataset.datasets[0].classes

    # 初始化预训练的 ResNet-18 模型
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用 weights 参数代替 pretrained
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # 添加 Dropout 防止过拟合
        nn.Linear(num_ftrs, len(class_names))  # 使用从第一个数据集获取的 classes
    )
    model = model.to(device)

    # 定义损失函数、优化器和学习率调度器（加入L2正则化）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)  # 加入L2正则化
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epoch减少学习率

    # 训练模型
    model, train_loss, val_loss = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes)

    # 在合并后的总测试集上评估
    accuracy, precision, recall, f1 = evaluate_model(model, dataloaders['test'], dataset_sizes['test'], "Combined Dataset")

    # 绘制训练损失曲线
    plot_loss_curve(train_loss, val_loss, len(train_loss))

    # 在每个数据集的单独测试集上进行评估
    for i, path in enumerate(dataset_paths, 1):
        dataloaders_i, dataset_sizes_i = load_dataset(path)
        evaluate_model(model, dataloaders_i['test'], dataset_sizes_i['test'], f"Dataset {i}")



# 确保程序在Windows上正确运行
if __name__ == '__main__':
    train_and_evaluate_on_combined_dataset()
