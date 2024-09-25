import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import copy
import logging
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet18_Weights

# 配置日志输出到文件
logging.basicConfig(
    filename='training_log_experiment3_without_replay.txt',  # 日志文件名
    level=logging.INFO,  # 记录级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    filemode='a'  # 文件模式 'w' 表示每次运行时覆盖日志文件，使用 'a' 表示追加
)

# 确定设备 (CPU 或 GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集路径
dataset_paths = [
    'D:/project/dataset/1',
    'D:/project/dataset/2',
    'D:/project/dataset/3'
]

# 图像预处理和数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 数据增强：颜色抖动
        transforms.RandomRotation(20),  # 数据增强：随机旋转
        transforms.RandomVerticalFlip(),  # 数据增强：随机垂直翻转
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

# 加载数据集
def load_datasets(data_dir):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

# 模型标准训练函数（每个任务只训练新任务数据，不回放旧任务数据）
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
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

# 绘制 loss 和 epoch 曲线
def plot_loss_curve(train_loss, val_loss, num_epochs, dataset_name, save_dir='plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    plt.plot(range(num_epochs), train_loss, label='Training Loss')
    plt.plot(range(num_epochs), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss over Epochs for {dataset_name}')
    plt.legend()
    # 生成完整的保存路径
    save_path = os.path.join(save_dir, f'loss_curve_{dataset_name}.png')

    # 直接保存图像到指定路径而不显示窗口
    plt.savefig(save_path)  # 保存为 PNG 格式文件
    plt.close()  # 关闭绘图窗口

    print(f"Saved loss curve for {dataset_name} at {save_path}")

# 测试模型并计算性能指标
def evaluate_model(model, dataloader, dataset_size, dataset_name=""):
    model.eval()  # 设置为评估模式
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
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} Precision: {precision:.4f}")
    print(f"{dataset_name} Recall: {recall:.4f}")
    print(f"{dataset_name} F1 Score: {f1:.4f}")

    logging.info(f"{dataset_name} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return accuracy, precision, recall, f1

# 主函数：训练新任务时，不使用旧任务数据，只训练新任务的数据
def train_without_replay():
    model = None
    all_test_datasets = []  # 保存所有任务的测试集数据集
    test_loaders = {}  # 用于保存每个任务的测试集

    for idx, dataset_path in enumerate(dataset_paths):
        # 加载当前任务数据集
        dataloaders, dataset_sizes, class_names = load_datasets(dataset_path)

        # 初始化模型
        if model is None:
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),  # 添加 Dropout 防止过拟合
                nn.Linear(num_ftrs, len(class_names))
            )
            model = model.to(device)

        # 定义损失函数、优化器和学习率调度器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)  # 加入 L2 正则化
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # 对当前任务进行标准的训练过程，不使用旧任务数据
        model, train_loss, val_loss = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes)

        # 保存当前任务的测试集数据集
        all_test_datasets.append(dataloaders['test'].dataset)

        # 绘制当前任务的loss曲线
        plot_loss_curve(train_loss, val_loss, len(train_loss), f"Dataset_{idx+1}")

        # 评估模型在当前任务上的表现
        evaluate_model(model, dataloaders['test'], dataset_sizes['test'], f"Dataset {idx+1}")

        # 在所有之前任务的测试集上进行评估
        print(f"\nEvaluating on all previous tasks after training task {idx+1}:")
        for task_idx, task_dataloader in enumerate(test_loaders.values(), 1):
            evaluate_model(model, task_dataloader, len(task_dataloader.dataset), f"Task {task_idx}")

        # 保存当前任务的测试集到测试加载器中
        test_loaders[f'task_{idx+1}'] = dataloaders['test']

    # 所有任务完成后，合并所有测试集进行评估
    print("\nFinal evaluation on the combined test dataset (all tasks):")
    combined_test_dataset = ConcatDataset(all_test_datasets)
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=32, shuffle=False, num_workers=4)
    evaluate_model(model, combined_test_loader, len(combined_test_dataset), "Combined Test Dataset")


# 确保程序在Windows上正确运行
if __name__ == '__main__':
    train_without_replay()
