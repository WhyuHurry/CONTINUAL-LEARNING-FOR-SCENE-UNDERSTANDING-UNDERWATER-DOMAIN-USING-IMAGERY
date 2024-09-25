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
from torch.optim.lr_scheduler import StepLR
import logging

# 配置日志输出到文件
logging.basicConfig(
    filename='training_log_experiment2_with_ewc.txt',  # 日志文件名
    level=logging.INFO,  # 记录级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    filemode='w'  # 文件模式 'w' 表示每次运行时覆盖日志文件，使用 'a' 表示追加
)

# 确定设备 (CPU 或 GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 假设你有六个数据集的路径
dataset_paths = [
    'D:/project/dataset/1',
    'D:/project/dataset/2',
    'D:/project/dataset/3',
    'D:/project/dataset/4',
    'D:/project/dataset/5',
    'D:/project/dataset/6'
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

# 加载单个数据集
def load_datasets(data_dir):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

# EWC 类：包含 Fisher 信息矩阵的计算与 EWC 损失
class EWC:
    def __init__(self, model, dataloader, device, lambda_ewc=1000):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.lambda_ewc = lambda_ewc  # EWC 惩罚系数
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher_information()

    # 计算 Fisher 信息矩阵
    def _compute_fisher_information(self):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()

        for inputs, labels in self.dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).detach()

        fisher = {n: p / len(self.dataloader) for n, p in fisher.items()}  # 归一化 Fisher 信息
        return fisher

    # 计算 EWC 损失
    def compute_ewc_loss(self):
        ewc_loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]).pow(2)
                ewc_loss += _loss.sum()
        return self.lambda_ewc * ewc_loss

# 模型训练函数（含 EWC 损失）
def train_model_with_ewc(model, criterion, optimizer, dataloaders, dataset_sizes, ewc, scheduler, num_epochs=25):
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

                    # 在训练阶段计算 EWC 损失
                    if phase == 'train' and ewc is not None:
                        ewc_loss = ewc.compute_ewc_loss()
                        loss += ewc_loss

                    # 反向传播和优化
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
def evaluate_model(model, dataloader, dataset_size):
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

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return accuracy, precision, recall, f1

# 绘制训练过程的 loss 曲线
def plot_loss_curve(train_loss, val_loss, num_epochs, dataset_name):
    plt.figure()
    plt.plot(range(num_epochs), train_loss, label='Training Loss')
    plt.plot(range(num_epochs), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss over Epochs for {dataset_name}')
    plt.legend()
    plt.savefig(f'loss_curve_{dataset_name}.png')
    plt.close()

# 主函数：基于 EWC 的持续学习方法，每次评估所有任务的测试集
def train_with_ewc():
    model = None  # 初始化模型为None
    ewc = None  # 初始化EWC对象
    test_loaders = {}  # 用于存储每个任务的测试集 DataLoader
    all_test_datasets = []  # 用于存储每个任务的测试集数据集

    for idx, dataset_path in enumerate(dataset_paths):
        # 加载当前任务的数据集
        dataloaders, dataset_sizes, class_names = load_datasets(dataset_path)

        # 如果模型未初始化，加载预训练的ResNet-18模型
        if model is None:
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),  # 引入 Dropout
                nn.Linear(num_ftrs, len(class_names))
            )
            model = model.to(device)

        # 定义损失函数和优化器（加入L2正则化：weight_decay）
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)  # 加入 L2 正则化
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # 学习率调度器，每 7 个 epoch 后减少学习率

        # 训练当前任务，计算 EWC 损失
        model, train_loss, val_loss = train_model_with_ewc(model, criterion, optimizer, dataloaders, dataset_sizes, ewc, scheduler)

        # 更新 EWC 对象
        ewc = EWC(model, dataloaders['val'], device)

        # 存储当前任务的测试集 DataLoader 和数据集
        test_loaders[f'task_{idx+1}'] = dataloaders['test']
        all_test_datasets.append(dataloaders['test'].dataset)  # 存储当前测试集数据集

        # 评估模型在所有任务的测试集上的表现
        logging.info(f"\nEvaluating model performance after training task {idx+1}:")
        for task_name, test_loader in test_loaders.items():
            logging.info(f"Evaluating on {task_name}...")
            test_size = len(test_loader.dataset)
            accuracy, precision, recall, f1 = evaluate_model(model, test_loader, test_size)
            logging.info(f"Results for {task_name} -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")

        # 绘制当前任务的训练损失曲线
        dataset_name = os.path.basename(dataset_path)
        plot_loss_curve(train_loss, val_loss, len(train_loss), dataset_name)

    # 最后将所有任务的测试集合并
    combined_test_dataset = ConcatDataset(all_test_datasets)
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 在合并后的总测试集上进行评估
    total_test_size = len(combined_test_dataset)
    logging.info(f"\nFinal evaluation on the combined test dataset (size: {total_test_size}):")
    accuracy, precision, recall, f1 = evaluate_model(model, combined_test_loader, total_test_size)
    logging.info(f"Final Results -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")

# 确保程序在Windows上正确运行
if __name__ == '__main__':
    train_with_ewc()
