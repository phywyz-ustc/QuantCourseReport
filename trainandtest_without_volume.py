import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from scipy.stats import norm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 读取数据
X = np.load('D:\AI4Charting\大作业\processed2\X.npy')         # shape: (N, 12, 4)
y = np.load('D:\AI4Charting\大作业\processed2\y.npy')         # shape: (N,)
data_type = np.load('D:\AI4Charting\大作业\processed2\data_type.npy')  # shape: (N,)
data_date = np.load('D:\AI4Charting\大作业\processed2\data_date.npy')  # shape: (N,)
# 划分训练和测试
X_train, y_train, data_date_train = X[data_type == 0], y[data_type == 0], data_date[data_type == 0]
X_test, y_test, data_date_test = X[data_type == 1], y[data_type == 1], data_date[data_type == 1]
print(f"训练集样本: {X_train.shape}, 测试集样本: {X_test.shape}")
print(f"训练集标签: {y_train.shape}, 测试集标签: {y_test.shape}")
# 转为 torch tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
#print(X_train)
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出一个值
        )
        self._init_weights()

    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.lstm(x)  # shape: (batch, seq, hidden_dim)
        out = out[:, -1, :]    # 取最后一个时间步的输出
        out = self.regressor(out)
        return out
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

model.eval()
with torch.no_grad():
    total_loss = 0
    all_preds = []
    all_targets = []
    for X_batch, y_batch in train_loader:
        #print(X_batch)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        total_loss += loss.item()
        all_preds.append(preds.cpu())
        all_targets.append(y_batch.cpu())
    avg_loss = total_loss / len(train_loader)
    print(f"Initial training set loss before training: {avg_loss:.4f}")

y_pred = torch.cat(all_preds).numpy()
y_pred = np.squeeze(y_pred)
y_true = torch.cat(all_targets).numpy()
y_true = np.squeeze(y_true)
print(f"Predicted labels: {y_pred[:10]}")
print(f"True labels: {y_true[:10]}")
# 画分布图
plt.figure(figsize=(8, 5))
plt.hist(y_true, bins=50, alpha=0.6, label='True Labels', color='blue', density=True)
plt.hist(y_pred, bins=50, alpha=0.6, label='Predicted', color='orange', density=True)
plt.title('Distribution of True vs Predicted Labels (Train Set)')
plt.xlabel('Value (0 ~ 1)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 训练过程
loss_history = []
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4)
for epoch in range(40):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    scheduler.step()

#间隔取
plt.figure(figsize=(10, 5))
plt.plot(loss_history[::10], label='Training Loss')
plt.title('Training Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('D:/AI4Charting/大作业/models/lstm_training_loss_without_volume.png')
plt.show()

model.eval()
with torch.no_grad():
    total_loss = 0
    all_preds = []
    all_targets = []
    for X_batch, y_batch in train_loader:
        #print(X_batch)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        total_loss += loss.item()
        all_preds.append(preds.cpu())
        all_targets.append(y_batch.cpu())
    avg_loss = total_loss / len(train_loader)
    print(f"Training set loss after training: {avg_loss:.4f}")

y_pred = torch.cat(all_preds).numpy()
y_pred = np.squeeze(y_pred)
y_true = torch.cat(all_targets).numpy()
y_true = np.squeeze(y_true)
print(f"Predicted labels: {y_pred[:10]}")
print(f"True labels: {y_true[:10]}")
# 画分布图
plt.figure(figsize=(8, 5))
plt.hist(y_true, bins=50, alpha=0.6, label='True Labels', color='blue', density=True)
plt.hist(y_pred, bins=50, alpha=0.6, label='Predicted', color='orange', density=True)
plt.title('Distribution of True vs Predicted Labels (Train Set)')
plt.xlabel('Value (0 ~ 1)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('D:/AI4Charting/大作业/models/lstm_distribution_train_without_volume.png')
plt.show()

model.eval()
with torch.no_grad():
    total_loss = 0
    all_preds = []
    all_targets = []
    for X_batch, y_batch in test_loader:
        #print(X_batch)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        total_loss += loss.item()
        all_preds.append(preds.cpu())
        all_targets.append(y_batch.cpu())
    avg_loss = total_loss / len(test_loader)
    print(f"Testing set loss after training: {avg_loss:.4f}")

y_pred = torch.cat(all_preds).numpy()
y_pred = np.squeeze(y_pred)
y_true = torch.cat(all_targets).numpy()
y_true = np.squeeze(y_true)
print(f"Predicted labels: {y_pred[:10]}")
print(f"True labels: {y_true[:10]}")
# 画分布图
plt.figure(figsize=(8, 5))
plt.hist(y_true, bins=50, alpha=0.6, label='True Labels', color='blue', density=True)
plt.hist(y_pred, bins=50, alpha=0.6, label='Predicted', color='orange', density=True)
plt.title('Distribution of True vs Predicted Labels (Test Set)')
plt.xlabel('Value (0 ~ 1)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('D:/AI4Charting/大作业/models/lstm_distribution_test_without_volume.png')
plt.show()

torch.save(model.state_dict(), 'D:/AI4Charting/大作业/models/lstm_model_without_volume.pth')

