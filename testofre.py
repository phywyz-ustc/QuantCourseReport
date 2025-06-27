import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# ==== 1. 加载数据 ====
X         = np.load('D:\AI4Charting\大作业\processed2\X.npy')         # shape: (N, 12, 4)
y         = np.load('D:\AI4Charting\大作业\processed2\y.npy')         # shape: (N,)
data_type = np.load('D:\AI4Charting\大作业\processed2\data_type.npy')  # shape: (N,)
data_date = np.load('D:\AI4Charting\大作业\processed2\data_date.npy')  # shape: (N,)
returns   = np.load('D:/AI4Charting/大作业/processed2/returns.npy')  # shape: (N,)

# ==== 2. 选择测试集 ====
test_mask = data_type == 1
X_test = X[test_mask]
y_test = y[test_mask]
date_test = data_date[test_mask]
returns_test = returns[test_mask]

indices_test = np.where(test_mask)[0]  # 原始索引映射

# ==== 3. 设置 device 与模型 ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
model.load_state_dict(torch.load('D:/AI4Charting/大作业/models/lstm_model_without_volume_2200.pth', map_location=device))
print("Model loaded successfully: D:/AI4Charting/大作业/models/lstm_model_without_volume_2200.pth")
model.eval()

# ==== 4. 按日期分组 ====
date_to_indices = defaultdict(list)
for i, date in enumerate(date_test):
    date_to_indices[date].append(i)

# ==== 5. 对每个日期做预测并评估 ====
with torch.no_grad():
    all_results = []
    for date, idx_list in sorted(date_to_indices.items()):
        X_batch = torch.tensor(X_test[idx_list], dtype=torch.float32).to(device)
        preds = model(X_batch).squeeze().cpu().numpy()  # shape: (batch,)

        # 找出预测值最小的1个样本（即未来收益预期最高）
        top_indices = np.argsort(preds)[:1]
        real_idx = [indices_test[idx_list[i]] for i in top_indices]  # 映射回原始全局索引
        selected_returns = returns[real_idx]

        avg_return = selected_returns.mean()
        all_results.append(avg_return)
        print(f"Date: {date}, Selected 1 avg return: {avg_return:.4f}")

# ==== 6. 整体平均收益 ====
overall_avg = np.mean(all_results)
print(f"\nOverall average return of top-1 selections per date: {overall_avg:.4f}\n\n")

with torch.no_grad():
    all_results = []
    for date, idx_list in sorted(date_to_indices.items()):
        X_batch = torch.tensor(X_test[idx_list], dtype=torch.float32).to(device)
        preds = model(X_batch).squeeze().cpu().numpy()  # shape: (batch,)

        # 找出预测值最小的3个样本（即未来收益预期最高）
        top_indices = np.argsort(preds)[:3]
        real_idx = [indices_test[idx_list[i]] for i in top_indices]  # 映射回原始全局索引
        selected_returns = returns[real_idx]

        avg_return = selected_returns.mean()
        all_results.append(avg_return)
        print(f"Date: {date}, Selected 3 avg return: {avg_return:.4f}")

# ==== 6. 整体平均收益 ====
overall_avg = np.mean(all_results)
print(f"\nOverall average return of top-3 selections per date: {overall_avg:.4f}")