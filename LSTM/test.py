#%%
import pandas as pd
import torch
import torch.nn as nn 
from tqdm import *
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs") 


# 读取excel文件
data = pd.read_excel('Problem_C_Data_Wordle_2.xlsx')

# 获取第三行第四列向下的所有元素
words = data.iloc[1:, 3]

# 定义字母与独热码之间的映射关系
letter_hot_mapping = {
    'a': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'b': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'c': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'd': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'e': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'f': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'g': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'h': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'i': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'j': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'k': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'l': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'n': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'o': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'p': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    's': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    't': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'u': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'v': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'w': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'x': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

# 将每个单词的每个字母都表示成一个独热码
words_hotcode = []

for word in words:
    word_hotcode = []
    for letter in word.lower():
       word_hotcode.append(letter_hot_mapping[letter])
    words_hotcode.append(word_hotcode)

date = [1+x for x in range(len(words))]

results = data.iloc[1:, 6:13].values
results = np.array(results)/100
results = results.tolist()


words_hotcode = [list(np.ravel(x)) for x in words_hotcode]

# 时间向量转换
def split_windows(data,output,seq_length):

    x=[]
    y=[]
    for i in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _x=data[i:(i+seq_length)]
        _y=output[i+seq_length]
        x.append(_x)
        y.append(_y)
    x,y=np.array(x),np.array(y)
    print('x.shape,y.shape=\n',x.shape,y.shape)
    return x,y

input_data,output_data = split_windows(words_hotcode,results,20)


input_data =  torch.autograd.Variable(torch.Tensor(input_data))

output_data = torch.autograd.Variable(torch.Tensor(output_data))

# 定义超参数
learning_rate = 0.0001

num_epochs = 500

# 将张量A和B转换为TensorDataset对象，并将数据集分成训练集和测试集

dataset = TensorDataset(input_data, output_data)



train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])








# 定义前馈神经网络
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和记忆单元
        h0 = torch.zeros(self.num_layers,20,  self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers,20,self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out



input_size = 130
hidden_size = 64
num_layers = 2
output_size = 7
model = LSTM(input_size, hidden_size, num_layers, output_size)
print(model)

#%%
# 初始化模型和优化器

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 定义数据加载器
train_loader = DataLoader(train_dataset, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=True)



#%%
# 开始训练
train_losses = []
test_losses = []
for epoch in tqdm(range(num_epochs)):
    # 训练模型
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        writer.add_scalar("train_losses", loss.detach(), epoch)

#%%
# 测试模型
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)


        test_losses.append(loss.item())

        writer.add_scalar("test_losses", loss.detach(), epoch)
# 打印训练和测试损失
print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'
      .format(epoch+1, num_epochs, train_losses[-1], test_losses[-1]))

writer.close() 





plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

torch.save(model.state_dict(),'model_no_time')
train_losses=np.array(train_losses)
np.save('train_losses.npy',train_losses) 

test_losses=np.array(test_losses)
np.save('test_losses.npy',test_losses) 

# %%
