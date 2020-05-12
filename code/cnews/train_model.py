import torch
from torch import nn
from torch import optim
import numpy as np
from model import TextRNN
from cnews_loader import read_category, read_vocab, process_file
import torch.utils.data as Data #将数据分批次需要用到
import matplotlib.pyplot as plt

train_file = 'cnews.train.txt'
test_file = 'cnews.test.txt'
val_file = 'cnews.val.txt'
vocab_file = 'cnews.vocab.txt'

train_loss = []
val_loss = []
train_acc = []
val_acc = []

cuda = torch.device('cuda')

def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call # 在第一次调用该函数时设置子图
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot) #设置子图
    ax.set_facecolor('#F8F8F8') #设置背景颜色
    ax.plot(training) #画训练集的曲线
    ax.plot(validation) #画测试集的曲线
    ax.set_title('model '+ title)
    ax.set_ylabel(title) #设置y轴标题
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch') #设置x轴标题
    ax.legend(['train', 'valid.']) #设置图例
    plt.savefig("model.png")

def train():
  #使用TextRNN
  model = TextRNN().cuda()
  #损失函数
  Loss = nn.MultiLabelSoftMarginLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  best_val_acc = 0
  for epoch in range(20):
    print('epoch:', epoch+1)
    for step, (x_batch, y_batch) in enumerate(train_loader):
      x = x_batch.cuda()
      y = y_batch.cuda()
      out = model(x)
      loss = Loss(out, y)
      print('loss=', loss)
      train_loss.append(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #计算准确率
      accuracy = np.mean((torch.argmax(out,1) == torch.argmax(y,1)).cpu().numpy())
      print('accuracy=', accuracy)
      train_acc.append(accuracy)
      #对模型进行验证
      if (epoch+1)%2 == 0:
        with torch.no_grad():
          for step, (x_batch, y_batch) in enumerate(val_loader):
            x = x_batch.cuda()
            y = y_batch.cuda()
            out = model(x)
            loss_val = Loss(out, y)
            val_loss.append(loss_val)
            accuracy = np.mean((torch.argmax(out,1) == torch.argmax(y,1)).cpu().numpy())
            val_acc.append(accuracy)
            if accuracy > best_val_acc:
              torch.save(model.state_dict(), 'model_params.pkl')
              best_val_acc = accuracy
              print('val_accuracy=', accuracy)
  display_training_curves(train_loss, val_loss, 'loss', 211)
  display_training_curves(train_acc, val_acc, 'acc', 212)

if __name__=="__main__":
	# 获取文本的类别及其对应id的字典
	categories, cat_to_id = read_category()
	#print(categories)
	# 获取训练文本中所有出现过的字及其所对应的id
	words, word_to_id = read_vocab('./code/cnews/cnews.vocab.txt')
	#print(words)
	#print(word_to_id)
	#print(word_to_id)
	#获取字数
	vocab_size = len(words)

	# 数据加载及分批
	# 获取训练数据每个字的id和对应标签的one-hot形式
	x_train, y_train = process_file('./code/cnews/cnews.train.txt', word_to_id, cat_to_id, 600)
	print('x_train=', x_train)
	x_val, y_val = process_file('./code/cnews/cnews.val.txt', word_to_id, cat_to_id, 600)

	x_train, y_train = torch.LongTensor(x_train), torch.Tensor(y_train)
	x_val, y_val = torch.LongTensor(x_val), torch.Tensor(y_val)

	#数据分批
	train_dataset = Data.TensorDataset(x_train, y_train)
	train_loader = Data.DataLoader(dataset=train_dataset, batch_size=500, shuffle=True, num_workers=2)

	val_dataset = Data.TensorDataset(x_val, y_val)
	val_loader = Data.DataLoader(dataset=val_dataset, batch_size=500, shuffle=True, num_workers=2)
	
	train()