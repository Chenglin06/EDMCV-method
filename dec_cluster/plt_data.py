# coding=gbk
import pandas as pd
import matplotlib.pyplot as plt


#exit()
# 读取xlsx文件
data = pd.read_excel('weight/train.xlsx')
# 获取数据
epochs = data['epoch'].values
loss = data['loss'].values
acc = data['acc'].values


#print(epochs)
#exit()

# 创建一个新的figure
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制loss
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color='tab:red')
ax1.plot(epochs, loss, color='tab:red', label='Loss')
ax1.tick_params(axis='y', labelcolor='tab:red')



# 使用双y轴绘制accuracy
ax2 = ax1.twinx()  # 创建第二个y轴
ax2.set_ylabel('acc', color='tab:blue')
ax2.plot(epochs, acc, color='tab:blue', label='Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# 显示图形
plt.title("Loss and Accuracy vs. Epoch")
plt.savefig('Loss and Accuracy vs. Epoch.png',dpi = 600)
#plt.show()
