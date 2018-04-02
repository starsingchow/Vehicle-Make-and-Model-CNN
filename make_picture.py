#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#%%
data = pd.read_csv('~/Downloads/paperfile/alexnet_accuracy.csv')
data.info()

# fig, (axis1) = plt.subplots(1,1,figsize=(7,7))
# sns.set(style="white", palette="muted", color_codes=True)
# sns.kdeplot(data['Value'],color='r',ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(7,4))
plt.grid(True)
plt.rcParams['font.family'] = 'SimSun'
axis1.plot(data['Step'], data['Value'])
plt.title(u'AlexNet 50000次训练准确率变化图')
plt.xlabel(u'训练次数')
plt.ylabel(u'准确率')
plt.xlim(xmin=0,xmax=50000)
plt.ylim(ymin=0,ymax=1.05)
plt.savefig('alexnet_accuracy.png')
plt.show()


#%%
data = pd.read_csv('~/Downloads/paperfile/alexnet_loss.csv')
data.info()
data.head()

# fig, (axis1) = plt.subplots(1,1,figsize=(7,7))
# sns.set(style="white", palette="muted", color_codes=True)
# sns.kdeplot(data['Value'],color='r',ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(7,4))
plt.grid(True)
plt.rcParams['font.family'] = 'SimSun'
axis1.plot(data['Step'], data['Value'])
plt.title(u'AlexNet 50000次训练损失值变化图')
plt.xlabel(u'训练次数')
plt.ylabel(u'损失值')
plt.xlim(xmin=0,xmax=50000)
plt.savefig('alexnet_loss.png')
plt.show()

#%%
import os
import numpy as np
filenames = os.listdir('/Users/starsingchow/Downloads/train')
photos = []
for filename in filenames:
    if filename == '.DS_Store':
        continue
    file_path = os.path.join('/Users/starsingchow/Downloads/train', filename)
    photo = plt.imread(file_path)
    photos.append(photo)

k1 = np.concatenate((photos[0],photos[1],photos[2],photos[3]),axis=1)
k2 = np.concatenate((photos[4],photos[5],photos[6],photos[7]),axis=1)
all = np.concatenate((k1,k2), axis=0)
plt.imshow(all)
plt.xticks([])
plt.yticks([])
# plt.xlabel('处理后训练集的车辆图片', fontsize=12)
plt.rcParams['font.family'] = 'SimSun'
plt.savefig('处理后训练集的车辆图片.png',dpi=150, bbox_inches='tight')
plt.show()

#%%
set ={50000: 0.9856335952848723, 11000: 0.8229371316306483, 43000: 0.9817043222003929, 44000: 0.981458742632613, 29000: 0.9636542239685658, 12000: 0.8390225933202358, 47000: 0.9840373280943026, 7000: 0.7321954813359528, 18000: 0.9097495088408645, 46000: 0.9841601178781926, 33000: 0.9734774066797642, 28000: 0.9603388998035364, 31000: 0.9695481335952849, 26000: 0.9551817288801572, 36000: 0.9744597249508841, 20000: 0.9232563850687623, 38000: 0.9780206286836935, 17000: 0.9011542239685658, 8000: 0.7579813359528488, 1000: 0.15213654223968565, 39000: 0.9788801571709234, 14000: 0.868492141453831, 25000: 0.9495333988212181, 34000: 0.9745825147347741, 5000: 0.649926326129666, 10000: 0.8025540275049116, 30000: 0.9666011787819253, 48000: 0.9846512770137524, 41000: 0.980844793713163, 37000: 0.9765471512770137, 2000: 0.3025540275049116, 9000: 0.7820481335952849, 6000: 0.7007612966601179, 45000: 0.9841601178781926, 22000: 0.9354125736738703, 27000: 0.9577603143418467, 49000: 0.9851424361493124, 4000: 0.5758840864440079, 23000: 0.9400785854616895, 32000: 0.9708988212180747, 16000: 0.8914538310412574, 13000: 0.8573182711198428, 42000: 0.9802308447937131, 40000: 0.9805992141453831, 35000: 0.9739685658153242, 21000: 0.9295186640471512, 15000: 0.8783153241650294, 3000: 0.45714636542239684, 24000: 0.9460952848722987, 19000: 0.9179764243614931}

m = pd.Series(set)
m.index
m = m.sort_index()
fig, (axis1) = plt.subplots(1,1,figsize=(7,4))
plt.grid(True)
plt.rcParams['font.family'] = 'SimSun'
axis1.plot(m.index, m)
plt.title(u'MobileNet_1.0 50000次验证集准确率变化图')
plt.xlabel(u'训练次数')
plt.ylabel(u'准确率')
plt.xlim(xmin=0,xmax=50000)
plt.ylim(ymin=0,ymax=1.05)
plt.savefig('mobilenet_1.0_valid_accuracy.png',dpi=150)
plt.show()

#%%
import numpy as np

matrix = np.load('/Users/starsingchow/Downloads/alexnet_matrix.npy')

f, (ax1) = plt.subplots(figsize=(10,8))
plt.rcParams['font.family'] = 'SimSun'
ax1.set_title('AlexNet 测试集混淆矩阵', size= 20)
sns.heatmap(matrix,linewidths=0,cmap="YlGnBu",fmt="d",annot=False,xticklabels=50, yticklabels=50,ax=ax1)
ax1.set_xlabel('真标签',size = 20)
ax1.set_ylabel('预测标签', size = 20)
plt.savefig('AlexNet 测试集混淆矩阵.png', bbox_inches='tight',dpi=300)
plt.show()

#%%
import numpy as np

matrix = np.load('/Users/starsingchow/Downloads/googlenet_matrix.npy')

f, (ax1) = plt.subplots(figsize=(10,8))
plt.rcParams['font.family'] = 'SimSun'
ax1.set_title('GoogLeNet 测试集混淆矩阵', size= 20)
sns.heatmap(matrix,linewidths=0,cmap="YlGnBu",fmt="d",annot=False,ax=ax1)
ax1.set_xlabel('真标签',size = 20)
ax1.set_ylabel('预测标签', size = 20)
plt.savefig('GoogLeNet 测试集混淆矩阵_1.png', bbox_inches='tight',dpi=300)
plt.show()