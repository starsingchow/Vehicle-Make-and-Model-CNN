#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#%%
data = pd.read_csv('~/Downloads/paperfile/mobilenet_1.0_accuracy.csv')
data.info()

# fig, (axis1) = plt.subplots(1,1,figsize=(7,7))
# sns.set(style="white", palette="muted", color_codes=True)
# sns.kdeplot(data['Value'],color='r',ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(7,4))
plt.grid(True)
plt.rcParams['font.family'] = 'SimSun'
axis1.plot(data['Step'], data['Value'])
plt.title(u'MobileNet_1.0 50000次训练准确率变化图')
plt.xlabel(u'训练次数')
plt.ylabel(u'准确率')
plt.xlim(xmin=0,xmax=50000)
plt.ylim(ymin=0,ymax=1.05)
plt.savefig('mobilenet1.0_accuracy.png')
plt.show()


#%%
data = pd.read_csv('~/Downloads/paperfile/mobilenet_1.0_loss.csv')
data.info()
data.head()

# fig, (axis1) = plt.subplots(1,1,figsize=(7,7))
# sns.set(style="white", palette="muted", color_codes=True)
# sns.kdeplot(data['Value'],color='r',ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(7,4))
plt.grid(True)
plt.rcParams['font.family'] = 'SimSun'
axis1.plot(data['Step'], data['Value'])
plt.title(u'MobileNet_1.0 50000次训练损失值变化图')
plt.xlabel(u'训练次数')
plt.ylabel(u'损失值')
plt.xlim(xmin=0,xmax=50000)
plt.savefig('MobileNet_1.0_loss.png')
plt.show()