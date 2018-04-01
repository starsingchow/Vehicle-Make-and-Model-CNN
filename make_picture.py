#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#%%
data = pd.read_csv('~/Downloads/paperfile/googlenet_accuracy.csv')
data.info()

# fig, (axis1) = plt.subplots(1,1,figsize=(7,7))
# sns.set(style="white", palette="muted", color_codes=True)
# sns.kdeplot(data['Value'],color='r',ax=axis1)
plt.plot(data['Step'], data['Value'])
plt.show()
