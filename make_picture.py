#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#%%
data = pd.read_csv('~/Downloads/paperfile/googlenet_accuracy.csv')
data.info()
