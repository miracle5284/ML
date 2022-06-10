import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston, load_iris

from practice.assorted import *

boston = load_boston()
print(boston.keys())
type(boston)
boston.data
boston.target
boston.feature_names
boston_df = pd.DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df.shape
boston_df.head()
boston_df['MEDV'] = boston.target
print(boston_df.shape)
boston_df.head()
boston_df.boxplot()
# sns.boxplot(x=boston_df['DIS'])

lower_range, upper_range = outlier_treatment(boston_df['DIS'])

print('lower_range:', lower_range)
print('upper_range:', upper_range)
# print(boston_df[boston_df['DIS'].values < lower_range])
# print(boston_df[boston_df['DIS'].values > upper_range])
rim = Rim(boston_df, y='MEDV')
rim.drop_outliers('DIS')

print(rim.df.mean())
print(rim.df.skew(axis=0, skipna=True))
# rim.df.skew(axis=0, skipna=True).plot()

# pearson_corr = rim.df.corr(method='pearson')
# print(pearson_corr)

rim.correlation_treatment()

# Heat Map

# plt.figure(figsize=(10, 10), dpi=100)
# sns.heatmap(pearson_corr, xticklabels=pearson_corr.columns, yticklabels=pearson_corr.columns,
#             annot=True, linewidth=0.5)
# plt.show()


