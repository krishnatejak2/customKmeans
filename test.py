
from numpy.linalg import norm

import numpy as np  
import scipy.stats
import pandas as pd
import seaborn as sns
import random
from matplotlib import cm
import matplotlib.pyplot as plt
# %matplotlib inline
 
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans,DBSCAN,SpectralClustering,AgglomerativeClustering,Birch
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import svm

n = 500
mean_1 = [0, 0]
cov_1 = [[20, 20], [20, 100]]
# x,y = np.random.multivariate_normal(mean_1,cov_1,n).T

mean_2 = [10, 50]
cov_2 = [[10, 70], [70, 100]]
# x_2,y_2 = np.random.multivariate_normal(mean_2,cov_2,n).T
random.seed(123)
x, y = np.append(
    np.random.multivariate_normal(mean_1, cov_1, n).T,
    np.random.multivariate_normal(mean_2, cov_2, n).T,
    axis=1,
)

data_2 = pd.DataFrame({'x':x,'y':y})
del x,y
data_2['label'] = np.append(np.repeat(0,n),np.repeat(1,n),axis = 0)
data_2['Constrained_label'] = [2 if (xx>0 and yy < 0) else zz for (xx,yy,zz) in zip(data_2.x,data_2.y,data_2.label)]
data_2.reset_index(inplace=True)
sns.scatterplot(data_2.x,data_2.y,hue = data_2.Constrained_label).set_title("2D Random Bimodal Distribution")
plt.savefig("./KMeans_ExpectedOutput.png")