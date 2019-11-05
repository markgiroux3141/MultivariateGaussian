from multiVariateGaussian import MultiVariateGaussian
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

x = [[0.1,0.1],[-0.1,0.1],[-0.1,-0.1],[0.1,-0.1]]

mu = MultiVariateGaussian.getMu(x)

#mu = np.array([0,0])

sig = MultiVariateGaussian.getSigma(x, mu)

#sig = np.array([[1,0.8],[0.8,1]])

val = [0.2,0.2]

p = MultiVariateGaussian.getMulVarGaussian(val, mu, sig)

x = np.array(x)
"""
plt.scatter(x[:,0], x[:,1])
plt.title('Multivariate Gaussian')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""
x_grid = np.linspace(-3,3,100)
y_grid = np.linspace(-3,3,100)

g = []
for i in range(len(y_grid)):
    g_row = []
    for n in range(len(x_grid)):
        g_row.append(MultiVariateGaussian.getMulVarGaussian([x_grid[n],y_grid[i]], mu, sig))
    g.append(g_row)
    
g = np.array(g)
        
ax = sns.heatmap(g)
print(mu)
print(sig)
print(p)