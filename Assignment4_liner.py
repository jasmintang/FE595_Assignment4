
# the Linear Regression class and the Boston housing data set

from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine


boston = load_boston()
print(boston.data.shape)

from sklearn.linear_model import LinearRegression

data_x = boston.data
data_y = boston.target
model = LinearRegression()
model.fit(data_x, data_y)
print("coef")
print(model.coef_)
maxeffect = max(model.coef_)
for i in range(len(model.coef_)):
    if model.coef_[i] == maxeffect:
        print(i)
print("RM average number of rooms per dwelling has highest effection")
