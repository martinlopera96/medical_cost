import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from regressors import stats


data_path = r'C:\Users\Martín\Documents\DATA SCIENCE\Platzi\ML_projects\linear_regression\medical_cost\insurance.csv'
df = pd.read_csv(data_path)

df.head()
print(df.dtypes)
print(df.shape)
df.charges.hist(bins=40)

df = df[df['charges'] < 50000]

sns.pairplot(df, height=2.5)
plt.show()

numeric_cols = ['age', 'bmi', 'children', 'charges']
correlation_matrix = np.corrcoef(df[numeric_cols].values.T)
sns.set(font_scale=1.5)
sns.heatmap(correlation_matrix, annot=True, yticklabels=numeric_cols, xticklabels=numeric_cols)

df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
df.head()

X_cols = list(set(df.columns) - {'charges'})
y_col = ['charges']

X = df[X_cols].values
y = df[y_col].values

X_train, X_test, y_train, y_test = train_test_split(X,y)
sc_x = StandardScaler().fit(X)
sc_y = StandardScaler().fit(y)

X_train = sc_x.transform(X_train)
X_test = sc_x.transform(X_test)
y_train = sc_y.transform(y_train)
y_test = sc_y.transform(y_test)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred.shape)

mse = metrics.mean_squared_error(y_test, y_pred)
print("mse: ", mse.round(4))
r2 = metrics.r2_score(y_test, y_pred)
print("r2 ", r2.round(4))

model.intercept_ = model.intercept_[0]
model.coef_ = model.coef_.reshape(-1)

y_test = y_test.reshape(-1)

print("==========Summary==========")
stats.summary(model, X_test, y_test, X_cols)

residuals = np.substract(y_test, y_pred.reshape(-1))
plt.scatter(y_pred, residuals)
plt.show()

# Second model. Let's improve the analysis

df_second = df.copy()
df_second['age2'] = df_second.age**2
df_second['sobrepeso'] = (df_second.bmi >= 30).astype(int)
df_second['sobrepeso*fumador'] = df_second.sobrepeso * df_second.smoker_yes

X_cols = ['sobrepeso*fumador', 'smoker_yes', 'age2', 'children']
y_col = ['charges']

X = df_second[X_cols].values
y = df_second[y_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y)
sc_x = StandardScaler().fit(X)
sc_y = StandardScaler().fit(y)

X_train = sc_x.transform(X_train)
X_test = sc_x.transform(X_test)
y_train = sc_y.transform(y_train)
y_test = sc_y.transform(y_test)

model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("r2: ", r2.round(4))
print("mse :", mse.round(4))

model.coef_ = model.coef_.reshape(-1)

y_test = y_test.reshape(-1)

print("==========Summary==========")
stats.summary(model, X_test, y_test, X_cols)

residuals = np.substract(y_test, y_pred.reshape(-1))
plt.scatter(y_pred, residuals)
plt.show()
