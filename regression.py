import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.metrics import mean_squared_error
import math

x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x)

X = x.reshape(-1, 1)
poly_features = PolynomialFeatures(degree=5)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

x_pred = np.linspace(-2 * np.pi, 2 * np.pi, 200)
X_pred = x_pred.reshape(-1, 1)
X_pred_poly = poly_features.transform(X_pred)
y_actual = np.sin(x_pred)

y_pred = model.predict(X_pred_poly)

plt.scatter(x, y, color='g', label='Actual')
plt.plot(x_pred, y_pred, color='r', label='Predicted')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Polynomial Regression for sin(x)')
plt.legend()
st.pyplot(plt)

mse = mean_squared_error(y_actual, y_pred)
rmse = math.sqrt(mse)
st.write("RMSE:")
st.write(rmse)

x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x)

X_custom = np.column_stack((np.sin(x), np.sin(x) ** 2, np.sin(x) ** 3))

model = LinearRegression()
model.fit(X_custom, y)

x_pred = np.linspace(-2 * np.pi, 2 * np.pi, 200)
X_pred_custom = np.column_stack((np.sin(x_pred), np.sin(x_pred) ** 2, np.sin(x_pred) ** 3))


y_pred = model.predict(X_pred_custom)
# Plot the actual sin(x) function and the predicted values
plt.figure()  # Create a new figure for the second plot
plt.scatter(x, y, color='r', label='Actual')
plt.plot(x_pred, y_pred, color='b', label='Predicted')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Polynomial Regression with Custom Features for sin(x)')
plt.legend()
plt.show()
st.pyplot(plt)


mse = mean_squared_error(y_actual, y_pred)
rmse = math.sqrt(mse)
st.write("RMSE:")
st.write(rmse)
