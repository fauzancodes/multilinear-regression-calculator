import math as mt
import statistics as sts
import numpy as np
import scipy.stats as scs
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#loading input file
print("\n","Data Requirements:", "\n",
    "1. Your data should have at least 5 column with the same length", "\n",
    "2. Three column for X-axis, one column for Y-axis, one column for color bar","\n",
    "3. Your data should have extension .txt or .dat","\n"
    )

user_input = input("Enter your input filename with extension (ex: input.txt): ")
column_x1 = input("The data for the X1-axis are in column (ex: 1 or 2 or etc.): ")
column_x2 = input("The data for the X2-axis are in column (ex: 1 or 2 or etc.): ")
column_x3 = input("The data for the X3-axis are in column (ex: 1 or 2 or etc.): ")
column_y = input("The data for the Y-axis are in column (ex: 1 or 2 or etc.): ")
column_colorbar = input("The data for the color bar are in column (ex: 1 or 2 or etc.): ")

fileInput = np.loadtxt(user_input)

#defining
x1 = np.array(fileInput[:, (int(column_x1) - 1)])
x2 = np.array(fileInput[:, (int(column_x2) - 1)])
x3 = np.array(fileInput[:, (int(column_x3) - 1)])
x = np.stack((x1, x2, x3), axis = 1)
y = np.array(fileInput[:, (int(column_y) - 1)])
colorbar = np.array(fileInput[:, (int(column_colorbar) - 1)])

#regression x1, x2 againts y
model = LinearRegression().fit(x, y)
score = model.score(x, y)
intercept = model.intercept_
slope = model.coef_
y_pred = model.predict(x)

#remove values lower than 0
remove = input("Do you want to remove values that lower than zero on the predicted result? (y/n) ")

if remove == "y" :
    for i in range(len(y_pred)) :
        if y_pred[i] < 0 :
            y_pred[i] = 0

#correlation coefficient actual y vs. predicted y
r = np.corrcoef(y, y_pred)[0, 1]

#display
print("\n")
print("\n", "The independents Variables: ", "\n", x)
print("\n", "The Dependent Variables: ", "\n", y)
print("\n", "The Coefficient of Determination: ", "\n", score)
print("\n", "The Intercept: ", "\n", intercept)
print("\n", "The Slopes: ", "\n", slope)
print("\n", "The Predicted Response: ", "\n", y_pred)
print("\n", "The Pearson-Product Moment Correlation Coefficient of Actual Response and Predicted Response: ", "\n", r)

#ploting actual y vs. predicted y
print("\n")
x_label = input("Enter the label of X-axis: ")
y_label = input("Enter the label of Y-axis: ")
c_label = input("Enter the label of color bar: ")
print("\n")

plt.scatter(y, y_pred, c = colorbar)

plt.grid

font1 = {"family":"serif","color":"#1D1D1D","size":20}
font2 = {"family":"serif","color":"#1D1D1D","size":15}

plt.colorbar().ax.set_ylabel(c_label, fontdict = font2)

plt.xlabel(x_label, fontdict = font2)
plt.ylabel(y_label, fontdict = font2)
plt.title("Actual vs. Predicted Plot, r = " + str(np.around(r, 3)), fontdict = font1)

plt.show()