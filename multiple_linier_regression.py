import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#loading input file
print("\n")
user_input = input("Enter your input filename with extension (ex: input.txt): ")
file_input = np.loadtxt(user_input)

#defining input y
print("\n")
column_y = input("The data to be predicted (input y) are in column (ex: 1 or 2 or etc.): ")
y = np.array(file_input[:, (int(column_y) - 1)])
print("\n")
print("Input Y", "\n", y)

#defining input x
print("\n")
number_input_x = input("How many data do you want to use to predict (input x)? : ")

i = 0

x = []

while i < int(number_input_x) :
    print("\n")
    print("Input X", i + 1)
    column_x = input("Input are in column: ")
    input_x =  np.array(file_input[:, (int(column_x) - 1)])
    print("Input Data:", "\n", input_x)
    x.append(input_x)
    i = i + 1

x = np.stack(x, axis = 1)

print("\n")
print("Input X", "\n", x)

#regression x against y
model = LinearRegression().fit(x, y)
score = model.score(x, y)
intercept = model.intercept_
slope = model.coef_
y_pred = model.predict(x)

print("\n")
print("The Coefficient of Determination: ", "\n", score)
print("\n")
print("The Intercept: ", "\n", intercept)
print("\n")
print("The Slopes: ", "\n", slope)
print("\n")
print("The Predicted Response: ", "\n", y_pred)

#remove values lower than 0
print("\n")
remove_aggrement = input("Do you want to remove values that lower than zero on the predicted result? (y/n) : ")

if remove_aggrement == "y" :
    for i in range(len(y_pred)) :
        if y_pred[i] < 0 :
            y_pred[i] = 0

#correlation coefficient actual y against predicted y
r = np.corrcoef(y, y_pred)[0, 1]

print("\n")
print("\n", "The Pearson-Product Moment Correlation Coefficient of Actual Response and Predicted Response: ", "\n", r)

#linear regression actual y against predicted y
regression_act_pre_aggrement = input("Do you want to draw linear regression line between actual y against predicted y? (y/n) : ")

if regression_act_pre_aggrement == "y" :
    model_y = LinearRegression().fit((y.reshape(-1, 1)), y_pred)
    score_y = model_y.score(y.reshape(-1, 1), y_pred)
    intercept_y = model_y.intercept_
    slope_y = model_y.coef_
    y_pred_y = model_y.predict(y.reshape(-1, 1))

    eq_y = "(" + str(float(np.around(slope_y, 3))) + ")X + " + "(" + str(float(np.around(intercept_y, 3))) + ")"

#defining the colorbar for the plot
print("\n")
column_colorbar_aggrement = input("Do you want to use the colorbar for the plot? (y/n) : ")

if column_colorbar_aggrement == "y" :
    column_colorbar = input("The data for the colorbar are in column (ex: 1 or 2 or etc.): ")
    colorbar = np.array(file_input[:, (int(column_colorbar) - 1)])

    print("\n")
    colorbar_negate_aggrement = input("Do you want to negate the colorbar data? (y/n) : ")
    
    if colorbar_negate_aggrement == "y" :
        colorbar = np.multiply(colorbar, -1)
        
        print("\n")
        print("The Colorbar: ", "\n", colorbar)

#ploting actual y against predicted y
print("\n")
x_label = input("Enter the label of X-axis: ")
y_label = input("Enter the label of Y-axis: ")

if column_colorbar_aggrement == "y" :
    c_label = input("Enter the label of color bar: ")

font1 = {"family":"serif","color":"#1D1D1D","size":20}
font2 = {"family":"serif","color":"#1D1D1D","size":15}

if column_colorbar_aggrement == "y" :
    plt.scatter(y, y_pred, c = colorbar)
else :
    plt.scatter(y, y_pred)

if regression_act_pre_aggrement == "y" :
    plt.plot(y, y_pred_y, label = eq_y, c = "black")

plt.grid
plt.legend(loc = "lower right")

if column_colorbar_aggrement == "y" :
    plt.colorbar().ax.set_ylabel(c_label, fontdict = font2)

plt.xlabel(x_label, fontdict = font2)
plt.ylabel(y_label, fontdict = font2)
plt.title("Actual vs. Predicted Plot, r = " + str(np.around(r, 3)), fontdict = font1)

plt.show()

print("\n")