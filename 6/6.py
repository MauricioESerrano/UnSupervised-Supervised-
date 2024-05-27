import csv
import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# ----------------------------------------
                # Part a
# ----------------------------------------


# dataFile = "data.csv"

# # Empty lists to store coordinates
# x_coord = []
# y_coord = []

# # Reader for file data.csv
# with open(dataFile, mode='r', newline='') as file:
   
#     reader = csv.reader(file)
#     next(reader)

#     # Store data into arrays
#     for row in reader:
#         x_coord.append(float(row[1]))  
#         y_coord.append(float(row[2]))  

# # Convert to numpy arrays
# x_array = np.array(x_coord)
# y_array = np.array(y_coord)

# d2 = Polynomial.fit(x_array,y_array,2)

# xfit = np.linspace(min(x_array),max(x_array),100)

# yfit = d2(xfit)

# plt.figure(figsize=(10,6))

# plt.scatter(x_array,y_array)

# plt.plot(xfit,yfit)

# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()



# ----------------------------------------
                # Part b
# ----------------------------------------


# dataFile = "data.csv"

# # Empty lists to store coordinates
# x_coord = []
# y_coord = []

# # Reader for file data.csv
# with open(dataFile, mode='r', newline='') as file:
   
#     reader = csv.reader(file)
#     next(reader)

#     # Store data into arrays
#     for row in reader:
#         x_coord.append(float(row[1]))  
#         y_coord.append(float(row[2]))  

# # Convert to numpy arrays
# x_array = np.array(x_coord)
# y_array = np.array(y_coord)


# def func(w):
#     meanSquareError = 0
#     for x,y  in zip(x_array,y_array):
#              meanSquareError += ((w)*x**2 - y)**2
#     meanSquareError = meanSquareError/len(x_array)
#     return meanSquareError

# print(func(-0.6))

# ----------------------------------------
                # Part c
# ----------------------------------------


# dataFile = "data.csv"

# # Empty lists to store coordinates
# x_coord = []
# y_coord = []

# # Reader for file data.csv
# with open(dataFile, mode='r', newline='') as file:
   
#     reader = csv.reader(file)
#     next(reader)

#     # Store data into arrays
#     for row in reader:
#         x_coord.append(float(row[1]))  
#         y_coord.append(float(row[2]))  

# # Convert to numpy arrays
# x_array = np.array(x_coord)
# y_array = np.array(y_coord)


# def func(w):
#     meanSquareError = 0
#     for x,y  in zip(x_array,y_array):
#              meanSquareError += ((w)*x**2 - y)**2
#     meanSquareError = meanSquareError/len(x_array)
#     return meanSquareError


# optimized = scipy.optimize.minimize(func, -0.6).x

# coeff = [optimized[0],0,0]

# poly = np.poly1d(coeff)

# yfit = poly(x_array)

# plt.plot(x_array, y_array, 'o', color='blue', markersize=2)

# plt.plot(x_array, yfit, color='red')

# for i in range(len(x_array)):

#     plt.plot([x_array[i], x_array[i]], [y_array[i], yfit[i]])


# plt.xlabel('x')

# plt.ylabel('y')

# plt.grid(True)

# plt.show()


# ----------------------------------------
                # Part d
# ----------------------------------------

# dataFile = "data.csv"

# # Empty lists to store coordinates
# x_coord = []
# y_coord = []

# # Reader for file data.csv
# with open(dataFile, mode='r', newline='') as file:
   
#     reader = csv.reader(file)
#     next(reader)

#     # Store data into arrays
#     for row in reader:
#         x_coord.append(float(row[1]))  
#         y_coord.append(float(row[2]))  

# # Convert to numpy arrays
# x_array = np.array(x_coord)
# y_array = np.array(y_coord)

# plt.plot(np.square(x_array), y_array, 'o', markersize=2)

# plt.show()


# ----------------------------------------
                # Part e
# ----------------------------------------

# dataFile = "data.csv"

# # Empty lists to store coordinates
# x_coord = []
# y_coord = []

# # Reader for file data.csv
# with open(dataFile, mode='r', newline='') as file:
   
#     reader = csv.reader(file)
#     next(reader)

#     # Store data into arrays
#     for row in reader:
#         x_coord.append(float(row[1]))  
#         y_coord.append(float(row[2]))  

# # Convert to numpy arrays
# x_array = np.array(x_coord)
# y_array = np.array(y_coord)

# featureData = np.square(x_array)[:,None]

# print(np.linalg.lstsq(featureData,y_array)[0])

# ----------------------------------------
                # Part f-c
# ----------------------------------------


dataFile = "data.csv"

# Empty lists to store coordinates
x_coord = []
y_coord = []

# Reader for file data.csv
with open(dataFile, mode='r', newline='') as file:
   
    reader = csv.reader(file)
    next(reader)

    # Store data into arrays
    for row in reader:
        x_coord.append(float(row[1]))  
        y_coord.append(float(row[2]))  

# Convert to numpy arrays
x_array = np.array(x_coord)
y_array = np.array(y_coord)


def func(w):
    meanSquareError = 0
    for x,y  in zip(x_array,y_array):
             meanSquareError += ((w**5)*(x**2) - y)**2
    meanSquareError = meanSquareError/len(x_array)
    return meanSquareError


optimized = scipy.optimize.minimize(func, -0.6).x

coeff = [optimized[0],0,0]

poly = np.poly1d(coeff)

yfit = poly(x_array)

plt.plot(x_array, y_array, 'o', color='blue', markersize=2)

plt.plot(x_array, yfit, color='red')

for i in range(len(x_array)):

    plt.plot([x_array[i], x_array[i]], [y_array[i], yfit[i]])


plt.xlabel('x')

plt.ylabel('y')

plt.grid(True)

plt.show()


# ----------------------------------------
                # Part f-c-d-e
# ----------------------------------------

newX = np.power(x_array, 2/5)

y_array += 90

newY = np.power(y_array, 1/5)


plt.plot(newX, newY, 'o', markersize=2)

plt.show()


y_array += 90

newY = np.power(y_array, 1/5)

featureData = np.power(x_array, 2/5)[:,None]

w, x, y, z = np.linalg.lstsq(featureData, newY, rcond = None)

print("new feature data:",  w[0])

print("optimal w:", optimized[0])