import math
import numpy as np
import matplotlib.pyplot as plt

# NumRep Unit 5 ODE-solving exercise
# Simon McLaren

# Function that uses the Euler method to solve a 1st order ODE over a given range
# Returns the x, y solutions as arrays
def eulerMethod(diffeq, inity, initx, finalx, numpoints): # Lambda (2 arguments), float, float, float, integer
    xvalues = np.linspace(initx, finalx, numpoints)
    yvalues = np.empty(numpoints)
    yvalues[0] = inity
    xvaluespacing = (finalx-initx) / numpoints
    for i in range(1, numpoints):
        yvalues[i] = yvalues[i - 1] + xvaluespacing * diffeq(xvalues[i - 1], yvalues[i - 1])
    return xvalues, yvalues

# 2nd order Runge-Kutta method to solve a 1st order ODE over a given range
# Returns the x, y solutions as arrays
def RungeKuttaMethod2ndorder(diffeq, inity, initx, finalx, numpoints):
    xvalues = np.linspace(initx, finalx, numpoints)
    yvalues = np.empty(numpoints)
    yvalues[0] = inity
    xvaluespacing = (finalx - initx) / numpoints
    for i in range(1, numpoints):
        xmidpoint = xvalues[i - 1] + xvaluespacing / 2
        ymidpoint = yvalues[i - 1] + xvaluespacing * diffeq(xvalues[i - 1], yvalues[i - 1]) / 2
        yvalues[i] = yvalues[i - 1] + xvaluespacing * diffeq(xmidpoint, ymidpoint)
    return xvalues, yvalues

# 4nd order Runge-Kutta method to solve a 1st order ODE over a given range
# Returns the x, y solutions as arrays
def RungeKuttaMethod4ndorder(diffeq, inity, initx, finalx, numpoints):
    xvalues = np.linspace(initx, finalx, numpoints)
    yvalues = np.empty(numpoints)
    yvalues[0] = inity
    xvaluespacing = (finalx - initx) / numpoints
    for i in range(1, numpoints):
        k1 = diffeq(xvalues[i - 1], yvalues[i - 1]) # slope at beginning
        k2 = diffeq(xvalues[i - 1] + xvaluespacing / 2, yvalues[i - 1] + xvaluespacing * k1 / 2) # slope at midpoint
        k3 = diffeq(xvalues[i - 1] + xvaluespacing / 2,
                    yvalues[i - 1] + xvaluespacing * k2 / 2) # another slope at midpoint
        k4 = diffeq(xvalues[i - 1] + xvaluespacing,
                    yvalues[i - 1] + xvaluespacing * k3) # slope at endpoint
        yvalues[i] = yvalues[i - 1] + xvaluespacing * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)
    return xvalues, yvalues

# Testing
if __name__ =="__main__":
    initx = 0.
    inity = 1.
    finalx = 10.
    numpoints = 100
    diffeq1 = lambda x, y: 1 - 6 * x + 3 * x ** 2
    diffeq2 = lambda x, y: y
    diffeq3 = lambda x, y: math.cos(x)
    xvalues, yvalues = RungeKuttaMethod4ndorder(diffeq1, inity, initx, finalx, numpoints)
    plt.plot(xvalues, yvalues)
    plt.show()


