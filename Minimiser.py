# NumRep Unit 7a
# Simon McLaren

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Minimiser class for a function of any number of parameters
class Minimiser():

    # Constructor for minimiser of n parameters; the format of the input parameters is as follows
    # n-dim array/list, integer, n-dim array/list
    def __init__(self, convergencelimits, maxiterations, stepsizes):
        self.convergencelimits = np.array(convergencelimits)
        self.maxiterations = maxiterations
        self.stepsizes = np.array(stepsizes)

    # Set new convergence limits
    def setConvergenceLimit(self, conv):
        self.convergencelimits = conv

    # Set new maximum iterations value
    def setMaxIterations(self, max):
        self.maxiterations = max

    # Set new step sizes
    def setStepSize(self, stepsizes):
        self.stepsizes = stepsizes

    # Minimises for the current minimiser parameters
    def minimise(self, function, startparameters): # Function class (n variables), n-dim array
        self.iterations = 0 # Internal iterations counter
        parameters = np.array(startparameters)
        # Create empty array to hold comparison function values for each parameter
        functionvalues = np.empty(len(parameters))
        # Loop until maxiterations is exceeded and convergence values are reached
        while (self.isFinished() == False):
            # Increment all parameters by their stepsizes
            parameters += self.stepsizes
            # Set reference function value
            valueoffunction = function.evaluate(parameters)
            # For each parameter:
            # Compare the adjusted function value to the reference one and adjust the stepsize if necessary
            # The adjusted function value is the reference function value with the current parameter
            # incremented by its stepsize
            for i in range(len(parameters)):
                parameterscopy = parameters
                parameterscopy[i] += self.stepsizes[i]
                functionvalues[i] = function.evaluate(parameterscopy)
                # If adjusted function value is greater than the reference one reverse the direction
                # of the steps and half the step size
                if functionvalues[i] > valueoffunction:
                    self.stepsizes[i] /= -2.
            self.iterations += 1
        self.iterations = 0
        return parameters

    # Check if the minimiser has reached convergence or iteration limits
    def isFinished(self):
        if self.iterations < self.maxiterations:
            return False
        for i in range(len(self.stepsizes)):
            if self.stepsizes[i] > self.convergencelimits[i]:
                return False
        return True

# Class for chiqsquared function
class chisquaredfunction():

    # Constructor
    # Inputs: y and x lists of measured values and the corresponding errors
    def __init__(self, ymeas, xmeas, errors):
        self.ymeas = ymeas
        self.xmeas = xmeas
        self.errors = errors

    # Evaluate function at user-defined parameter values
    def evaluate(self, params): # params = [m, c]
        eval = 0.
        m = params[0]
        c = params[1]
        for i in range(len(self.ymeas)):
            eval += ((m * self.xmeas[i] + c - self.ymeas[i])/(self.errors[i]))**2
        return eval

    # Returns arrays for plotting the value of the function for fixed m as c is varied
    def plotFixedm(self, c_range, num_points, fixedparams): # Width of c values to plot, number of points to plot
        m = fixedparams[0]
        c = fixedparams[1]
        cvals = np.linspace(c-c_range, c+c_range, num_points)
        funcvals = np.empty(num_points)
        for i in range(len(funcvals)):
            funcvals[i] = self.evaluate([m, cvals[i]])
        return cvals, funcvals

    # Returns arrays for plotting the value of the function for fixed c as m is varied
    def plotFixedc(self, m_range, num_points, fixedparams): # Width of m values to plot, number of points to plot
        m = fixedparams[0]
        c = fixedparams[1]
        mvals = np.linspace(m-m_range, m+m_range, num_points)
        funcvals = np.empty(num_points)
        for i in range(len(funcvals)):
            funcvals[i] = self.evaluate([mvals[i], c])
        return mvals, funcvals

if __name__ == '__main__':

    # Read in data to fit
    inputfilename = 'testData.txt'
    xvals = []
    yvals = []
    errors = []
    testdata = open(inputfilename, 'r')
    for line in testdata:
        tokens = line.split()
        xvals.append(float(tokens[0]))
        yvals.append(float(tokens[1]))
        errors.append(float(tokens[2]))

    # Define chisquared function
    function = chisquaredfunction(yvals, xvals, errors)

    # Define grid search upper and lower limits for m and c
    lowerlimits = [-2., -2.] # [m, c]
    upperlimits = [2., 2.] # [m, c]

    # Find an approximate minimum using a grid search to set start parameters
    gridsize = 10
    mspace = np.linspace(lowerlimits[0], upperlimits[0], gridsize)
    cspace = np.linspace(lowerlimits[1], upperlimits[1], gridsize)
    grid = np.empty([gridsize, gridsize])
    for i in range(gridsize):
        for j in range(gridsize):
            grid[i][j] = function.evaluate([mspace[i], cspace[j]])
    gridminimum = np.unravel_index(grid.argmin(), grid.shape)
    startparam = [mspace[gridminimum[0]], cspace[gridminimum[1]]]

    # Initial stepsizes for m and c
    stepsizes = [0.01, 0.01] # [m, c]
    maxiterations = 50000
    convergence_limits = [0.000001, 0.000001] # [m, c]

    # Set up minimiser and find the minimums of m and c
    minim = Minimiser(convergence_limits, maxiterations, stepsizes)
    minimised_Parameters = minim.minimise(function, startparam)
    minimum_chisquared = function.evaluate(minimised_Parameters)
    print "Minimum m = %f.  Minimum c = %f.  Minimum chisquared = %f" \
          %(minimised_Parameters[0], minimised_Parameters[1], minimum_chisquared)

    # Plot chisquared around the minimum m and c
    plot_width = 0.1
    plot_points = 5000
    c_vals, fixedm_funcvals = function.plotFixedm(plot_width, plot_points, minimised_Parameters)
    m_vals, fixedc_funcvals = function.plotFixedc(plot_width, plot_points, minimised_Parameters)

    plt.plot(c_vals, fixedm_funcvals)
    plt.title('Chisquared against c at minimum m')
    plt.xlabel('c value')
    plt.ylabel('Chisquared')
    plt.savefig('chifixedm.png')
    plt.clf()

    plt.plot(m_vals, fixedc_funcvals)
    plt.title('Chisquared against m at minimum c')
    plt.xlabel('m value')
    plt.ylabel('Chisquared')
    plt.savefig('chifixedc.png')
    plt.clf()


    # Find the errors on m and c
    m_error = 0.
    c_error = 0.

    for i in range(len(c_vals)):
        delta_chisquared = fixedm_funcvals[i] - minimum_chisquared
        if delta_chisquared <= 1:
            c_error = abs(c_vals[i] - minimised_Parameters[1])

    for i in range(len(m_vals)):
        delta_chisquared = fixedc_funcvals[i] - minimum_chisquared
        if delta_chisquared <= 1:
            m_error = abs(m_vals[i] - minimised_Parameters[0])

    print "m error = %f.  c error = %f." %(m_error, c_error)

    res = minimize(function.evaluate, startparam)
    print res
