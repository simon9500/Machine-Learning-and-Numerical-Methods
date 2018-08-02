import math
import numpy as np
import matplotlib.pyplot as plt

# NumRep Unit 5 Root-finding exercise
# Simon McLaren

class rootFinder:

    def __init__(self, function, function1stderivative, range): # Lambda, 2D array/list
        self.function = function
        self.function1stderivative = function1stderivative
        self.range = range

    # Divide range into N subregions and determine which regions the function changes sign in and returns them
    def divideRange(self, numDivisions): # Integer
        subRegionswithRoots = []
        subRegionWidth = (self.range[1]-self.range[0])/numDivisions
        subRegionminx = self.range[0]
        subRegionmaxx = self.range[0] + subRegionWidth
        for i in range(numDivisions):
            if self.function(subRegionminx) < 0. and self.function(subRegionmaxx) > 0. \
                or self.function(subRegionminx) > 0. and self.function(subRegionmaxx) < 0. \
                or self.function(subRegionmaxx) == 0. or \
                self.function(subRegionminx) == 0.:
                subRegionswithRoots.append([subRegionminx, subRegionmaxx])
            subRegionminx += subRegionWidth
            subRegionmaxx += subRegionWidth
        return np.array(subRegionswithRoots)

    # Method for finding a root using the bisection method
    # Returns the value of the root to a desired accuracy
    def bisectionMethod(self, subRegion, desiredAccuracy): # 2 element array/list, float
        subRegionminx = subRegion[0]
        subRegionmaxx = subRegion[1]
        realAccuracy = abs(subRegionmaxx-subRegionminx)
        # Run until the subregion is below a user-defined accuracy
        while realAccuracy > desiredAccuracy:
            # Calculate function values centrally and at the boundaries
            centreValue = self.function((subRegionmaxx+subRegionminx)/2.)
            minValue = self.function(subRegionminx)
            maxValue = self.function(subRegionmaxx)
            if centreValue > 0. and minValue > 0. or centreValue < 0. and minValue < 0.:
                subRegionminx = (subRegionmaxx+subRegionminx)/2. # Set minimum boundary to central value
            elif centreValue > 0. and maxValue > 0. or centreValue < 0. and maxValue < 0.:
                subRegionmaxx = (subRegionmaxx+subRegionminx)/2. # Set maximum boundary to central value
            elif centreValue == 0.:
                return (subRegionmaxx+subRegionminx)/2.
            elif minValue == 0.:
                return subRegionminx
            elif maxValue == 0.:
                return subRegionmaxx
            realAccuracy = abs(subRegionmaxx - subRegionminx)
        return (subRegionmaxx+subRegionminx)/2.

    # Method for finding a root using the Newton-Raphson method
    # Returns the value of the root to a desired accuracy
    def newtonRaphsonMethod(self, subRegion, desiredAccuracy):
        # Check subregion boundaries for the root
        if self.function(subRegion[0]) == 0.:
            return subRegion[0]
        elif self.function(subRegion[1]) == 0.:
            return subRegion[1]
        rootGuess = (subRegion[0]+subRegion[1])/2. # First guess of root is middle of subregion
        rootFound = False
        # Loop until root is found to the desired accuracy
        while rootFound is False:
            d = -self.function(rootGuess)/self.function1stderivative(rootGuess)
            rootGuess += d
            if abs(d) < desiredAccuracy:
                rootFound = True
        return rootGuess

    # Method for finding a root using the Newton-Raphson method
    # Returns the value of the root to a desired accuracy
    def secantMethod(self, subRegion, desiredAccuracy):
        return 0

    # Plots the function over its full range using N points
    def plotFunction(self, Npoints): # Integer
        functionValues = []
        xValues = []
        pointSpacing = (self.range[1]-self.range[0])/Npoints
        currentxValue = self.range[0]
        for i in range(Npoints):
            xValues.append(currentxValue)
            functionValues.append(self.function(currentxValue))
            currentxValue += pointSpacing
        plt.plot(xValues, functionValues)
        plt.show()

    # Finds and returns all the roots in the subregions with roots
    def findAllRoots(self, subRegionsWithRoots, methodNumber, desiredAccuracy): # list of 2 element arrays,
                                                                                # integer, float
        rootList = []
        for subRegion in subRegionsWithRoots:
            if methodNumber == 1:
                rootList.append(self.bisectionMethod(subRegion, desiredAccuracy))
            if methodNumber == 2.:
                rootList.append(self.newtonRaphsonMethod(subRegion, desiredAccuracy))
        return np.array(rootList)

if __name__ =="__main__":
    function1 = lambda x: float(10.2 - 7.4 * x - 2.1 * x ** 2 + x ** 3)
    function1der = lambda x: float(7.4 - 4.2 * x + 3 * x ** 2)
    function2 = lambda x: float(math.exp(x) - 2.)
    function2der = lambda x: float(math.exp(x))
    function3 = lambda x: float(math.cos(x) * math.sin(3 * x))
    function3der = lambda x: float(math.cos(2 * x) + 2 * math.cos(4 * x))
    f1rootfinder = rootFinder(function1, function1der, [-5.,5.])
    rootSubRanges = f1rootfinder.divideRange(5000)
    rootList = f1rootfinder.findAllRoots(rootSubRanges, 2, 0.001)
    f1rootfinder.plotFunction(1000)

