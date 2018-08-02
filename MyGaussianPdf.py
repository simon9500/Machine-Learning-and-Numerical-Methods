import numpy as np
import math
import matplotlib.pylab as pl
import random


"""
NumRep Exercises 1 and 2
Simon McLaren
"""

# Class for a gaussian function
class MyGaussianPdf:

    # Constructor
    def __init__(self, mean, width):
        self.mean = mean
        self.width = width

    # Return Gaussian width
    def getWidth(self):
        return self.width

    # Return Gaussian mean
    def getMean(self):
        return self.mean

    # Return a random value of x with Gaussian distribution
    def next(self, minx, maxx):
        fmax = 1.
        y2 = 10.
        y1 = 1.
        while y2>y1:
            randomx = minx + (maxx-minx)*random.random()
            y1 = self.evaluate(randomx)
            y2 = fmax*random.random()
        return randomx

    # Evaluate the Gaussian at point x
    def evaluate(self, x):
        return math.exp(-((x-self.mean)**2)/2*self.width**2)

    # Calculate the integral of the gaussian between -inf and inf
    def integralAnalytic(self):
        return self.width*math.sqrt(2*math.pi)

    # Numerical integration by the 0th order rectangle method
    def integralNumericRectangle0thOrder(self, numofrectangles):
        minx = -5*self.width
        maxx = 5*self.width
        rectanglewidth = (maxx-minx)/numofrectangles
        integral = 0.
        evalxpoint = minx + rectanglewidth*0.5 # Midpoint of rectangle
        for i in range(numofrectangles):
            integral += self.evaluate(evalxpoint) * rectanglewidth
            evalxpoint += rectanglewidth
        return integral

    # Numerical integration by the 1st order rectangle method
    def integralNumericRectangle1stOrder(self, numofrectangles):
        minx = -5 * self.width
        maxx = 5 * self.width
        rectanglewidth = (maxx - minx) / numofrectangles
        integral = 0.
        lowerxbound = minx
        higherxbound = minx + rectanglewidth
        for i in range(numofrectangles):
            integral += rectanglewidth * self.evaluate(lowerxbound) + \
                        0.5 * abs(self.evaluate(higherxbound) - self.evaluate(lowerxbound)) * rectanglewidth
            lowerxbound += rectanglewidth
            higherxbound += rectanglewidth
        return integral

if __name__ == "__main__":
    gaussianfunction = MyGaussianPdf(0.,1.)
    # gaussiansampledata = []
    # for i in range(10000):
    #     gaussiansampledata.append(gaussianfunction.next(-4.,4.))
    # pl.hist(gaussiansampledata[:], bins = 50, range = [-4.,4.])
    # pl.savefig("gaussianhist.png")
    # pl.show()
    # data = np.random.normal(0.,1.,10000)
    # pl.hist(data[:], bins = 50, range = [-4.,4.])
    # pl.savefig("numpygaussianhist.png")
    # pl.show()
    integralanalytic = float(gaussianfunction.integralAnalytic())
    integralnumeric = float(gaussianfunction.integralNumericRectangle1stOrder(10000))
    integraldifference = abs(integralanalytic-integralnumeric)
    percentdifference = float(100*integraldifference/integralanalytic)
    print "The analytic integral is %f.  The numeric integral is %f.  The percentage difference between them is %.5f." \
          % (integralanalytic, integralnumeric, percentdifference)

