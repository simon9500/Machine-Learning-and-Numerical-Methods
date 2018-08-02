# ======================================================
# Checkpoint 2  :  code supplied to students at start.
#
# Exact form of Charge Distribution


import math
import matplotlib.pyplot as pl
import numpy as np

# Plot the solutions to an ODE - one found by the RK4 method and one by Euler
def plotRK4Euler(xvaleuler, yvaleuler, xvalRK4, yvalRK4, eulercolor, RK4color, plottitle, filename):
    # Plot euler and RK4 points on canvas
    pl.plot(xvaleuler, yvaleuler, color=eulercolor, label='Euler')
    pl.plot(xvalRK4, yvalRK4, color=RK4color, label='RK4')
    pl.legend()
    pl.title(plottitle)
    pl.savefig(filename)
    pl.clf() # Clear canvas

# Plot the difference between the RK4 and euler solutions
def plotDifferenceRK4Euler(xval, yvaleuler, yvalRK4, linecolor, marker, plottitle, filename):
    # Plot difference between euler and RK4 on canvas
    # The length of is assumed to be the same (at the discretion of the user)
    pl.plot(xval, yvalRK4 - yvaleuler, color=linecolor, marker=marker)
    pl.title(plottitle)
    pl.savefig(filename)
    pl.clf()  # Clear canvas


#---------------------------------------
# Charge distribution at the PN junction
class ChargeDistribution:
    
    #..............................................
    # Methods for the user of this class
    
    # To evaluate the y-value of the charge for an input x-value
    def evaluate(self, x):
        if( x < self.x1): return 0
        if( x < self.x2): return self._shape( self.x1, self.x2, x)
        if( x < self.x3): return -self._shape( self.x3, self.x2, x)
        return 0.
    
    # To plot the function on the screen
    def show(self, title='', disp=True ):
        xvalues, yvalues = self._get()
        pl.plot( xvalues, yvalues )
        pl.title( title )
        if(disp):pl.show()

    # Solve the charge distribution for the electric field over the range [x0, x1]
    # using the initial conditions E(x0)=y0
    def getelectricfield(self, method, x0, y0, x1, numpoints):
        xvalues = np.linspace(x0, x1, numpoints) # Define equally spaced x values over the range in an array
        yvalues = np.empty(numpoints) # Define empty array for E values
        yvalues[0] = y0
        xvaluespacing = float(x1 - x0) / numpoints
        for i in range(1, numpoints):
            if method == 'RK4':
                k1 = self.evaluate(xvalues[i - 1])  # slope at beginning
                k2 = self.evaluate(xvalues[i - 1] + xvaluespacing / 2)  # slope at midpoint
                k3 = k2  # another slope at midpoint
                k4 = self.evaluate(xvalues[i - 1] + xvaluespacing)  # slope at endpoint
                yvalues[i] = yvalues[i - 1] + xvaluespacing * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)
            if method == 'euler':
                yvalues[i] = yvalues[i - 1] + xvaluespacing * self.evaluate(xvalues[i - 1])
        return xvalues, yvalues

    # Solve for the voltage using the electric field over the range [x0, x1]
    # using the initial conditions V(x0)=y0
    def getvoltage(self, method, x0, y0, x1, Eyvalues):
        numpoints = len(Eyvalues)
        xvalues = np.linspace(x0, x1, numpoints)
        yvalues = np.empty(numpoints)
        yvalues[0] = y0
        xvaluespacing = float(x1 - x0) / numpoints
        for i in range(1, numpoints):
            # Choose Runge-Kutta 4 method
            if method == 'RK4':
                k1 = - Eyvalues[i - 1]  # slope at beginning
                k2 = - (Eyvalues[i - 1] + Eyvalues[i]) / 2  # slope at midpoint
                k3 = k2  # another slope at midpoint
                k4 = - Eyvalues[i]  # slope at endpoint
                yvalues[i] = yvalues[i - 1] + xvaluespacing * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)
            # Choose Euler method
            if method == 'euler':
                yvalues[i] = yvalues[i - 1] + xvaluespacing * - Eyvalues[i - 1]
        return xvalues, yvalues
    
    #...........................................
    
    #constructor
    def __init__(self):
        self.x0 = -2.
        self.x1 = -1.
        self.x2 = 0.
        self.x3 = 1
        self.x4 = 2
        self.k = math.pi/(self.x3-self.x1)
    
    # pseudo internal methods
    def _shape(self, x0, x1, x):
        z = (x-x0)/(x1-x0)
        return (z**2)* (math.exp(1-z)-1.) / 0.18
    
    def _get( self, start=-2, end=2., n=1000 ):
        xvalues= []
        yvalues = []
        dx = (end-start)/n
        for i in range(n):
            xvalues.append(start+i*dx)
            yvalues.append(self.evaluate(start+i*dx))
        return xvalues, yvalues

# Testing
if __name__ == '__main__':

    # Define charge distribution
    chargeDistribution = ChargeDistribution()

    # Compute electric field and voltage using both methods
    numpoints = 50 # Number of points to solve for the ODE
    # Initial conditions for voltage
    Vx0 = -2
    Vy0 = 0
    # Initial conditions for electric field
    Ex0 = -2
    Ey0 = 0
    # Final x values for electric field and voltage
    Exf = 2
    Vxf = 2

    ExvalRK4, EyvalRK4 = chargeDistribution.getelectricfield('RK4', Ex0, Ey0, Exf, numpoints)
    Exvaleuler, Eyvaleuler = chargeDistribution.getelectricfield('euler', Ex0, Ey0, Exf, numpoints)
    VxvalRK4, VyvalRK4 = chargeDistribution.getvoltage('RK4', Vx0, Vy0, Vxf, EyvalRK4)
    Vxvaleuler, Vyvaleuler = chargeDistribution.getvoltage('euler', Vx0, Vy0, Vxf, Eyvaleuler)

    # Save plot of electric field found using RK4 and euler methods
    plotRK4Euler(Exvaleuler, Eyvaleuler, ExvalRK4, EyvalRK4, 'red', 'blue',
                 'Electric field against x', 'electricfield.png')

    # Save plot of voltage found using RK4 and euler methods
    plotRK4Euler(Vxvaleuler, Vyvaleuler, VxvalRK4, VyvalRK4, 'red', 'blue',
                 'Voltage against x', 'voltage.png')

    # Save plot of difference between electric field for RK4 and euler methods
    plotDifferenceRK4Euler(Exvaleuler, Eyvaleuler, EyvalRK4, 'red', '.',
                           'Difference between Electric field solutions using RK4/Euler', 'electricfieldcomparison.png')

    # Save plot of difference between voltage for RK4 and euler methods
    plotDifferenceRK4Euler(Vxvaleuler, Vyvaleuler, VyvalRK4, 'red', '.',
                           'Difference between Voltage solutions using RK4/Euler', 'voltagecomparison.png')





    


