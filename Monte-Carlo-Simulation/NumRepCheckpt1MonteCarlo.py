import matplotlib.pylab as pl
import random
import math
import numpy as np

"""
NumRep Checkpoint 1: MonteCarlo simulations of Muon decay
Simon McLaren
"""

# Returns a value for the muon lifetime probability density at a specified time
def evaluateMuonDistribution(time, tau):
    return float((1/tau)*math.exp(-(time/tau)))

muonLifetime = 2.2 # in microseconds
numberofsimulateddecays = 1000
numberofsimulatedlifetimes = 500
totalMuonLifetimes = []
# Parameters for sampling a random number from the distribution
fmax = evaluateMuonDistribution(0, muonLifetime) # Max value of e^-x function is at 0 for the range 0 to inf
mint = 0.
maxt = 22.

# Simulate 500 muon lifetimes
for j in range(numberofsimulatedlifetimes):
    listOfDecayTimes = []
    # Simulate 1000 muon decay times
    for i in range(numberofsimulateddecays):
        # Draw a random time from the distribution using the box method
        y2 = 10.
        y1 = 1.
        while y2 > y1:
            randomt = mint + (maxt - mint) * random.random()
            y1 = evaluateMuonDistribution(randomt, muonLifetime)
            y2 = fmax * random.random()
        listOfDecayTimes.append(randomt)

    estimatedMuonLifetime = (sum(listOfDecayTimes)/len(listOfDecayTimes))
    totalMuonLifetimes.append(estimatedMuonLifetime)

# Plot histogram
pl.hist(totalMuonLifetimes[:], bins = 100, range = [1,3])
pl.xlabel("Estimated Muon Lifetime (microseconds)")
pl.title("Hisogram of Muon Lifetimes")
pl.savefig("muonlifetimehistogram.png")

totalMuonLifetimes = np.array(totalMuonLifetimes)

# Print results
simulationMuonLifetime = np.mean(totalMuonLifetimes) # Estimated muon lifetime from the simulation
standardDeviationMuonLifetime = np.std(totalMuonLifetimes) # Standard deviation of the data
print "The estimated muon lifetime is %f microseconds." %(simulationMuonLifetime)
print "The standard deviation of the 500 muon lifetimes is %f." %(standardDeviationMuonLifetime)
print "The standard error of the mean is %f." %(standardDeviationMuonLifetime/math.sqrt(numberofsimulatedlifetimes))
