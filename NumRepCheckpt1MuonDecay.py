import matplotlib.pylab as pl
import random
import math
import numpy as np

"""
NumRep Checkpoint 1: simulating 1000 muon decays
Simon McLaren
"""

# Open output file for writing
muonDecayTimesOutputFile = open("muondecaytimes.out", "w")

# Returns a value for the muon lifetime probability density at a specified time
def evaluateMuonDistribution(time, tau):
    return float((1/tau)*math.exp(-(time/tau)))

muonLifetime = 2.2 # in microseconds
numberofdecays = 1000
simulatedDecayTimes = []
# Parameters for sampling a random number from the distribution
fmax = evaluateMuonDistribution(0, muonLifetime)
mint = 0.
maxt = 22.

# Simulate 1000 muon lifetimes
for i in range(numberofdecays):
    # Draw a random time from the distribution using the box method
    y2 = 10.
    y1 = 1.
    while y2 > y1:
        randomt = mint + (maxt - mint) * random.random()
        y1 = evaluateMuonDistribution(randomt, muonLifetime)
        y2 = fmax * random.random()
    simulatedDecayTimes.append(randomt)
    muonDecayTimesOutputFile.write("{0:f}\n".format(randomt))

simulatedDecayTimes = np.array(simulatedDecayTimes)
estimatedMuonLifetime = np.mean(simulatedDecayTimes)

# Produce histogram
pl.hist(simulatedDecayTimes[:], bins = 100, range = [mint,maxt])
pl.savefig("muondecaytimeshistogram.png")

# Print results
print "The estimated muon lifetime is %f microseconds." %(estimatedMuonLifetime)
print "The standard deviation is %f." %(np.std(simulatedDecayTimes))

muonDecayTimesOutputFile.close()

