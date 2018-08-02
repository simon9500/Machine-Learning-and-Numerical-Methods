import matplotlib.pylab as pl
import random
import math
import numpy as np

# NumRep Checkpoint1 class for muon decay simulations
# Simon McLaren

class muonPDF:

    def __init__(self, muonLifetime, mint, maxt): # Float, float, float
        self.tau  = float(muonLifetime)
        self.mint = float(mint)
        self.maxt = float(maxt)

    # Returns the value of the PDF at a particular time
    def evaluatePDF(self, time):
        return float((1 / self.tau) * math.exp(-(time / self.tau)))

    # Draws a random decay time from the PDF using the box method
    def drawRandomNumBoxMethod(self):
        fmax = self.evaluatePDF(0.)
        y2 = 10.
        y1 = 1.
        while y2 > y1:
            randomt = self.mint + (self.maxt - self.mint) * random.random()
            y1 = self.evaluatePDF(randomt)
            y2 = fmax * random.random()
        return randomt

    # Simulates N decays using the PDF, returning the list of decay times and the mean of the decay times
    def simulateNDecays(self, numOfMuonDecays): # Integer
        decayTimes = np.empty(numOfMuonDecays)
        for i in range(numOfMuonDecays):
            decayTimes[i] = self.drawRandomNumBoxMethod()
        return decayTimes, np.mean(decayTimes)

    # Writes an array of decay times to an output file
    def writeDecayTimesToFile(self, outputFileName, decayTimes): # String, integer
        muonDecayTimesOutputFile = open(outputFileName, "w")
        for decayTime in decayTimes:
            muonDecayTimesOutputFile.write("{0:f}\n".format(decayTime))
        muonDecayTimesOutputFile.close()

    # Produces and saves a histogram of given data
    def saveHistogram(self, data, numBins, outputFileName, xlabel, title): # Array/list, integer, string, string, string
        pl.hist(data[:], bins=numBins)
        pl.xlabel(xlabel)
        pl.title(title)
        pl.savefig(outputFileName)

    # Simulate N experiments with M decay times and return the N muon lifetimes, their mean and standard deviation
    def simulateNExperiments(self, numOfExperiments, numOfMuonDecays):
        muonLifetimes = np.empty(numOfExperiments)
        for i in range(numOfExperiments):
            currentDecayTimesList, muonLifetimes[i] = self.simulateNDecays(numOfMuonDecays)
        return muonLifetimes, np.mean(muonLifetimes), np.std(muonLifetimes)

# Testing
if __name__ == "__main__":
    myMuonDistribution = muonPDF(2.2, 0., 15.)
    # Simulate 500 experiments with 1000 muon decays each
    muonLifetimes, meanLifetime, stdLifetime = myMuonDistribution.simulateNExperiments(500, 1000)
    print meanLifetime, stdLifetime
    myMuonDistribution.saveHistogram(muonLifetimes, 100, "muonlifetimehistogram.png",
                                     "Estimated Muon Lifetime (microseconds)",
                                     "Hisogram of Muon Lifetimes")






