# Report 1
# Image correction using DFT
# Numerical Recipes
# Simon McLaren

import numpy as np
from numpy import fft
from scipy import misc
import matplotlib.pyplot as plt

# Take in image file (pgm format) and return as an np array
def readImage():
    # Get the filename as string
    fn = str(raw_input("File : "))
    # Read file as np array
    im = misc.imread(fn)
    return im

# Cross correlate 2 1D numpy arrays
def crossCorrelate(line1, line2):
    # Take the FT of both lines
    fftline1 = fft.fft(line1)
    fftline2 = fft.fft(line2)
    fftline2conjugate = np.conj(fftline2)  # Take the complex conjugate of the 2nd line
    product = fftline1 * fftline2conjugate  # Multiply it with the FT of the first line
    return fft.ifft(product)  # Take the inverse FT of the product and return it

# Gaussian function
def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

# Scale the cross correlation product by a Gaussian centred on index 0
def scaleWithGaussian(ccproduct, sigma):
    gaussianKernel = np.empty(len(ccproduct))
    ccproductcopy = np.copy(ccproduct)
    # Define centre of Gaussian as midpoint of the cc array
    mu = len(ccproduct)/2
    # Calculate gaussian array
    for i in range(len(gaussianKernel)):
        gaussianKernel[i] = gaussian(i, mu, sigma)
    # Rotate the 0 index point of the cc array to the midpoint of the gaussian
    ccproductcopy = np.roll(ccproductcopy, mu)
    # Scale the cc array with the gaussian by multiplying the 2 cc and gaussian arrays
    ccproductcopy = ccproductcopy * gaussianKernel
    # Rotate the cc back so the 0 index point is back in its original position
    ccproductcopy = np.roll(ccproductcopy, - mu)
    return ccproductcopy

# Correct the distorted input image by shifting the rows
# Returns the corrected image and the array of maxindex values for debugging
def correctImage(inputimage, numoflines, linelength, sigma, indexshiftcutoff):
    correctedimage = np.copy(inputimage) # Create copy of corrected image
    maxindexlist = np.empty(numoflines - 1, dtype = np.int)
    # Assumption: the first line of the input image is correct; all other lines
    # are relatively shifted
    for i in range(numoflines - 1):
        # Cross correlate adjacent lines
        ifftproduct = crossCorrelate(correctedimage[i], correctedimage[i + 1])

        # Take magnitude of each cross correlation value
        ifftproduct = np.abs(ifftproduct)

        # Scale cross correlation array with a Gaussian centred on index 0
        ifftproduct = scaleWithGaussian(ifftproduct, sigma)

        maxindex = np.argmax(ifftproduct)  # Find the array index of the maximum value
        # Shift maxindex values to range (-linelength/2 < maxindex =< linelength/2
        if maxindex > linelength/2:
            maxindex -= linelength
        # Filter out large shifts relative to the row length
        if abs(maxindex) > indexshiftcutoff:
            maxindex = 0
        maxindexlist[i] = maxindex
        # Shift line by the location of the peak
        correctedimage[i + 1] = np.roll(correctedimage[i + 1], maxindex)
    return correctedimage, maxindexlist

# Main Program
def main():
    inputimage = readImage() # Read image in as array
    numoflines = inputimage.shape[0]  # Number of rows
    linelength = inputimage.shape[1]  # Length of row in pixels

    # Parameters for the correction process
    # sigma is the sigma value for the gaussin kernel
    # that scales the cross correlation array
    # indexshiftcutoff is the maximum allowed value for the shift
    sigma = float(linelength / 2)
    indexshiftcutoff = linelength / 20

    # Correct image
    correctedimage, maxindexlist = correctImage(inputimage, numoflines, linelength
                                                        , sigma, indexshiftcutoff)

    # Save input image
    plt.imshow(inputimage, cmap=plt.cm.gray)
    plt.title('Input Image')
    plt.savefig('inputimage.png')
    plt.clf()

    # Save corrected image
    plt.imshow(correctedimage, cmap=plt.cm.gray)
    plt.title('Corrected Image')
    plt.savefig('correctedimage.png')
    plt.clf()

    # Save plot of shifts
    plt.plot(np.arange(len(maxindexlist)), maxindexlist)
    plt.ylabel('Shift')
    plt.title('Max Values')
    plt.savefig('maxvalues.png')

main() # Run program
