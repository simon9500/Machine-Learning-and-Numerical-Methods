import numpy as np

# Matrix class for Numerical Recipes
class Matrix:

    # Matrix entries are set by a 2D numpy array
    # The length of each column is assumed to be the same
    def __init__(self, coeff):
        self.coeff = coeff
        
    # Returns the number of rows in the matrix
    def getRows(self):
        return self.coeff.shape[1]

    # Returns the number of columns in the matrix
    def getColumns(self):
        return self.coeff.shape[0]

    # Sets the element in the ith column, and jth row to val
    def setElement(self, i, j, val):
        self.coeff[i][j] = val

    # Returns the element in the ith column and the jth row
    def getElement(self, i, j):
        return self.coeff[i][j]

    # Adds two matrices of the same dimensions and returns the result
    # If the matrices do not have the same dimensions then the None (null) result is returned
    def addMatrices(self, m):
        if m.getRows() == self.getRows() and m.getColumns() == self.getColumns():
            return self.coeff + m.coeff
        else:
            return None
    
    # Multiplies two matrices if the number of columns of the 'instance' matrix are the same as the number of columns of m
    # Otherwise the null result is returned
    def multiplyMatrices(self, m):
        if self.getColumns() == m.getRows():
            resultantMatrix = np.empty([m.getColumns(), self.getRows()])
            for i in range(self.getRows()):
                for j in range(m.getColumns()):
                    currentSum = 0
                    for k in range(m.getRows()):
                        currentSum += self.coeff[k][i]*m.coeff[j][k]
                    resultantMatrix[j][i] = currentSum
            return resultantMatrix
        else:
            return None
                
# Testing
if __name__ == '__main__':
    coeff1 = np.array([[2],[3],[4]])
    coeff2 = np.array([[1,1,1]])

    M1 = Matrix(coeff1)
    M2 = Matrix(coeff2)

    print M1.getColumns()
    print M1.getRows()
    print M2.getColumns()
    print M2.getRows()

    print M1.multiplyMatrices(M2)
    print M1.addMatrices(M1)
