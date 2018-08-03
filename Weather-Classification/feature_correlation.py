import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

## Main program
def main():
    # Load the weather data you created by FeatureExtraction.py
    weather = pickle.load(open('data/mldata.p'))

    # Define dataframe for weather data
    df = pd.DataFrame(weather.data, columns = weather.getFeatures())

    f, ax = plt.subplots(figsize=(10, 8)) # Define figure

    # Create correlation matrix heatmap for the features in the weather data
    sns.heatmap(df.corr(), mask=np.zeros_like(df.corr(), dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, annot=True, vmin=-1.)

    f.savefig('correlation_matrix.png')

main() # Run program

