import pandas as pd
import matplotlib.pyplot as plt
import sys

from algorithms.estimateBaseline import estimateBaseline

df = pd.read_csv(r"./data/example_spectra_LIBS.csv", delimiter = ";")

bl = estimateBaseline(list(df.wavelength), list(df.intensity), 
                        window = 251, notch = True, fill_between = [522, 542])

df.plot("wavelength","intensity")
plt.plot(list(df.wavelength),bl)
plt.show()