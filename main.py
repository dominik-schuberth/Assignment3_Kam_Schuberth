print("assignment3")

import numpy as np
import pandas as pd


pd.set_option("display.precision", 12)
#https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
daten2 = pd.read_csv('input_form.csv',delimiter=";", header = None)
print(daten2)

#import matplotlib.pyplot as plt
#plt.figure(0)
#plt.scatter(daten2, daten2)


#funktion um distance zu berechnen
def manhattan_distance(x1,x2):
    return np.sqrt(np.sum(abs(x1-x2)))

