print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
data = np.loadtxt("features-google.txt")

# Use only one feature
#X1 = data[0][0]
cols = []
a=1
for row in data:
    cols.append(row[0])

print(max(cols))

plt.plot(cols, color='blue',linewidth=1)

plt.show()