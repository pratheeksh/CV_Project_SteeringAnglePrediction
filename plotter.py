import matplotlib.pyplot as plt
import csv
import sys

plots = []
x = range(5614)
for filename in sys.argv[1:]:
    dummy = []
    for row in open(filename,'rb'):
        dummy.append(float(row))
    plt.plot(x,dummy,label=filename.split('.')[0])
plt.legend()
plt.show()
