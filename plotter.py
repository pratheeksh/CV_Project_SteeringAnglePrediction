import pickle
import sys

import matplotlib.pyplot as plt

plots = []
evaluation_data = pickle.load(open("eval.p", "rb"))

x = range(5614)
i = 1
k = len(sys.argv[1:])

for filename in sys.argv[1:]:
    original_angles = []

    ax = plt.subplot(k, 1, i)
    i += 1
    dummy = []
    for row in open(filename, 'rb'):
        if "Filename" in row:
            continue

        frame_id, angle = row.split(',')
        original_angles.append(evaluation_data[frame_id])
        dummy.append(float(angle))
    ax.plot(x, dummy, label="Predicted Angles")
    ax.scatter(x, original_angles, label="Original Angles", color='green', s=1)

    ax.legend()
plt.show()
