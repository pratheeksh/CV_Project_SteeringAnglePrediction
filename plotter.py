import pickle
import sys

import matplotlib.pyplot as plt

plots = []
evaluation_data = pickle.load(open("eval.p", "rb"))

# x = range(5614)
i = 1
k = len(sys.argv[1:])



plt.suptitle('Plots for NVIDIA Network', fontsize=12)
for filename in sys.argv[1:]:
    original_angles = []

    ax = plt.subplot(k, 1, i)
    i += 1
    dummy = []
    new_angles = {}
    for row in open(filename, 'rb'):
        if "Filename" in row:
            continue
        if "Model" in row:
            ax.set_title(row.split(',')[1],  fontsize=10)
            continue
        frame_id, angle = row.split(',')
        # original_angles.append(evaluation_data[frame_id])
        new_angles[frame_id] = float(angle)
    key_list = evaluation_data.keys()
    key_list.sort()
    for frame_id in key_list:
        original_angles.append(evaluation_data[frame_id])
        dummy.append(new_angles[frame_id])



    x = range(len(dummy))
    ax.set_xlim([min(x) - 100, max(x) + 100])
    ax.plot(x, dummy, label="Predicted Angles")
    ax.plot(x, original_angles, label="Original Angles", color='green')

    ax.legend(fontsize='xx-small')
plt.show()
