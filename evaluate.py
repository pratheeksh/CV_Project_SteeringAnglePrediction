import csv
import math
import pickle
import sys
from optparse import OptionParser

import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("-p", "--plot", dest="isplot", action="store_true",
                  help="To be able to plot or not")
results = csv.reader(open(sys.argv[1]))
(options, args) = parser.parse_args()

evaluation_data = {}
try:
    evaluation_data = pickle.load(open("eval.p", "rb"))
except:
    eval_csv = csv.reader(open("eval.csv"))
    print("Eval csv is opened")
    for l in eval_csv:
        frame_id, angle = l[0], l[1]
        if frame_id == "frame_id":
            continue
        evaluation_data[frame_id] = float(angle)
        pickle.dump(evaluation_data, open("eval.p", "wb"))
error = 0
n = 0
count = 0
original_angles = []
new_angles = []
next(results)
for l in results:
    frame_id, angle = l[0], l[1]
    if frame_id == "frame_id":
        continue
    if frame_id in evaluation_data:
        count = count + 1
    error += (evaluation_data[frame_id] - float(angle)) ** 2
    original_angles.append(evaluation_data[frame_id])
    new_angles.append(float(angle))
    n += 1
final_error = math.sqrt(error / n)
print("Rows in common", count)
print final_error  # /10000.00
if (options.isplot):
    x = range(5614)
    plt.plot(x, original_angles, label="original Angles")
    plt.plot(x, new_angles, label="Predicted Angles")
    plt.legend()
    plt.show()
    plt.savefig("result.png")
