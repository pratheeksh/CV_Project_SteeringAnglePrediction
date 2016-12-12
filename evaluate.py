import sys
import pickle
import math
results = open(sys.argv[1])

evaluation_data = {}
try:
	evaluation_data = pickle.load(open("eval.p", "rb"))
except:
	eval_csv = open("eval.csv",r)
	for l in eval_csv.readlines():
		frame_id, angle = eval_csv.split(',')
		if frame_id == "frame_id":
			continue
		evaluation_data[frame_id] = float(angle)
		pickle.dump(evaluation_data, open("eval.p", wb))
error = 0
n = 0
for l in results.readlines():
	frame_id, angle = results.split(',')
	if frame_id == "frame_id":
		continue
	error += (evaluation_data[frame_id] - float(angle)) * (evaluation_data[frame_id] - float(angle))
	n += 1
final_error = math.sqrt(error/n)
print final_error