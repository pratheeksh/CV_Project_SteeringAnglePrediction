import sys
import numpy as np
steering = open(sys.argv[1])
camera = open(sys.argv[2])
steering_dict = dict()
time_scale = int(1e8)
to_write_center = open("images_to_angles_center.csv", 'w')
to_write_right = open("images_to_angles_right.csv", 'w')
to_write_left = open("images_to_angles_left.csv", 'w')

for s in steering.readlines():
    if 'angle' in s:
        continue
    s = s.split(',')
    timestamp = int(s[0])
    angle = float(s[1])
    key = (timestamp/time_scale)
    if key in steering_dict:
        steering_dict[key].append(angle)
    else:
        steering_dict[key] = [angle]
averaged_angles = dict()
for key in steering_dict:
    averaged_angles[key] = np.average(steering_dict[key])

center = dict()
left = dict()
right = dict()
for c in camera.readlines():
    if 'filename' in c:
        continue
    c = c.split(',')
    timestamp = int(c[0])
    key = (timestamp / time_scale)
    if "center" in c[-1]:
        center[timestamp] = averaged_angles[key]
    elif "left" in c[-1]:
        left[timestamp] = averaged_angles[key]
    else:
        right[timestamp] = averaged_angles[key]
    

for k in center:
    to_write_center.write(str(k) + "," + str(center[k]) + "\n")

for k in left:
    to_write_left.write(str(k) + "," + str(left[k]) + "\n")
for k in right:
    to_write_right.write(str(k) + "," + str(right[k]) + "\n")








