__author__ = 'bharathipriyaa'

import numpy as np
file_ = open("data_/csv/centerclasses.csv",'w')
csvfile = open('data_/csv/center.csv', 'r').readlines()
# 20 classes > 0 and 20 classes < 0
# from -2.4 to + 2.4
for l in csvfile:
    filename, angle = l.split(',')
    angle = float(angle)
    print(filename)
    angle_class = 0
    if angle < 0 :
        angle_class = 12 - int((angle * 100 * -1 ) /12)
        print(angle, angle_class)
    else :
        angle_class =  12 + int((angle * 100 ) /12)
        print(angle, angle_class)
    file_.write(filename + ',' + str(angle_class) + '\n')
file_.close()


