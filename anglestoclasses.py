__author__ = 'bharathipriyaa'

import numpy as np

def convert_angles_class() :
    file_ = open("data_/csv/testclasses.csv",'w')
    csvfile = open('data_/csv/test_center.csv', 'r').readlines()
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

def shift_angles(inputfile, outputfile):
    csvfile = open(inputfile, 'r').readlines()
    file_ = open(outputfile,'w')

    # 20 classes > 0 and 20 classes < 0
    # from -2.4 to + 2.4
    for l in csvfile:
        filename, angle = l.split(',')
        angle = float(angle) + 2.4
        print(filename, angle)
        file_.write(filename + ',' + str(angle) + '\n')
    file_.close()

if __name__ == "__main__":
    inputfile = "data_/csv/center.csv"
    outputfile = "data_/csv/centerpositive.csv"
    shift_angles(inputfile, outputfile)