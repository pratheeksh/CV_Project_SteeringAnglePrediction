import csv
import sys
res = open("res.csv",'w')
csvfile = open(sys.argv[1], 'rb').readlines()
for l in csvfile:
	filename, angle=l.split(',')
	val = float(angle)
	if (val<=-0.75):
		if (val>=-2.25):
			cl = 2
		else:
			cl = 1
	
	else:
		if (val<=0.50):
			cl = 3
		elif (val<=2.25):
			cl = 4
		else:
			cl = 5
	
		
	res.write(filename + ',' + str(cl) + '\n')
