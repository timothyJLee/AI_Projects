import pickle
import numpy as np
def rowstr(row):
	return ''.join('+' if e > 0 else '-' for e in row)

def viewmnist(x,y):

	img = (x > 0.25).astype(int).reshape((28, 28))
	print ('\n'.join(rowstr(row) for row in img))
	print ('n = {0}'.format(y), input())



