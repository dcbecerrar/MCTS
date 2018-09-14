import numpy as np
from mayavi.mlab import *
import sys

def print_mesh(filename):
	f = np.loadtxt(filename)

	coord = f[:-f.shape[0]/4]
	indices = f[-f.shape[0]/4:]

	'''
	indices = np.arange(f.shape[0])
	indices = indices.reshape(indices.shape[0]/3,3)
	'''

	triangular_mesh(coord[:,0], coord[:,1], coord[:,2], indices)

	np.save(filename.split(".")[0]+".npy",coord.reshape(coord.shape[0]/3,3,3))
	input("press")
	return

if __name__=="__main__":
	print_mesh(sys.argv[1])
