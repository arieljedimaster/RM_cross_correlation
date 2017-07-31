import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib as mpl
from astropy.io import fits
import astropy.cosmology as cosmo
import pylab as pyl
import pyfits as pf
import healpy as hp

import time

import sys
sys.path.insert(0, '/Users/arielamaral/Documents/AST430/Code/Code_new_cat')
import bfield_functions_cat as bf
import dist_matrix_code_tessa as dmcT


#1st test case

distmatrix1 = [np.zeros(100) + 1.]

#2nd test case
distmatrix2 = np.zeros(100)

for i in np.arange(0, 100, 10):
	distmatrix2[i:i+10] = np.arange(1,11)

distmatrix2 = [distmatrix2]

#3rd test case

distmatrix3 = [np.random.normal(loc=100.0, scale=40.0, size=100)]

#4th real test case for ONE random RM source

distmatrix4 = [np.load('random_matrices/hammond_matrices/dist_matrix_kpc_run_0.npy')[0]]


RA_RM = [20.]
Dec_RM = [40.]

RM = [12.]

radii_list = np.linspace(0., 2511.88643150958, 9)

radii_list = np.arange(1,11)

radii_list = np.arange(0, 200, 20)


radii_list = np.linspace(0., 2511.88643150958, 9)



distmatrix5 = np.load('random_matrices/hammond_matrices/dist_matrix_kpc_run_0.npy')

print len(distmatrix5)

print len(distmatrix5[0])

RA_RM, Dec_RM, RM = np.loadtxt('random_RM_sources/hammond_random_sources/random_RM_RA_DEC_run_0.txt', unpack = True)



avg_gal_num1, bin_edges1, avg_gal_std1, total_gal_num1 = bf.gal_sources_hist(RA_RM, Dec_RM, RM, distmatrix5, radii_list)



print avg_gal_num1
'''
fig, ax = plt.subplots()
ax1 = plt.subplot(111)
ax1.set_title('Test 4')
ax1.bar(bin_edges1[:-1], avg_gal_num1, width=[(bin_edges1[j+1]-bin_edges1[j]) for j in range(len(bin_edges1)-1)], alpha = 0.5, color = 'r')
mid = 0.5*(bin_edges1[1:] + bin_edges1[:-1])
#ax1.errorbar(mid, avg_gal_num1, yerr=rand_gal_linbins_err, fmt='none', color='k')
ax1.set_ylabel('Galaxy counts')
ax1.set_xlabel('Radial Bins')
ax1.legend(loc = 'upper left')
ax1.grid(True)
plt.show()
'''


