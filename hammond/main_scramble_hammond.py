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
from bfield_functions_cat import *

hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)

############################################
#import data

#data directory
pwd_data = '/Users/arielamaral/Documents/AST430/Data/'

print ' '
print "Uploading Catalogues...."

#RM Hammond et al. catalogue

RM_hammond = fits.open(pwd_data+'rm_redshift_catalog.fits')
RM_hammond_header = RM_hammond[1].header
RM_hammond_data = RM_hammond[1].data

RA_RM = RM_hammond_data['RA_DEG_J2000']
Dec_RM = RM_hammond_data['DEC_DEG_J2000']

RM = RM_hammond_data['NVSS_RM ']

RM_err = RM_hammond_data['NVSS_RM_ERR']

RM_z = RM_hammond_data['SELECTED_REDSHIFT']

#WISE/superCOSMOS

WISE_cat = fits.open(pwd_data+'WISExSCOS.photoZ.MAIN.fits')
WISE_header = WISE_cat[1].header
WISE_data = WISE_cat[1].data

gal_RA = WISE_data['ra_WISE ']
gal_Dec = WISE_data['dec_WISE']
gal_photo_z = WISE_data ['zANNz   ']


print "Number of RM sources which have redshifts greater than z = 0.5: ", len(np.where(RM_z> 0.5)[0])



RA_RM = RA_RM[np.where(RM_z>0.5)[0]]

Dec_RM = Dec_RM[np.where(RM_z>0.5)[0]]

RM = RM[np.where(RM_z>0.5)[0]]

RM_err = RM_err[np.where(RM_z>0.5)[0]]


RM_z = RM_z[np.where(RM_z>0.5)[0]]



#########################################################
#specifying healpy resolution

#using the same resolution as Niels' maps (for now)
nside= 128
npix = 12*nside**2
res = int(np.log(nside)/np.log(2))



print "total number of galaxy sources: ", len(gal_RA)

gal_RA, gal_Dec, gal_photo_z = z_reduc(gal_RA, gal_Dec, gal_photo_z)

print "total number of galaxy sources after filtering out 0.1 < z < 0.5: ", len(gal_RA)
print " "



############################################################


print ' '
print "** Subtracting foreground....."
print ' '

RM, RM_err = foreground_subtract_niels(RA_RM,Dec_RM,RM,RM_err)
RM = abs(RM)

print ' '
print "*** Uploading distance matrices....."
print ' '
dist_matrix_deg = np.load("dist_matrix_deg_hammond.npy")
dist_matrix_kpc = np.load('dist_matrix_kpc_hammond.npy')
dist_indeces = np.load('dist_indeces_hammond.npy')
dist_z_split = np.load('dist_z_split_hammond.npy')



n = 1000


#radii_list = np.logspace(1.,3.4,9)
#radii_list[0] = 0.

#starting from 100 Mpc

#radii_list = np.logspace(2.,3.4,9)

#linear spaced bins

radii_list = np.linspace(0., 2511.88643150958, 9)

print "Radial Bins: ", radii_list

cross_corr_matrix = np.zeros((n,len(radii_list)-1))

print ' '
print "**** Starting to Cross Correlate for different realizations...."
print ' '


for i in np.arange(0, n):
	if (i % 10 == 0) & (i!=0):
		print "On scramble run number " + str(i)
	np.random.shuffle(RM)

	z_split, z_mean, RA_split, Dec_split, z_bin_list = photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)

	num_dens_grid = healpy_grid(res,gal_RA,gal_Dec)

	kpc2deg_list = gal_conv_fact(z_bin_list,z_mean)

	t5=time.time()

	RA_RM, Dec_RM, RM = mask2(num_dens_grid, RA_RM, Dec_RM, RM, radii_list)

	t6 = time.time()

	total_gal_in_z = total_num_gals_in_z(z_bin_list)

	sky_frac = 0.75

	avegal_wt = predict_numbdense(total_gal_in_z,z_mean,radii_list,sky_frac)

	cross_corr_func = cross_corr_predict(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split,avegal_wt)

	#cross_corr_func= cross_corr(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split)

	cross_corr_matrix[i] = cross_corr_func


cross_corr_func_avg = np.mean(cross_corr_matrix, axis = 0)

#we need to keep track of errors

cross_corr_err = np.std(cross_corr_matrix, axis = 0)

print cross_corr_func_avg

print cross_corr_err

#cross_corr_real = np.load('cross_corr_file.npy')
cross_corr_real = np.load('cross_corr_predict_file_hammond.npy')

#plotting the cross correlation function
radii_list = [(a + b) /2. for a, b in zip(radii_list[::], radii_list[1::])]

plt.figure()
plt.plot(radii_list, cross_corr_func_avg, 'r-', linewidth = 2, markersize = 30, label = 'Scrambeled RM for '+str(n)+' runs')
plt.errorbar(radii_list, cross_corr_func_avg, yerr = cross_corr_err, color = 'red', ecolor='r')
plt.plot(radii_list, cross_corr_real, 'g-', linewidth = 2, markersize = 30, label = 'Real RM')
plt.plot(radii_list, cross_corr_real, 'go')
plt.title("Average Scrambled Cross Correlation Function for "+str(n)+" times - Hammond catalogue")
plt.xlabel('Radii [kpc]')
plt.ylabel('Cross-Correlation')
plt.legend()
plt.grid()
plt.yscale('symlog')
plt.xscale('log')
plt.xlim((np.amin(radii_list) - 10,np.amax(radii_list)+100))
plt.figtext(0,0,"Subtracting galactic foreground with Opperman et al (2014) foreground, Using Hammond et al RM catalogue")
plt.savefig("/Users/arielamaral/Documents/AST430/Plots/cross_correlation/wise_hammond/cross_corr_plot_wise_hammond_scrambled" + str(n)+".png")
plt.show()


print " "
print "ALL DONE, go check your plots folder :)"
print " "


