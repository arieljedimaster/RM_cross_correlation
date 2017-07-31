import numpy as np
import matplotlib.pyplot as plt
from bfield_functions_april11 import *
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib as mpl
from astropy.io import fits
import astropy.cosmology as cosmo
import pylab as pyl
import pyfits as pf
import healpy as hp
import bfield_functions_april11 as bf

import time

# Tessa's edits to my code, done on april 11th 2017

hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)

############################################
#import data

#data directory
pwd_data = '/Users/arielamaral/Documents/AST430/Data/'

print ' '
print "Uploading Catalogues...."

#RM catalogue Taylor et al.

RA_hms, err_RA, Dec_hms, err_DEC, b, I, S_I, S_I_err, ave_P, ave_P_err, pol_percent, pol_percent_err,RM, RM_err = np.genfromtxt(pwd_data + 'RMcat_taylor.csv', delimiter = ',', unpack=True, skip_header=1,dtype=str)

RA_hms1, err_RA, Dec_hms1, err_DEC, b, I, S_I, S_I_err, ave_P, ave_P_err, pol_percent, pol_percent_err,RM, RM_err = np.genfromtxt(pwd_data + 'RMcat_taylor.csv', delimiter = ',', unpack=True, skip_header=1)

#RM Hammond et al. catalogue
'''
RM_hammond = fits.open(pwd_data+'rm_redshift_catalog.fits')
RM_hammond_header = RM_hammond[1].header
RM_hammond_data = RM_hammond[1].data
'''

#WISE/superCOSMOS

WISE_cat = fits.open(pwd_data+'WISExSCOS.photoZ.MAIN.fits')
WISE_header = WISE_cat[1].header
WISE_data = WISE_cat[1].data

gal_RA = WISE_data['ra_WISE ']
gal_Dec = WISE_data['dec_WISE']
gal_photo_z = WISE_data ['zANNz   ']



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
print "* Converting RM catalogue coordinates from HMS to degrees...."
print ' '

#randomly shuffling the RMs

#np.random.shuffle(RM)




RA_RM, Dec_RM = np.loadtxt('RA_DEC_degrees.txt',unpack=True)

#RA_RM = np.random.uniform(0, 360., len(RM))
#Dec_RM = np.random.uniform(-60., 60, len(RM))

#get rid of RM sources not in the specific area we want

'''
RA_RM, Dec_RM, RM = RM_reduc(RA_RM, Dec_RM, RM)
'''

#reduce RM

print ' '
print "** Subtracting foreground....."
print ' '

#RM = foreground_subtract(RA_RM,Dec_RM,RM)
RM, RM_err = foreground_subtract_niels(RA_RM,Dec_RM,RM, RM_err)

#we want to use the absolute value of the RM for this calculation

RM = abs(RM)


print ' '
print "*** Spliting up galaxies into redshift bins....."
print ' '

z_split, z_mean, RA_split, Dec_split, z_bin_list = photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)
#print "zmean is: ", z_mean

print ' '
print "**** Calculating galaxy number density grid....."
print ' '

num_dens_grid = healpy_grid(res,gal_RA,gal_Dec)

total_gal_in_z = total_num_gals_in_z(z_bin_list)

print ' '
print "**** Calculating PREDICTED galaxy number density grid....."
print ' '

sky_frac = 0.75
radii_list = np.concatenate([[0.],np.logspace(1,3.04,9)])

'''
radii_list1 = np.logspace(1,3.04,9)
radii_list1[0]=0.
radii_list = radii_list1
'''
'''
radii_binavg = (radii_list1[0] + radii_list1[1])/2.
radii_list = np.zeros(len(radii_list1)-1)
radii_list[0] = radii_binavg
radii_list[1:] = radii_list1[2:]
#radii_list=np.concatenate(np.array(radii_list1[0.]),radii_list1[2:])
'''


#weight average number of galaxies per radial bin over all z bin
avegal_wt = predict_numbdense(total_gal_in_z,z_mean,radii_list,sky_frac)


print ' '
print "***** Calculating the deg --> kpc conversion factor for each galaxy....."
print ' '

kpc2deg_list = gal_conv_fact(z_bin_list,z_mean)

print ' '
print "****** Calculating Mask which gets rid of RM in cells with zero galaxies and those which are too close to the edges....."
print ' '

#radii_list = np.array([0.,15.,30.,60.,125.,255.,515.,1025., 1200.]) #list of radii from graph in Ue-Li's paper
#radii_list = np.logspace(1,3.04,14)
#radii_list = np.logspace(1,3.04,5)



t5=time.time()

RA_RM, Dec_RM, RM = mask2(num_dens_grid, nside, RA_RM, Dec_RM, RM, radii_list)

t6 = time.time()
'''

plt.figure()
plt.plot(RA_RM, Dec_RM, 'r.')
plt.show()

np.save('masked_RM_taylor.npy',(RA_RM,Dec_RM, RM))
np.savetxt('masked_taylor.txt',(RA_RM,Dec_RM, RM))
'''


print "Masking took " + str((t6-t5)) + " seconds."


print "Number of RM sources after mask", len(RA_RM)


'''
print ' '
print "******* Calculating distance matrices from each galaxy to each RM source in postage stamps (might take a while)....."
print ' '


t1=time.time()

#dist_matrix_deg, dist_matrix_kpc, dist_indeces, dist_z_split = distance_func_tessa(RA_RM, Dec_RM, gal_RA, gal_Dec, kpc2deg_list, radii_list, z_bin_list)


dist_matrix_deg2, dist_matrix_kpc2, dist_indeces2, dist_z_split2 = distance_func2(nside,RA_RM,Dec_RM,gal_RA,gal_Dec,kpc2deg_list, radii_list, z_bin_list)
t2=time.time()

print "This took " + str((t2-t1)/60.) + " minutes."


np.save("dist_matrix_deg_new_healpy.npy", dist_matrix_deg2)
np.save('dist_matrix_kpc_new_healpy.npy',dist_matrix_kpc2)
np.save('dist_indeces_new_healpy.npy',dist_indeces2)
np.save('dist_z_split_new_healpy.npy',dist_z_split2)

'''
#add in autocorrelation
#use the predicted mean in the cross correlation (instead of the rho mean in cross_corr)

print ' '
print "******* Uploading distance matrices....."
print ' '

t1=time.time()

dist_matrix_deg = np.load("dist_matrix_deg_new_tessa.npy")
dist_matrix_kpc = np.load('dist_matrix_kpc_new_tessa.npy')
dist_indeces = np.load('dist_indeces_new_tessa.npy')
dist_z_split = np.load('dist_z_split_new_tessa.npy')

t2=time.time()

#print "This took " + str((t2-t1)/60.) + " minutes."


print ' '
print "******** Cross Correlating....."
print ' '

t3=time.time()


cross_corr_func, rho_avg = bf.cross_corr(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split)

print "main_april11 predited mean input:  radii_list, z_mean, total_gal_in_z"
print radii_list, z_mean, total_gal_in_z


cross_corr_func_predict,rho_avg = cross_corr_predict(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split)

#print cross_corr_func_predict
print cross_corr_func
#
t4=time.time()

print "Cross correlating took " + str((t4-t3)/60.) + " minutes."



#cross_corr_plot(radii_list, cross_corr_func)

radial_bins = radii_list

areabins = np.pi*radial_bins**2
areabins1 = areabins[1:]-areabins[:-1]


radii_list = [(a + b) /2. for a, b in zip(radii_list[::], radii_list[1::])]

plt.figure()
plt.plot(radii_list, cross_corr_func,'b-', linewidth = 2, label = "Using Local Mean")
#plt.plot(radii_list, cross_corr_func_predict,'r--', linewidth = 2, label = "Using Predicted Mean")
plt.legend(loc='lower right')
plt.xlabel("Radial Bins [Mpc]")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation")
plt.xscale('log')
plt.show()

plt.figure()
#plt.plot(radii_list, cross_corr_func,'b-', linewidth = 2, label = "Using Local Mean")
plt.plot(radii_list, cross_corr_func_predict,'r--', linewidth = 2, label = "Using Predicted Mean")
plt.legend()
plt.xlabel("Radial Bins [Mpc]")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation")
plt.xscale('log')
plt.show()

'''
plt.figure()
plt.plot(radii_list, rho_avg*areabins1, linewidth = 2, label = "Local Mean")
plt.plot(radii_list, avegal_wt, linewidth = 2, label = "Predicted Mean")
plt.legend()
plt.xlabel("Radial Bins [Mpc]")
plt.ylabel("Weighted Mean")
plt.title("Weighted Mean Number Density Per Radial Bin")
plt.xscale('log')
plt.show()
'''


#np.savetxt("cross_corr_unnorm4.txt", (cross_corr_func,radii_list))


np.save("cross_corr_file.npy", cross_corr_func_predict)

'''

cross_corr_file = np.load('cross_corr_file.npy')
cross_corr_file = np.stack((cross_corr_file,cross_corr_func),axis = 0)
np.save("cross_corr_file.npy", cross_corr_func)
'''

print " "
print "ALL DONE, go check your plots folder :)"
print " "

