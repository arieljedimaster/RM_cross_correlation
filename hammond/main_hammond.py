# main hammond

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
import dist_matrix_code_tessa as dmcT


hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)


#data directory
pwd_data = '/Users/arielamaral/Documents/AST430/Data/'

pwd_taylor = '/Users/arielamaral/Documents/AST430/Code/Code_new_cat/'

print ' '
print "Uploading Catalogues...."


#RM Hammond et al. catalogue

RM_hammond = fits.open(pwd_data+'rm_redshift_catalog.fits')
RM_hammond_header = RM_hammond[1].header
RM_hammond_data = RM_hammond[1].data
'''
'SDSS_Z  '
'NED_DEFAULT_REDSHIFT'
'SIMBAD_DEFAULT_REDSHIFT'
'SIXDF_Z '
'TWOQZ_SIXQZ_Z'
'SELECTED_REDSHIFT'
'''

RA_RM_hammond = RM_hammond_data['RA_DEG_J2000']
Dec_RM_hammond = RM_hammond_data['DEC_DEG_J2000']

RM_hammond = RM_hammond_data['NVSS_RM ']

RM_err_hammond = RM_hammond_data['NVSS_RM_ERR']

RM_z_hammond = RM_hammond_data['SELECTED_REDSHIFT']



#deleting all sources which are less than redshift 0.5

print "Number of RM sources which have redshifts greater than z = 0.5: ", len(np.where(RM_z_hammond > 0.5)[0])


RA_RM_hammond = RA_RM_hammond[np.where(RM_z_hammond>0.5)[0]]

Dec_RM_hammond = Dec_RM_hammond[np.where(RM_z_hammond>0.5)[0]]

RM_hammond = RM_hammond[np.where(RM_z_hammond>0.5)[0]]

RM_err_hammond = RM_err_hammond[np.where(RM_z_hammond>0.5)[0]]

RM_z_hammond = RM_z_hammond[np.where(RM_z_hammond>0.5)[0]]



#TAYLOR

RA_hms_taylor, err_RA_taylor, Dec_hms_taylor, err_DEC_taylor, b_taylor, I_taylor, S_I_taylor, S_I_err_taylor, ave_P_taylor, ave_P_err_taylor, pol_percent_taylor, pol_percent_err_taylor, RM_taylor, RM_err_taylor = np.genfromtxt(pwd_data + 'RMcat_taylor.csv', delimiter = ',', unpack=True, skip_header=1)


RA_RM_taylor, Dec_RM_taylor = np.loadtxt(pwd_taylor+'RA_DEC_degrees.txt',unpack=True)



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


print ' '
print "** Subtracting foreground....."
print ' '

#RM = foreground_subtract(RA_RM,Dec_RM,RM)
RM_hammond, RM_err_hammond = foreground_subtract_niels(RA_RM_hammond,Dec_RM_hammond,RM_hammond, RM_err_hammond)

RM_taylor, RM_err_taylor = foreground_subtract_niels(RA_RM_taylor, Dec_RM_taylor, RM_taylor, RM_err_taylor)


#we want to use the absolute value of the RM for this calculation

RM_hammond = abs(RM_hammond)

RM_taylor = abs(RM_taylor)


print ' '
print "*** Spliting up galaxies into redshift bins....."
print ' '

z_split, z_mean, RA_split, Dec_split, z_bin_list = photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)


print ' '
print "**** Calculating galaxy number density grid....."
print ' '

num_dens_grid = healpy_grid(res,gal_RA,gal_Dec)

total_gal_in_z = total_num_gals_in_z(z_bin_list)

print ' '
print "**** Calculating PREDICTED galaxy number density grid....."
print ' '

sky_frac = 0.75


#radii_list= np.logspace(1.,3.4,9)
#radii_list[0] = 0.

'''
#starting the first radial bin at 100Mpc
radii_list = np.logspace(2.,3.4,9)
'''


#linear spaced bins

radii_list_linbins = np.linspace(0., 2511.88643150958, 9)


print "   "
print "The radii list we're using in kpc is: ", radii_list_linbins
print "   "


#weight average number of galaxies per radial bin over all z bin
avegal_wt = predict_numbdense(total_gal_in_z,z_mean,radii_list_linbins,sky_frac)


print ' '
print "***** Calculating the deg --> kpc conversion factor for each galaxy....."
print ' '

kpc2deg_list = gal_conv_fact(z_bin_list,z_mean)

print ' '
print "****** Calculating Mask which gets rid of RM in cells with zero galaxies and those which are too close to the edges....."
print ' '

t5=time.time()

RA_RM_hammond, Dec_RM_hammond, RM_hammond = mask2(num_dens_grid, RA_RM_hammond, Dec_RM_hammond, RM_hammond, radii_list_linbins)


RA_RM_taylor, Dec_RM_taylor, RM_taylor = mask2(num_dens_grid, RA_RM_taylor, Dec_RM_taylor, RM_taylor, radii_list_linbins)


t6 = time.time()


print "Masking took " + str((t6-t5)) + " seconds."


print "Number of RM sources after mask", len(RA_RM_hammond)


'''
print ' '
print "******* Calculating distance matrices from each galaxy to each RM source in postage stamps (might take a while)....."
print ' '

#calculating coord_RM,coord_gal,rra_RM,rra_gal

coord_gal=SkyCoord(gal_RA*u.deg,gal_Dec*u.deg,frame='fk5')
coord_RM=SkyCoord(RA_RM_hammond*u.deg,Dec_RM_hammond*u.deg,frame='fk5')
rra_RM=coord_RM.ra.wrap_at(180*u.degree).value
rra_gal=coord_gal.ra.wrap_at(180*u.degree).value


t1=time.time()

#dist_matrix_deg, dist_matrix_kpc, dist_indeces, dist_z_split = distance_func_tessa(RA_RM, Dec_RM, gal_RA, gal_Dec, kpc2deg_list, radii_list, z_bin_list)


#dist_matrix_deg, dist_matrix_kpc, dist_indeces, dist_z_split = distance_func2(nside,RA_RM_hammond,Dec_RM_hammond,gal_RA,gal_Dec,kpc2deg_list, radii_list, z_bin_list)


dist_matrix_deg_hammond, dist_matrix_kpc_hammond, dist_indeces_hammond, dist_z_split_hammond  = dmcT.distance_func_tessa3(coord_RM,coord_gal,rra_RM,rra_gal, kpc2deg_list, radii_list_linbins, z_bin_list)
t2=time.time()

print "This took " + str((t2-t1)/60.) + " minutes."


np.save("dist_matrix_deg_hammond.npy", dist_matrix_deg_hammond)
np.save('dist_matrix_kpc_hammond.npy',dist_matrix_kpc_hammond)
np.save('dist_indeces_hammond.npy',dist_indeces_hammond)
np.save('dist_z_split_hammond.npy',dist_z_split_hammond)

'''

print ' '
print "******* Uploading distance matrices....."
print ' '



dist_matrix_deg_hammond = np.load("dist_matrix_deg_hammond.npy")
dist_matrix_kpc_hammond = np.load('dist_matrix_kpc_hammond.npy')
dist_indeces_hammond = np.load('dist_indeces_hammond.npy')
dist_z_split_hammond = np.load('dist_z_split_hammond.npy')


dist_matrix_deg_taylor = np.load(pwd_taylor + "dist_matrix_deg_large.npy")
dist_matrix_kpc_taylor = np.load(pwd_taylor + 'dist_matrix_kpc_new_large.npy')
dist_indeces_taylor = np.load(pwd_taylor + 'dist_indeces_new_large.npy')
dist_z_split_taylor = np.load(pwd_taylor + 'dist_z_split_new_large.npy')

print ' '
print "******** Cross Correlating....."
print ' '

t3=time.time()


#cross_corr_func = cross_corr(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split)

'''
cross_corr_func_predict_hammond = cross_corr_predict(RA_RM_hammond, Dec_RM_hammond, RM_hammond, dist_matrix_kpc_hammond, radii_list_linbins, z_mean, total_gal_in_z, dist_z_split_hammond,avegal_wt)

cross_corr_func_predict_taylor = cross_corr_predict(RA_RM_taylor, Dec_RM_taylor, RM_taylor, dist_matrix_kpc_taylor, radii_list_linbins, z_mean, total_gal_in_z, dist_z_split_taylor,avegal_wt)
'''


cross_corr_func_classical_hammond =classical_cross_corr(RA_RM_hammond, Dec_RM_hammond, RM_hammond, dist_matrix_kpc_hammond, radii_list_linbins, z_mean, total_gal_in_z, dist_z_split_hammond)

cross_corr_func_classical_taylor = classical_cross_corr(RA_RM_taylor, Dec_RM_taylor, RM_taylor, dist_matrix_kpc_taylor, radii_list_linbins, z_mean, total_gal_in_z, dist_z_split_taylor)

t4=time.time()

print "Cross correlating took " + str((t4-t3)/60.) + " minutes."
'''

radial_bins = radii_list

areabins = np.pi*radial_bins**2
areabins1 = areabins[1:]-areabins[:-1]

radii_list = [(a + b) /2. for a, b in zip(radii_list[::], radii_list[1::])]
'''
'''
np.save("cross_corr_predict_file_hammond.npy", cross_corr_func_predict_hammond)
np.save("cross_corr_predict_file_taylor.npy", cross_corr_func_predict_taylor)
'''

np.save("cross_corr_func_classial_file_hammond.npy", cross_corr_func_classical_hammond)
np.save("cross_corr_func_classial_file_taylor.npy", cross_corr_func_classical_taylor)

plt.figure()

# Ue-Li Cross Correlation with Predicted Mean
#plt.plot(radii_list_linbins[1:], cross_corr_func_predict_hammond,'b-', linewidth = 2, label = "Using Hammond Cat with " + str(len(RA_RM_hammond))+" sources")
#plt.plot(radii_list_linbins[1:], cross_corr_func_predict_taylor,'g-', linewidth = 2, label = "Using Taylor Cat with " + str(len(RA_RM_taylor))+" sources")

#classical Cross- correlation
plt.plot(radii_list_linbins[1:], cross_corr_func_classical_hammond,'b-', linewidth = 2, label = "Using Hammond Cat with " + str(len(RA_RM_hammond))+" sources")
plt.plot(radii_list_linbins[1:], cross_corr_func_classical_taylor,'g-', linewidth = 2, label = "Using Taylor Cat with " + str(len(RA_RM_taylor))+" sources")

#zero correlation line.
plt.plot(radii_list_linbins[1:], radii_list_linbins[1:]*0.,'r--', linewidth = 4, alpha = 0.5, label = "Zero Correlation Line")

plt.legend(loc = 'upper right')
plt.xlabel("Upper Bound of Radial Bins [kpc]")
plt.ylabel("Cross-Correlation")
plt.title("Classical Cross-Correlation between RM and galaxy density")
#plt.title("Cross-Correlation between RM and galaxy density")

plt.xlim((np.amin(radii_list_linbins[1:]) ,np.amax(radii_list_linbins[1:])))
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/cross_correlation/wise_hammond/CCF_taylor_hammmond_overplot_classical.png')
plt.show()




print " "
print "ALL DONE, go check your plots folder :)"
print " "

