import numpy as np
import matplotlib.pyplot as plt
from bfield_functions_cat import *
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib as mpl
from astropy.io import fits
import astropy.cosmology as cosmo
import pylab as pyl
import pyfits as pf
import healpy as hp

import time

hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)

############################################
#import data

#data directory
pwd_data = '/Users/arielamaral/Documents/AST430/Data/'

print ' '
print "Uploading Catalogues...."

#RM catalogue Taylor et al.

RA_hms, err_RA, Dec_hms, err_DEC, b, I, S_I, S_I_err, ave_P, ave_P_err, pol_percent, pol_percent_err,RM, RM_err = np.genfromtxt(pwd_data + 'RMcat_taylor.csv', delimiter = ',', unpack=True, skip_header=1)

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

RA_RM, Dec_RM = np.loadtxt('RA_DEC_degrees.txt',unpack=True)

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

z_split, z_mean, gal_RA_split, gal_Dec_split, z_bin_list = photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)

print ' '
print "**** Calculating galaxy number density grid....."
print ' '

num_dens_grid = healpy_grid(res,gal_RA,gal_Dec)

total_gal_in_z = total_num_gals_in_z(z_bin_list)

print ' '
print "**** Calculating PREDICTED galaxy number density grid....."
print ' '

sky_frac = 0.75



#log spaced bins

radii_list_logbins= np.logspace(1.,3.4,9)
radii_list_logbins[0] = 0.

#linear spaced bins

radii_list_linbins = np.linspace(0., 2511.88643150958, 9)

print "   "
print "The log binned radii list we're using in kpc is: ", radii_list_logbins
print "The linear binned radii list we're using in kpc is: ", radii_list_linbins
print "   "


#radii_list = np.concatenate([[0.]

#weight average number of galaxies per radial bin over all z bin
avegal_wt_logbins = predict_numbdense(total_gal_in_z,z_mean,radii_list_logbins,sky_frac)

avegal_wt_linbins = predict_numbdense(total_gal_in_z,z_mean,radii_list_linbins,sky_frac)


print ' '
print "***** Calculating the deg --> kpc conversion factor for each galaxy....."
print ' '

kpc2deg_list = gal_conv_fact(z_bin_list,z_mean)

print ' '
print "****** Calculating Mask which gets rid of RM in cells with zero galaxies and those which are too close to the edges....."
print ' '

t5=time.time()

RA_RM, Dec_RM, RM = mask2(num_dens_grid, RA_RM, Dec_RM, RM, radii_list_logbins)

#I dont need to do the masking twice because both of the binning methods have the same max bins

t6 = time.time()


print "Masking took " + str((t6-t5)) + " seconds."


print "Number of RM sources after mask", len(RA_RM)



print ' '
print "******* Calculating distance matrices from each galaxy to each RM source in postage stamps (might take a while)....."
print ' '

'''
t1=time.time()

#dist_matrix_deg, dist_matrix_kpc, dist_indeces, dist_z_split = distance_func_tessa(RA_RM, Dec_RM, gal_RA, gal_Dec, kpc2deg_list, radii_list, z_bin_list)


dist_matrix_deg, dist_matrix_kpc, dist_indeces, dist_z_split = distance_func2(nside,RA_RM,Dec_RM,gal_RA,gal_Dec,kpc2deg_list, radii_list, z_bin_list)
t2=time.time()

print "This took " + str((t2-t1)/60.) + " minutes."


np.save("dist_matrix_deg_large.npy", dist_matrix_deg2)
np.save('dist_matrix_kpc_new_large.npy',dist_matrix_kpc2)
np.save('dist_indeces_new_large.npy',dist_indeces2)
np.save('dist_z_split_new_large.npy',dist_z_split2)

'''
#add in autocorrelation
#use the predicted mean in the cross correlation (instead of the rho mean in cross_corr)

print ' '
print "******* Uploading distance matrices....."
print ' '

t1=time.time()

'''
dist_matrix_deg = np.load("dist_matrix_deg_new_tessa.npy")
dist_matrix_kpc = np.load('dist_matrix_kpc_new_tessa.npy')
dist_indeces = np.load('dist_indeces_new_tessa.npy')
dist_z_split = np.load('dist_z_split_new_tessa.npy')
'''

dist_matrix_deg = np.load("dist_matrix_deg_large.npy")
dist_matrix_kpc = np.load('dist_matrix_kpc_new_large.npy')
dist_indeces = np.load('dist_indeces_new_large.npy')
dist_z_split = np.load('dist_z_split_new_large.npy')


t2=time.time()

#print "This took " + str((t2-t1)/60.) + " minutes."


print ' '
print "******** Cross Correlating....."
print ' '

t3=time.time()


#cross_corr_func = cross_corr(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split)


#log bins
avg_gal_num_logbins, bin_edges_logbins = gal_sources_hist(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_logbins, z_mean, total_gal_in_z, dist_z_split,avegal_wt_logbins)

#linear bins
avg_gal_num_linbins, bin_edges_linbins = gal_sources_hist(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_linbins, z_mean, total_gal_in_z, dist_z_split,avegal_wt_linbins)


#log bins
cross_corr_func_predict_logbins = cross_corr_predict(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_logbins, z_mean, total_gal_in_z, dist_z_split,avegal_wt_logbins)

#linear bins
cross_corr_func_predict_linbins = cross_corr_predict(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_linbins, z_mean, total_gal_in_z, dist_z_split,avegal_wt_linbins)


t4=time.time()

print "Cross correlating took " + str((t4-t3)/60.) + " minutes."

'''
#log bins

radial_bins_logbins = radii_list_logbins

areabins_logbins = np.pi*radial_bins_logbins**2
areabins1_logbins = areabins_logbins[1:]-areabins_logbins[:-1]

radii_list_logbins = [(a + b) /2. for a, b in zip(radii_list_logbins[::], radii_list_logbins[1::])]


#linear bins

radial_bins_linbins = radii_list_linbins
radial_bins_linbins1 = radial_bins_linbins[0:-2]


areabins_linbins = np.pi*radial_bins_linbins**2
areabins1_linbins = areabins_linbins[1:]-areabins_linbins[:-1]

radii_list_linbins = [(a + b) /2. for a, b in zip(radii_list_linbins[::], radii_list_linbins[1::])]
'''


#plotting the histograms

fig, ax = plt.subplots()
ax1 = plt.subplot(211)
ax1.set_title('Plot of average number of galaxies around RM source (logscale)')
#ax.bar(radii_list_linbins, avg_gal_num_linbins, width=[(radial_bins_linbins[j+1]-radial_bins_linbins[j]) for j in range(len(radial_bins_linbins)-1)], alpha = 0.5, label = "linear bins", color = 'b')
#ax.bar(radii_list_logbins, avg_gal_num_logbins, width=[(radial_bins_logbins[j+1]-radial_bins_logbins[j]) for j in range(len(radial_bins_logbins)-1)], alpha = 0.5, label = "log bins", color = 'r')
ax1.bar(bin_edges_linbins[:-1], avg_gal_num_linbins, width=[(bin_edges_linbins[j+1]-bin_edges_linbins[j]) for j in range(len(bin_edges_linbins)-1)], alpha = 0.5, label = "linear bins", color = 'b')
ax1.bar(bin_edges_logbins[:-1], avg_gal_num_logbins, width=[(bin_edges_logbins[j+1]-bin_edges_logbins[j]) for j in range(len(bin_edges_logbins)-1)], alpha = 0.7, label = "log bins", color = 'r')

ax1.set_ylabel('Galaxy counts [log]')
ax1.set_xlabel('Radial Bins [log]')
ax1.legend(loc = 'upper left')
ax1.grid(True)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2 = plt.subplot(212)
ax2.set_title('Plot of average number of galaxies around RM source')
#ax.bar(radii_list_linbins, avg_gal_num_linbins, width=[(radial_bins_linbins[j+1]-radial_bins_linbins[j]) for j in range(len(radial_bins_linbins)-1)], alpha = 0.5, label = "linear bins", color = 'b')
#ax.bar(radii_list_logbins, avg_gal_num_logbins, width=[(radial_bins_logbins[j+1]-radial_bins_logbins[j]) for j in range(len(radial_bins_logbins)-1)], alpha = 0.5, label = "log bins", color = 'r')
ax2.bar(bin_edges_logbins[:-1], avg_gal_num_logbins, width=[(bin_edges_logbins[j+1]-bin_edges_logbins[j]) for j in range(len(bin_edges_logbins)-1)], alpha = 0.5, label = "log bins", color = 'r')
ax2.bar(bin_edges_linbins[:-1], avg_gal_num_linbins, width=[(bin_edges_linbins[j+1]-bin_edges_linbins[j]) for j in range(len(bin_edges_linbins)-1)], alpha = 0.7, label = "linear bins", color = 'b')
ax2.set_ylabel('Galaxy counts [#]')
ax2.set_xlabel('Radial Bins')
ax2.legend(loc = 'upper left')
ax2.grid(True)


fig.savefig('/Users/arielamaral/Documents/AST430/Plots/cross_correlation/wise_taylor/binning/average_galaxies_binning_test_binplot.png' )
fig.show()



plt.figure()
#plt.plot(radii_list, cross_corr_func,'g-', linewidth = 2, label = "Using Local Mean")
#plt.plot(radii_list_logbins[1:], cross_corr_func_predict_logbins, 'b+', markersize = 30)
#plt.plot(radii_list_logbins[1:], cross_corr_func_predict_logbins,'r-', linewidth = 2, label = "Log bins")
plt.plot(radii_list_linbins[1:], cross_corr_func_predict_linbins, 'm+', markersize = 30)
plt.plot(radii_list_linbins[1:], cross_corr_func_predict_linbins,'g-', linewidth = 2, label = "Linear Bins")
plt.legend()
plt.xlabel("Radial Bins [Mpc]")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation")
plt.xscale('log')
plt.xlim((np.amin(radii_list_logbins) - 30 ,np.amax(radii_list_linbins)+100))
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/cross_correlation/wise_taylor/binning/CCF_maxbin_' + str(max(radii_list_logbins))+ '_binning test.png' )
plt.show()

#np.save("cross_corr_local_file.npy", cross_corr_func)
#np.save("cross_corr_predict_file.npy", cross_corr_func_predict)

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

np.save("cross_corr_local_file_large.npy", cross_corr_func)
'''

print " "
print "ALL DONE, go check your plots folder :)"
print " "

