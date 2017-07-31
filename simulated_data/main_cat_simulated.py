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

hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)

############################################
#import data

#data directory
pwd_data = '/Users/arielamaral/Documents/AST430/Data/'

print ' '
print "Uploading Simulated RM Catalogue + Galaxy Catalogue...."

#Uploading simulated data

RA_RM, Dec_RM, RM, RM_err, gal_sort, RM_simulated, simulated_pix = np.loadtxt('tessa_sim/simulated_RM_data_zbin_weighted1.txt', unpack = True)

print "RM: ", RM
print "RM_simulated: ", RM_simulated

simulated_pix = simulated_pix.astype(int)
#WISE/superCOSMOS

WISE_cat = fits.open(pwd_data+'WISExSCOS.photoZ.MAIN.fits')
WISE_header = WISE_cat[1].header
WISE_data = WISE_cat[1].data

gal_RA = WISE_data['ra_WISE ']
gal_Dec = WISE_data['dec_WISE']
gal_photo_z = WISE_data ['zANNz   ']


res=10 #SWITCH THIS TO CHANGE UP RESOLUTION
nside=2**res
npix=12*nside**2

print "total number of galaxy sources: ", len(gal_RA)

gal_RA, gal_Dec, gal_photo_z = bf.z_reduc(gal_RA, gal_Dec, gal_photo_z)

print "total number of galaxy sources after filtering out 0.1 < z < 0.5: ", len(gal_RA)
print " "


############################################################

#we want to use the absolute value of the RM for this calculation

RM = np.absolute(RM)


print ' '
print "*** Spliting up galaxies into redshift bins....."
print ' '

z_split, z_mean, RA_split, Dec_split, z_bin_list = bf.photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)




#print "zmean is: ", z_mean

print ' '
print "**** Calculating galaxy number density grid....."
print ' '


num_dens_grid = bf.healpy_grid2(res,gal_RA,gal_Dec)#np.zeros(npix)

#total number density grid

#setting all zero pixels (the galactic plane) to NaN
num_dens_grid[num_dens_grid==0.] = np.nan
num_dens_grid_masked = hp.pixelfunc.ma(num_dens_grid)
num_dens_grid_masked.mask = np.isnan(num_dens_grid_masked)

total_gal_in_z = bf.total_num_gals_in_z(z_bin_list)


print ' '
print "* Double checking that the when we convert back to RM healpy that it gives us the same thing, if it doesnt then there's something wrong with our"
print ' '


pix_RM = bf.DeclRaToIndex(nside,Dec_RM,RA_RM)


print pix_RM


print "Number of pixels in pix_RM: ", pix_RM
print "Number of pixels in simulated_RM: ", simulated_pix

print "  "

print "Elements that are different between pix_RM and simulated_pix: ", np.setdiff1d(simulated_pix, pix_RM, assume_unique=False)
print "Elements that are the same between the two: ", np.intersect1d(simulated_pix, pix_RM, assume_unique=False)
print "length of intersect1d: ", len(np.intersect1d(simulated_pix, pix_RM, assume_unique=False))


gal_vals_from_RADEC = num_dens_grid_masked[pix_RM]

print ">> output from clustering_simulated_data: "
print RM_simulated
print gal_sort
print " "
print ">> output from uploading and converting back to healpy:"
print RM
print gal_vals_from_RADEC

plt.figure()
plt.plot(gal_vals_from_RADEC,np.absolute(RM),'go')
plt.title('Number of galaxies at each simulated RM location from RA/Dec')
plt.ylabel('abs(RM)')
plt.xlabel('Number of galaxies')
plt.grid()
plt.figtext(0,0,"z-bin weighted simulated data")
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/RM_vs_numgal_CCFinput_zbin_weighted_binning.png')


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



#weight average number of galaxies per radial bin over all z bin
avegal_wt_logbins = bf.predict_numbdense(total_gal_in_z,z_mean,radii_list_logbins,sky_frac)
avegal_wt_linbins = bf.predict_numbdense(total_gal_in_z,z_mean,radii_list_linbins,sky_frac)


print ' '
print "***** Calculating the deg --> kpc conversion factor for each galaxy....."
print ' '

kpc2deg_list = bf.gal_conv_fact(z_bin_list,z_mean)




'''
print ' '
print "******* Calculating distance matrices from each galaxy to each RM source in postage stamps (might take a while)....."
print ' '


t1=time.time()

dist_matrix_deg, dist_matrix_kpc, dist_indeces, dist_z_split = bf.distance_func_tessa(RA_RM, Dec_RM, gal_RA, gal_Dec, kpc2deg_list, radii_list, z_bin_list)
#dist_matrix_deg, dist_matrix_kpc, dist_indeces, dist_z_split = distance_func2(nside,RA_RM,Dec_RM,gal_RA,gal_Dec,kpc2deg_list, radii_list, z_bin_list)

t2=time.time()

print "This took " + str((t2-t1)/60.) + " minutes."


np.save("dist_matrix_deg_simulated_zbin_weighted.npy", dist_matrix_deg)
np.save('dist_matrix_kpc_new_simulated_zbin_weighted.npy',dist_matrix_kpc)
np.save('dist_indeces_new_simulated_zbin_weighted.npy',dist_indeces)
np.save('dist_z_split_new_simulated_zbin_weighted.npy',dist_z_split)


'''

print ' '
print "******* Uploading distance matrices....."
print ' '


t1=time.time()
'''
dist_matrix_deg = np.load("dist_matrix_deg_simulated.npy")
dist_matrix_kpc = np.load('dist_matrix_kpc_new_simulated.npy')
dist_indeces = np.load('dist_indeces_new_simulated.npy')
dist_z_split = np.load('dist_z_split_new_simulated.npy')
'''
#uploading the ones from Tessa's Run
print "Uploading from Tessa's Run...."
dist_matrix_deg = np.load("tessa_sim/dist_matrix_deg_simulated_zbin_weighted1.npy")
dist_matrix_kpc = np.load('tessa_sim/dist_matrix_kpc_new_simulated_zbin_weighted1.npy')
dist_indeces = np.load('tessa_sim/dist_indeces_new_simulated_zbin_weighted1.npy')
dist_z_split = np.load('tessa_sim/dist_z_split_new_simulated_zbin_weighted1.npy')

t2=time.time()
print "This took " + str((t2-t1)/60.) + " minutes."



print ' '
print "******** Cross Correlating....."
print ' '

t3=time.time()

#log bins
avg_gal_num_logbins, bin_edges_logbins = bf.gal_sources_hist(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_logbins, z_mean, total_gal_in_z, dist_z_split,avegal_wt_logbins)

#linear bins
avg_gal_num_linbins, bin_edges_linbins = bf.gal_sources_hist(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_linbins, z_mean, total_gal_in_z, dist_z_split,avegal_wt_linbins)



cross_corr_func_logbins = bf.cross_corr(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_logbins, z_mean, total_gal_in_z, dist_z_split)

cross_corr_func_predict_logbins = bf.cross_corr_predict(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_logbins, z_mean, total_gal_in_z, dist_z_split,avegal_wt_logbins)

#using with linear bins now
cross_corr_func_linbins = bf.cross_corr(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_linbins, z_mean, total_gal_in_z, dist_z_split)

cross_corr_func_predict_linbins = bf.cross_corr_predict(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_linbins, z_mean, total_gal_in_z, dist_z_split,avegal_wt_linbins)


t4=time.time()

print "Cross correlating took " + str((t4-t3)/60.) + " minutes."
'''
radial_logbins = radii_list_logbins

areabins_logbins = np.pi*radial_logbins**2
areabins1_logbins = areabins_logbins[1:]-areabins_logbins[:-1]

radii_list_logbins = [(a + b) /2. for a, b in zip(radii_list_logbins[::], radii_list_logbins[1::])]

############## for linear bins now
radial_linbins = radii_list_linbins

areabins_linbins = np.pi*radial_linbins**2
areabins1_regbins = areabins_linbins[1:]-areabins_linbins[:-1]

radii_list_linbins = [(a + b) /2. for a, b in zip(radii_list_linbins[::], radii_list_linbins[1::])]
'''

plt.figure()
#plt.plot(radii_list, cross_corr_func,'g-', linewidth = 2, label = "Using Local Mean")
plt.plot(radii_list_logbins[1:], cross_corr_func_predict_logbins, 'b+', markersize = 30)
plt.plot(radii_list_logbins[1:], cross_corr_func_predict_logbins,'r-', linewidth = 2, label = "Log bins")
plt.plot(radii_list_linbins[1:], cross_corr_func_predict_linbins, 'm+', markersize = 30)
plt.plot(radii_list_linbins[1:], cross_corr_func_predict_linbins,'g-', linewidth = 2, label = "Linear Bins")
plt.legend()
plt.xlabel("Upper Radial Bin [Mpc]")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation with Predicted mean")
#plt.xscale('log')
plt.xlim((np.amin(radii_list_logbins) - 30 ,np.amax(radii_list_logbins)+100))
plt.figtext(0,0,"z-bin weighted simulated data")
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/CCF_simulated_predicted_zbin_weighted_binning.png' )
plt.show()

plt.figure()
plt.plot(radii_list_logbins[1:], cross_corr_func_logbins, 'b+', markersize = 30)
plt.plot(radii_list_logbins[1:], cross_corr_func_logbins,'r-', linewidth = 2, label = "Log Bins")
plt.plot(radii_list_linbins[1:], cross_corr_func_linbins, 'm+', markersize = 30)
plt.plot(radii_list_linbins[1:], cross_corr_func_linbins,'g-', linewidth = 2, label = "Linear Bins")
plt.legend()
plt.xlabel("Upper Radial Bin [Mpc]")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation with Local Mean")
#plt.xscale('log')
plt.xlim((np.amin(radii_list_logbins) - 30 ,np.amax(radii_list_logbins)+100))
plt.figtext(0,0,"z-bin weighted simulated data")
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/CCF_simulated_local_zbin_weighted_binning.png' )
plt.show()

####################

fig, ax = plt.subplots()
ax1 = plt.subplot(211)
ax1.set_title('Plot of average number of galaxies around simulated RM source (logscale)')
ax1.bar(bin_edges_linbins[:-1], avg_gal_num_linbins, width=[(bin_edges_linbins[j+1]-bin_edges_linbins[j]) for j in range(len(bin_edges_linbins)-1)], alpha = 0.5, label = "linear bins", color = 'b')
ax1.bar(bin_edges_logbins[:-1], avg_gal_num_logbins, width=[(bin_edges_logbins[j+1]-bin_edges_logbins[j]) for j in range(len(bin_edges_logbins)-1)], alpha = 0.7, label = "log bins", color = 'r')

ax1.set_ylabel('Galaxy counts [log]')
ax1.set_xlabel('Radial Bins [log]')
ax1.legend(loc = 'upper left')
ax1.grid(True)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2 = plt.subplot(212)
ax2.set_title('Plot of average number of galaxies around simulated RM source')
ax2.bar(bin_edges_logbins[:-1], avg_gal_num_logbins, width=[(bin_edges_logbins[j+1]-bin_edges_logbins[j]) for j in range(len(bin_edges_logbins)-1)], alpha = 0.5, label = "log bins", color = 'r')
ax2.bar(bin_edges_linbins[:-1], avg_gal_num_linbins, width=[(bin_edges_linbins[j+1]-bin_edges_linbins[j]) for j in range(len(bin_edges_linbins)-1)], alpha = 0.7, label = "linear bins", color = 'b')
ax2.set_ylabel('Galaxy counts [#]')
ax2.set_xlabel('Radial Bins')
ax2.legend(loc = 'upper left')
ax2.grid(True)
fig.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/simulated_average_galaxies_binning_test_binplot.png' )
#fig.show()




'''
np.save("cross_corr_local_simulated_file.npy", cross_corr_func)
np.save("cross_corr_predict_simulated_file.npy", cross_corr_func_predict)
'''


print " "
print "ALL DONE, go check your plots folder :)"
print " "

