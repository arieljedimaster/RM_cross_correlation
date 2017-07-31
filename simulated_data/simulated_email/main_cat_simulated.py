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
print "Uploading Catalogues...."

#Uploading simulated data

RA_RM,Dec_RM,RM, RM_err,gal_sort,RM_simulated,simulated_pix = np.loadtxt('simulated_RM_data_zbin_weighted.txt', unpack = True)

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
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/testing/RM_vs_numgal_CCFinput_zbin_weighted.png')


print ' '
print "**** Calculating PREDICTED galaxy number density grid....."
print ' '

sky_frac = 0.75


radii_list= np.logspace(1.,3.4,9)
radii_list[0] = 0.


print "   "
print "The radii list we're using in kpc is: ", radii_list
print "   "



#weight average number of galaxies per radial bin over all z bin
avegal_wt = bf.predict_numbdense(total_gal_in_z,z_mean,radii_list,sky_frac)


print ' '
print "***** Calculating the deg --> kpc conversion factor for each galaxy....."
print ' '

kpc2deg_list = bf.gal_conv_fact(z_bin_list,z_mean)





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

dist_matrix_deg = np.load("dist_matrix_deg_simulated.npy")
dist_matrix_kpc = np.load('dist_matrix_kpc_new_simulated.npy')
dist_indeces = np.load('dist_indeces_new_simulated.npy')
dist_z_split = np.load('dist_z_split_new_simulated.npy')

t2=time.time()
print "This took " + str((t2-t1)/60.) + " minutes."

'''

print ' '
print "******** Cross Correlating....."
print ' '

t3=time.time()


cross_corr_func = bf.cross_corr(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split)


cross_corr_func_predict = bf.cross_corr_predict(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split,avegal_wt)


t4=time.time()

print "Cross correlating took " + str((t4-t3)/60.) + " minutes."

radial_bins = radii_list

areabins = np.pi*radial_bins**2
areabins1 = areabins[1:]-areabins[:-1]

radii_list = [(a + b) /2. for a, b in zip(radii_list[::], radii_list[1::])]


plt.figure()
#plt.plot(radii_list, cross_corr_func,'g-', linewidth = 2, label = "Using Local Mean")
plt.plot(radii_list, cross_corr_func_predict, 'b+', markersize = 30)
plt.plot(radii_list, cross_corr_func_predict,'r-', linewidth = 2, label = "Using Predicted Mean")
plt.legend()
plt.xlabel("Radial Bins [Mpc]")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation")
plt.xscale('log')
plt.xlim((np.amin(radii_list) - 30 ,np.amax(radii_list)+100))
plt.figtext(0,0,"z-bin weighted simulated data")
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/testing/CCF_simulated_predicted_zbin_weighted.png' )
plt.show()

plt.figure()
plt.plot(radii_list, cross_corr_func, 'b+', markersize = 30)
plt.plot(radii_list, cross_corr_func,'r-', linewidth = 2, label = "Using Local Mean")
plt.legend()
plt.xlabel("Radial Bins [Mpc]")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation")
plt.xscale('log')
plt.xlim((np.amin(radii_list) - 30 ,np.amax(radii_list)+100))
plt.figtext(0,0,"z-bin weighted simulated data")
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/testing/CCF_simulated_local_zbin_weighted.png' )
plt.show()

np.save("cross_corr_local_simulated_file.npy", cross_corr_func)
np.save("cross_corr_predict_simulated_file.npy", cross_corr_func_predict)



print " "
print "ALL DONE, go check your plots folder :)"
print " "

