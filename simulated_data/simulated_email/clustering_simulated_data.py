# fake data and clustering

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
import matplotlib.mlab as mlab
import operator

import time

import sys
sys.path.insert(0, '/Users/arielamaral/Documents/AST430/Code/Code_new_cat')
from bfield_functions_cat import *


def RM_grid(RM,res_map,RA_RM,Dec_RM,subave=True):

    nside_map = 2**res_map
    npix_map = 12*nside_map**2
    #mean subtracted
    Dec_RM1=(np.pi/180.)*(90.- Dec_RM)
    RA_RM1=(np.pi/180.)*(RA_RM)
    pix_RM = hp.ang2pix(nside_map,Dec_RM1,RA_RM1)
    vals=np.zeros(npix_map)
    RM_avg_vals, bin_edges, bin_num = st.binned_statistic(pix_RM, RM, statistic='mean', bins=np.arange(max(pix_RM)), range=None)
    RM_avg_vals[np.isnan(RM_avg_vals)]=0.
    RM_mean = np.mean(RM) # do mean
    vals[bin_num] = RM_avg_vals
    vals[vals == 0.] = np.nan

    if subave==True:
        vals -= vals.mean()

    return vals

def num_dens_reduce_compare_plot(gal_sort,gal_sort_reduced):

    plt.subplot(2, 1, 1)
    plt.hist(gal_sort, 20, normed=1, facecolor='green', alpha=0.5)
    plt.title('Histogram 1: before reducing')
    plt.ylabel('Galaxy counts [#]')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title('Histogram 2: after reducing')
    plt.hist(gal_sort_reduced, 20, normed=1, facecolor='red', alpha=0.5)
    plt.ylabel('Galaxy counts [#]')
    plt.grid(True)
    plt.show()
    return


################################################################################################################################
################################################################################################################################


#upload WISE catalogue

WISE_cat = fits.open(pwd_data+'WISExSCOS.photoZ.MAIN.fits')
WISE_header = WISE_cat[1].header
WISE_data = WISE_cat[1].data

gal_RA = WISE_data['ra_WISE ']
gal_Dec = WISE_data['dec_WISE']
gal_photo_z = WISE_data ['zANNz   ']

#what resolution should we use?

res=10 #SWITCH THIS TO CHANGE UP RESOLUTION
nside=2**res
npix=12*nside**2



print "total number of galaxy sources: ", len(gal_RA)

gal_RA, gal_Dec, gal_photo_z = z_reduc(gal_RA, gal_Dec, gal_photo_z)

print "total number of galaxy sources after filtering out 0.1 < z < 0.5: ", len(gal_RA)
print " "

z_split, z_mean, gal_RA_split, gal_Dec_split, z_bin_list = photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)



radii_list = np.logspace(1.,3.4,9)
radii_list[0] = 0.
arcsec_per_kpc =cosmo1.arcsec_per_kpc_comoving(z_mean).value
r_deg = (radii_list[np.newaxis,:] * arcsec_per_kpc[:,np.newaxis])/3600.



#making number density grid

num_dens_total = healpy_grid2(res,gal_RA,gal_Dec)#np.zeros(npix)


#total number density grid

#setting all zero pixels (the galactic plane) to NaN
num_dens_total[num_dens_total==0.] = np.nan
num_dens_total_masked = hp.pixelfunc.ma(num_dens_total)
num_dens_total_masked.mask = np.isnan(num_dens_total_masked)


#z-weighted number density grids
num_dens_zbin = np.zeros((4,npix))
num_dens_zbin_masked=[]
z_weight = (1.+np.array(z_mean))**(-2.)
num_dens_zbin_total = np.zeros(npix)

for i in np.arange(0,len(z_split)):
    #dont want 'nan's' though right now
    num_dens_grid_split = healpy_grid2(res,gal_RA_split[i],gal_Dec_split[i])
    #subtract off the mean
    num_dens_grid_split = hp.ma(num_dens_grid_split)
    num_dens_grid_split.mask = num_dens_total_masked.mask
    #weighting it by z-bins:
    num_dens_grid_split *= z_weight[i]
    num_dens_zbin_masked.append(num_dens_grid_split)
    num_dens_zbin_total += num_dens_grid_split

#using the weighted zbin number density in this:
#num_dens_total_masked = num_dens_zbin_total_masked
num_dens_zbin_total[num_dens_zbin_total==0.] = np.nan
num_dens_total_masked = hp.pixelfunc.ma(num_dens_zbin_total)
num_dens_total_masked.mask = np.isnan(num_dens_zbin_total)




print "num_dens_total_masked.mask before: ", num_dens_total_masked.mask

total_gals = np.ma.sum(num_dens_total_masked)


print "total number of galaxies: ", total_gals


gal_prob = num_dens_total_masked.astype(float)/float(total_gals) #this is p



######################### real RM map:

#upload RM map

RA_hms, err_RA, Dec_hms, err_DEC, b, I, S_I, S_I_err, ave_P, ave_P_err, pol_percent, pol_percent_err,RM, RM_err = np.genfromtxt(pwd_data + 'RMcat_taylor.csv', delimiter = ',', unpack=True, dtype=None)


RA_hms = RA_hms[1:]
Dec_hms = Dec_hms[1:]
RM = RM[1:]
RM = RM.astype(np.float) #changing the read in RM vals to floats
RM_err = RM_err[1:]
RM_err = RM_err.astype(np.float)

RA_RM, Dec_RM = np.loadtxt('RA_DEC_degrees.txt',unpack=True)

num_dens_pix = np.arange(0,len(gal_prob))

#getting rid of masked indeces:
num_dens_pix = num_dens_pix[~num_dens_total_masked.mask]
gal_prob = gal_prob[~num_dens_total_masked.mask]

#generating simulated pixel indeces:
simulated_pix = np.random.choice(num_dens_pix, size=len(RA_RM), replace=False, p=gal_prob)

print "simulated pixel vals from np.random.choice: ", simulated_pix

###################### fake RM data:

#sorting the RM values
#this gives us the indeces yay

RM_abs = np.absolute(RM) #sorting by absolute value
RM_ii = np.argsort(RM_abs) #sorting the RM values, and returning the indeces


RM_sorted = RM[RM_ii] #this works, I checked

RM_err_sorted = RM_err[RM_ii]

#number density values at pixels we chose from numpy.random.choice

gal_reduced = num_dens_total_masked[simulated_pix]

gal_dic = dict(zip(simulated_pix,gal_reduced))

sort_ii = np.argsort(gal_reduced)

gal_sort = gal_reduced[sort_ii]


#is this right?

gal_ii = simulated_pix[sort_ii]

print " "
print "*** Checking that the sorted RM and gal arrays are doing what we want: "
print "galaxy indeces", gal_ii

print "RM_sorted: ", RM_sorted

print "gal_sort:", gal_sort
print " "


print "The number of cells in the galaxy, and RM catalogues are: ", len(gal_sort), len(RM_sorted)


RM_ii = gal_ii #which will give the locations of the galaxy density cells, its already been sorted.

def IndexToDeclRa(index, NSIDE):
    theta,phi=hp.pixelfunc.pix2ang(NSIDE,index)
    return -np.degrees(theta-pi/2.),np.degrees(pi*2.-phi)

RM_Dec_simulated, RM_RA_simulated = IndexToDeclRa(RM_ii,nside)


RM_simulated = RM_sorted
RM_err_simulated = RM_err_sorted

plt.figure()
plt.title('Number of galaxies at each simulated RM location')
plt.plot(gal_sort,abs(RM_simulated), 'bo')
plt.xlabel("Number of galaxies")
plt.ylabel("abs(RM)")
plt.grid()
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/testing/RM_vs_numgal_zbin_weighted.png')
plt.show()

#generating healpy version of RM data (to visualize)

npix = 12*nside**2

RM_healpy_simulated = np.zeros(len(num_dens_total_masked))
RM_healpy_simulated[RM_ii] = RM_simulated
RM_healpy_simulated[RM_healpy_simulated == 0.] = np.nan

RM_healpy_sim_masked = hp.pixelfunc.ma(RM_healpy_simulated)
RM_healpy_sim_masked.mask = np.isnan(RM_healpy_sim_masked)

cov_before  = np.ma.cov(np.absolute(RM_simulated), gal_sort)

corrcoeff_before = np.ma.corrcoef(np.absolute(RM_simulated), gal_sort)

corrcoeff_healpy = np.ma.corrcoef(np.absolute(RM_healpy_sim_masked), num_dens_total_masked)

print " "

print " **** Checking that the simulated RM and the galaxy number density are correlated: "

print ">using abs(RM)"

print "correlation matrix before putting into healpy arrays: ", corrcoeff_before

print "correlation matrix of healpy arrays: ", corrcoeff_healpy

print "covariance matrix of the original arrays: ", cov_before

print " "

#RM_healpy_sim_masked -= np.mean(RM_healpy_sim_masked)


print "RM_healpy_simulated: ", RM_healpy_sim_masked

#now we'll visualize the simulated data along with the galaxy number density which we based it on:

hp.visufunc.mollview(RM_healpy_sim_masked, sub = (2,1,1))
plt.title("Simulated RM Data")
hp.visufunc.mollview(num_dens_total_masked, sub =(2,1,2))
plt.title("Galaxy Number Density")
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/testing/simulated_mollview_zbin_weighted.png')
plt.show()
#now we can write the RM values + RA/Dec to a file to use in our code

np.savetxt('simulated_RM_data_zbin_weighted.txt', np.c_[RM_RA_simulated,RM_Dec_simulated,RM_simulated, RM_err_simulated,gal_sort,RM_simulated,simulated_pix])

print " "
print "*** Checking the following are the same: "
print "* should be the same* num_dens_total_masked[gal_ii] == gal_sort: ", num_dens_total_masked[gal_ii] == gal_sort

print "* should be the same* RM_healpy_sim_masked[gal_ii] = RM_healpy_sim_masked[RM_ii]: ", RM_healpy_sim_masked[gal_ii] == RM_healpy_sim_masked[RM_ii]

print "* should be the same* RM_healpy_sim_masked[gal_ii] = RM_simulated: ", RM_healpy_sim_masked[gal_ii] == RM_simulated
print " "

cross_corr_z=hp.sphtfunc.anafast(num_dens_total_masked, RM_healpy_sim_masked)
ctheta,cl_theta = CCF_theta(cross_corr_z,l_start=0,end_theta=r_deg.max(),theta_samples=radii_list.size)


plt.figure()
plt.plot(ctheta, cl_theta, 'b-', linewidth = 2, label = "num dens total")
plt.xlabel('Degrees')
plt.ylabel('$C_\ell$')
plt.legend(loc = 'upper left')
plt.grid()
plt.title('Image Cross Correlation, hp res: ' + str(res))
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/simulated_data/testing/CCI_simulated_zbin_weighted.png')
plt.show()


print "Done!"





