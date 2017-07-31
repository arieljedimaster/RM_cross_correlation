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

hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)

############################################
#import data

#data directory
pwd_data = '/Users/arielamaral/Documents/AST430/Data/'

#pwd to the taylor RM text file with the RA/Dec/RM of all the sources

pwd_RMtxtfile = '/Users/arielamaral/Documents/AST430/Code/Code_new_cat/'


pwd_hammond = '/Users/arielamaral/Documents/AST430/Code/Code_new_cat/hammond/'

print ' '
print "Uploading Catalogues...."

#RM Hammond et al. catalogue

RM_hammond = fits.open(pwd_data+'rm_redshift_catalog.fits')
RM_hammond_header = RM_hammond[1].header
RM_hammond_data = RM_hammond[1].data

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




#WISE/superCOSMOS

WISE_cat = fits.open(pwd_data+'WISExSCOS.photoZ.MAIN.fits')
WISE_header = WISE_cat[1].header
WISE_data = WISE_cat[1].data

gal_RA = WISE_data['ra_WISE ']
gal_Dec = WISE_data['dec_WISE']
gal_photo_z = WISE_data ['zANNz   ']



#########################################################
#specifying healpy resolution
'''
res=10 #SWITCH THIS TO CHANGE UP RESOLUTION
nside=2**res
npix=12*nside**2
'''

#niels' resolution
nside= 128
npix = 12*nside**2
res = int(np.log(nside)/np.log(2))



print "total number of galaxy sources: ", len(gal_RA)

gal_RA, gal_Dec, gal_photo_z = bf.z_reduc(gal_RA, gal_Dec, gal_photo_z)

print "total number of galaxy sources after filtering out 0.1 < z < 0.5: ", len(gal_RA)
print " "




############################################################

print ' '
print "* Converting RM catalogue coordinates from HMS to degrees...."
print ' '

############## change n and num_of_sources for testing purposes

n = 100

num_of_sources = len(RA_RM_hammond)

print str(n), " runs for ", str(num_of_sources), " random sources"


#log spaced bins

radii_list_logbins= np.logspace(1.,3.4,9)
radii_list_logbins = np.concatenate(([0.],radii_list_logbins))

#linear spaced bins

radii_list_linbins = np.linspace(0., 2511.88643150958, 9)

print "   "
print "The log binned radii list we're using in kpc is: ", radii_list_logbins
print "The linear binned radii list we're using in kpc is: ", radii_list_linbins
print "   "




print ' '
print "**** Starting to generate distance matrices for different random realizations...."
print ' '


z_split, z_mean, gal_RA_split, gal_Dec_split, z_bin_list = bf.photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)
#print "zmean is: ", z_mean

num_dens_grid = bf.healpy_grid(res,gal_RA,gal_Dec)

kpc2deg_list = bf.gal_conv_fact(z_bin_list,z_mean)

total_gal_in_z = bf.total_num_gals_in_z(z_bin_list)


print ' '
print "****** Calculating Mask which gets rid of RM in cells with zero galaxies and those which are too close to the edges....."
print ' '


RA_RM_hammond, Dec_RM_hammond, RM_hammond = bf.mask2(num_dens_grid, RA_RM_hammond, Dec_RM_hammond, RM_hammond, radii_list_linbins)


print ' '
print "****** Uploading random data....."
print ' '

sky_frac = 0.75

#avegal_wt_logbins = bf.predict_numbdense(total_gal_in_z,z_mean,radii_list_logbins,sky_frac)

#avegal_wt_linbins = bf.predict_numbdense(total_gal_in_z,z_mean,radii_list_linbins,sky_frac)


#uploading hammond distance matrices


dist_matrix_deg_hammond = np.load(pwd_hammond + "dist_matrix_deg_hammond.npy")
dist_matrix_kpc_hammond = np.load(pwd_hammond + 'dist_matrix_kpc_hammond.npy')
dist_indeces_hammond = np.load(pwd_hammond + 'dist_indeces_hammond.npy')
dist_z_split_hammond = np.load(pwd_hammond + 'dist_z_split_hammond.npy')


#hammond galaxy histogram



hammond_gal_hist_linbins, bin_edges_linbins, hammond_gal_std_linbins, total_gal_num_linbins = bf.gal_sources_hist(RA_RM_hammond, Dec_RM_hammond, RM_hammond, dist_matrix_kpc_hammond, radii_list_linbins)

hammond_gal_hist_logbins, bin_edges_logbins, hammond_gal_std_logbins, total_gal_num_logbins = bf.gal_sources_hist(RA_RM_hammond, Dec_RM_hammond, RM_hammond, dist_matrix_kpc_hammond, radii_list_logbins)


hammond_1stbin_linbins = total_gal_num_linbins[:,0]

hammond_1stbin_logbins = total_gal_num_logbins[:,0]




'''
plt.figure()
hp.cartview(num_dens_grid)
plt.plot(bf.CartviewRA(RA_RM_hammond), Dec_RM_hammond, 'o', label = "Hammond RM Sources")
'''



rand_gal_hist_all_linbins = np.zeros((n, len(radii_list_linbins) - 1))

rand_gal_hist_std_linbins = np.zeros((n, len(radii_list_linbins) - 1))


rand_gal_hist_all_logbins = np.zeros((n, len(radii_list_logbins) - 1))

rand_gal_hist_std_logbins = np.zeros((n, len(radii_list_logbins) - 1))



rand_1stbin_logbins = np.zeros((3, len(RA_RM_hammond)))

rand_1stbin_linbins = np.zeros((3, len(RA_RM_hammond)))

for i in np.arange(0,n):
	if (i % 5 == 0) & (i!=0):
		print "On random run number " + str(i)

	RA_RM, Dec_RM, RM = np.loadtxt('random_RM_sources/hammond_random_sources/random_RM_RA_DEC_run_'+str(i)+'.txt', unpack = True)
	#print "txt file output: ", np.c_[RA_RM, Dec_RM, RM]

	RA_RM = RA_RM[:len(RA_RM_hammond)]
	Dec_RM = Dec_RM[:len(RA_RM_hammond)]
	RM = RM[:len(RA_RM_hammond)]

	#plt.plot(bf.CartviewRA(RA_RM), Dec_RM, 'o', alpha = 0.5, label = "Random Data # " +str(i))

	dist_matrix_deg = np.load("random_matrices/hammond_matrices/dist_matrix_deg_run_"+str(i)+".npy")[:len(RA_RM_hammond)]
	dist_matrix_kpc  = np.load('random_matrices/hammond_matrices/dist_matrix_kpc_run_'+str(i)+'.npy')[:len(RA_RM_hammond)]
	dist_indeces = np.load('random_matrices/hammond_matrices/dist_indeces_new_run_'+str(i)+'.npy')[:len(RA_RM_hammond)]
	dist_z_split = np.load('random_matrices/hammond_matrices/dist_z_split_new_run_'+str(i)+'.npy')[:len(RA_RM_hammond)]

	if i < 2:
		print i
		single_source_hist_linbins, bin_edges_linbins, single_source_std_linbins,total_gal_num_rand_linbins = bf.gal_sources_hist(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_linbins)
		rand_gal_hist_all_linbins[i] = single_source_hist_linbins
		rand_gal_hist_std_linbins[i] = single_source_std_linbins

		rand_1stbin_linbins[i] = total_gal_num_rand_linbins[:,0]

		single_source_hist_logbins, bin_edges_logbins, single_source_std_logbins, total_gal_num_rand_logbins = bf.gal_sources_hist(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_logbins)
		rand_gal_hist_all_logbins[i] = single_source_hist_logbins
		rand_gal_hist_std_logbins[i] = single_source_std_logbins

		rand_1stbin_logbins[i] = total_gal_num_rand_logbins[:,0]
	
	else:

		single_source_hist_linbins, bin_edges_linbins, single_source_std_linbins, _ = bf.gal_sources_hist(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_linbins)
		rand_gal_hist_all_linbins[i] = single_source_hist_linbins
		rand_gal_hist_std_linbins[i] = single_source_std_linbins

		single_source_hist_logbins, bin_edges_logbins, single_source_std_logbins, _ = bf.gal_sources_hist(RA_RM, Dec_RM, RM, dist_matrix_kpc, radii_list_logbins)
		rand_gal_hist_all_logbins[i] = single_source_hist_logbins
		rand_gal_hist_std_logbins[i] = single_source_std_logbins
	


'''
plt.title("Random vs. Hammond Source Locations")	
plt.xlabel("Right Ascension [degrees]")
plt.ylabel("Declination [degrees]")
plt.legend()
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/cross_correlation/wise_hammond/random_run_locations.png')
plt.show()
'''



rand_gal_linbins_err = np.std(rand_gal_hist_all_linbins, axis = 0)

rand_gal_avg_linbins = np.average(rand_gal_hist_all_linbins, axis = 0)

rand_gal_avg_std_linbins = np.average(rand_gal_hist_std_linbins, axis = 0)





rand_gal_logbins_err = np.std(rand_gal_hist_all_logbins, axis = 0)

rand_gal_avg_logbins = np.average(rand_gal_hist_all_logbins, axis = 0)

rand_gal_avg_std_logbins = np.average(rand_gal_hist_std_logbins, axis = 0)


###############################
# Counts in the first bin

max_count1 = np.amax(rand_1stbin_linbins)
max_count2 = np.amax(hammond_1stbin_linbins)
max_count = np.amax([max_count1,max_count2])

bins_1stbin_linbins = np.arange(0,max_count+2)

print "lin bins: ", bins_1stbin_linbins


plt.figure()
plt.title('Normalized Histogram of Galaxy Counts in the First Linear Radial Bin')
plt.hist(rand_1stbin_linbins[0], bins =bins_1stbin_linbins, normed = True,  color = 'b', alpha = 0.5, label = 'Random Run 1')
plt.hist(rand_1stbin_linbins[1], bins =bins_1stbin_linbins, normed = True, color = 'y', alpha = 0.5, label = 'Random Run 2')
#plt.hist(rand_1stbin_linbins[2], bins =bins_1stbin, normed = True, color = 'g', alpha = 0.5, label = 'Random Run 3')
plt.hist(hammond_1stbin_linbins, bins =bins_1stbin_linbins, normed = True, color = 'r', alpha = 0.5, label = 'Hammond Data')
plt.ylabel('Number of RM Sources')
plt.xlabel('Number of Galaxies Around RM Source')
plt.legend()
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/cross_correlation/wise_hammond/firstbin_galnum_linbins.png' )


max_count1 = np.amax(rand_1stbin_logbins)
max_count2 = np.amax(hammond_1stbin_logbins)
max_count = np.amax([max_count1,max_count2])

bins_1stbin_logbins = np.arange(0,max_count+2)

print "lin bins: ", bins_1stbin_logbins

plt.figure()
plt.title('Normalized Histogram of Galaxy Counts in the First Log Radial Bin')
plt.hist(rand_1stbin_logbins[0], bins =bins_1stbin_logbins, normed = True,  color = 'b', alpha = 0.5, label = 'Random Run 1')
plt.hist(rand_1stbin_logbins[1], bins =bins_1stbin_logbins, normed = True, color = 'y', alpha = 0.5, label = 'Random Run 2')
#plt.hist(rand_1stbin_linbins[2], bins =bins_1stbin, normed = True, color = 'g', alpha = 0.5, label = 'Random Run 3')
plt.hist(hammond_1stbin_logbins, bins =bins_1stbin_logbins, normed = True, color = 'r', alpha = 0.5, label = 'Hammond Data')
plt.ylabel('Number of RM Sources')
plt.xlabel('Number of Galaxies Around RM Source')
plt.legend()
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/cross_correlation/wise_hammond/firstbin_galnum_logbins.png' )






############################### PLOTTING


#bins1 = np.linspace(0,np.amax(dist_matrix_kpc_tessa[0]), 20)


fig, ax = plt.subplots()
ax1 = plt.subplot(211)
ax1.set_title('Avg gal # around RM and Random Sources for ' +str(n)+' of ' + str(len(RA_RM_hammond))+ ' sources [linear bins]')
ax1.bar(bin_edges_linbins[:-1], hammond_gal_hist_linbins, width=[(bin_edges_linbins[j+1]-bin_edges_linbins[j]) for j in range(len(bin_edges_linbins)-1)], alpha = 0.5, label = "Hammond RM Data", color = 'r')
ax1.bar(bin_edges_linbins[:-1], rand_gal_avg_linbins, width=[(bin_edges_linbins[j+1]-bin_edges_linbins[j]) for j in range(len(bin_edges_linbins)-1)], alpha = 0.5, label = "Random Locations", color = 'b')
mid = 0.5*(bin_edges_linbins[1:] + bin_edges_linbins[:-1])
ax1.errorbar(mid, rand_gal_avg_linbins, yerr=rand_gal_linbins_err, fmt='none', color='k')
ax1.set_ylabel('Galaxy counts')
ax1.set_xlabel('Radial Bins')
ax1.legend(loc = 'upper left')
ax1.grid(True)
#ax1.set_xscale('log')
#ax1.set_yscale('log')



ax2 = plt.subplot(212)
ax2.set_title('Avg gal # around RM and Random Sources for ' +str(n)+' of ' + str(len(RA_RM_hammond))+ ' sources [log bins]')
ax2.bar(bin_edges_logbins[:-1], hammond_gal_hist_logbins, width=[(bin_edges_logbins[j+1]-bin_edges_logbins[j]) for j in range(len(bin_edges_logbins)-1)], alpha = 0.5, label = "Hammond RM Data", color = 'r')
ax2.bar(bin_edges_logbins[:-1], rand_gal_avg_logbins, width=[(bin_edges_logbins[j+1]-bin_edges_logbins[j]) for j in range(len(bin_edges_logbins)-1)], alpha = 0.5, label = "Random Locations", color = 'b')
mid = 0.5*(bin_edges_logbins[1:] + bin_edges_logbins[:-1])
ax2.errorbar(mid, rand_gal_avg_logbins, yerr=rand_gal_logbins_err, fmt='none', color='k')
ax2.set_ylabel('Galaxy counts [log]')
ax2.set_xlabel('Radial Bins [log]')
ax2.legend(loc = 'upper left')
ax2.grid(True)
ax2.set_xscale('log')
ax2.set_yscale('log')

fig.savefig('/Users/arielamaral/Documents/AST430/Plots/cross_correlation/wise_hammond/avg_gal_binning_datVSrandom_linlog.png' )
fig.show()


####################################
#standard deviations plot

fig, ax = plt.subplots()
ax1 = plt.subplot(211)
ax1.set_title('STD of avg gal # around RM and Random Sources for ' +str(n)+' of ' + str(len(RA_RM_hammond))+ ' sources [lin bins])')
ax1.bar(bin_edges_linbins[:-1], hammond_gal_std_linbins, width=[(bin_edges_linbins[j+1]-bin_edges_linbins[j]) for j in range(len(bin_edges_linbins)-1)], alpha = 0.5, label = "Hammond RM Data", color = 'r')
ax1.bar(bin_edges_linbins[:-1], rand_gal_avg_std_linbins, width=[(bin_edges_linbins[j+1]-bin_edges_linbins[j]) for j in range(len(bin_edges_linbins)-1)], alpha = 0.5, label = "Random Locations", color = 'b')
#mid = 0.5*(bin_edges_linbins[1:] + bin_edges_linbins[:-1])
#ax1.errorbar(mid, rand_gal_hist_linbins, yerr=rand_gal_std_linbins, fmt='none', color='k')
ax1.set_ylabel('Std Galaxy counts')
ax1.set_xlabel('Radial Bins')
ax1.legend(loc = 'upper left')
ax1.grid(True)



ax2 = plt.subplot(212)
ax2.set_title('STD of avg gal # around RM and Random Sources for ' +str(n)+' of ' + str(len(RA_RM_hammond))+ ' sources [log bins]')
ax2.bar(bin_edges_logbins[:-1], hammond_gal_std_logbins, width=[(bin_edges_logbins[j+1]-bin_edges_logbins[j]) for j in range(len(bin_edges_logbins)-1)], alpha = 0.5, label = "Hammond RM Data", color = 'r')
ax2.bar(bin_edges_logbins[:-1], rand_gal_avg_std_logbins, width=[(bin_edges_logbins[j+1]-bin_edges_logbins[j]) for j in range(len(bin_edges_logbins)-1)], alpha = 0.5, label = "Random Locations", color = 'b')
#mid = 0.5*(bin_edges_logbins[1:] + bin_edges_logbins[:-1])
#ax2.errorbar(mid, rand_gal_hist_logbins, yerr=rand_gal_std_logbins, fmt='none', color='k')
ax2.set_ylabel('Std Galaxy counts [log]')
ax2.set_xlabel('Radial Bins [log]')
ax2.legend(loc = 'upper left')
ax2.grid(True)
ax2.set_xscale('log')
ax2.set_yscale('log')

fig.savefig('/Users/arielamaral/Documents/AST430/Plots/cross_correlation/wise_hammond/std_gal_binning_datVSrandom_linlog.png' )
fig.show()





