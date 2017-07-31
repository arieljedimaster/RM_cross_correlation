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
#import dist_matrix_code as dmc
import dist_matrix_code_tessa as dmcT

hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)

############################################
#import data

#data directory
pwd_data = '/Users/arielamaral/Documents/AST430/Data/'

pwd_RMtxtfile = '/Users/arielamaral/Documents/AST430/Code/Code_new_cat/'

print ' '
print "Uploading Catalogues...."

#RM catalogue Taylor et al.

#RA_hms, err_RA, Dec_hms, err_DEC, b, I, S_I, S_I_err, ave_P, ave_P_err, pol_percent, pol_percent_err,RM, RM_err = np.genfromtxt(pwd_data + 'RMcat_taylor.csv', delimiter = ',', unpack=True, dtype=None)

#RM Hammond et al. catalogue

RM_hammond = fits.open(pwd_data+'rm_redshift_catalog.fits')
RM_hammond_header = RM_hammond[1].header
RM_hammond_data = RM_hammond[1].data

RA_RM = RM_hammond_data['RA_DEG_J2000']
Dec_RM = RM_hammond_data['DEC_DEG_J2000']

RM = RM_hammond_data['NVSS_RM ']

RM_err = RM_hammond_data['NVSS_RM_ERR']

RM_z = RM_hammond_data['SELECTED_REDSHIFT']



#deleting all sources which are less than redshift 0.5

print "Number of RM sources which have redshifts greater than z = 0.5: ", len(np.where(RM_z > 0.5)[0])


RM_z = RM_z[np.where(RM_z>0.5)[0]]

RA_RM = RA_RM[np.where(RM_z>0.5)[0]]

Dec_RM = Dec_RM[np.where(RM_z>0.5)[0]]

RM = RM[np.where(RM_z>0.5)[0]]

RM_err = RM_err[np.where(RM_z>0.5)[0]]



#WISE/superCOSMOS

WISE_cat = fits.open(pwd_data+'WISExSCOS.photoZ.MAIN.fits')
WISE_header = WISE_cat[1].header
WISE_data = WISE_cat[1].data

gal_RA = WISE_data['ra_WISE ']
gal_Dec = WISE_data['dec_WISE']
gal_photo_z = WISE_data ['zANNz   ']



#########################################################
#specifying healpy resolution

res=10 #SWITCH THIS TO CHANGE UP RESOLUTION
nside=2**res
npix=12*nside**2



print "total number of galaxy sources: ", len(gal_RA)

gal_RA, gal_Dec, gal_photo_z = bf.z_reduc(gal_RA, gal_Dec, gal_photo_z)

print "total number of galaxy sources after filtering out 0.1 < z < 0.5: ", len(gal_RA)
print " "



############################################################

print ' '
print "* Converting RM catalogue coordinates from HMS to degrees...."
print ' '
'''
RA_hms = RA_hms[1:]
Dec_hms = Dec_hms[1:]
RM = RM[1:]
RM = RM.astype(np.float) #changing the read in RM vals to floats
RM_err = RM_err[1:]
RM_err = RM_err.astype(np.float)

RA_RM, Dec_RM = np.loadtxt(pwd_RMtxtfile+'RA_DEC_degrees.txt',unpack=True)
'''
############## change n and num_of_sources for testing purposes

n = 100

num_of_sources = len(RA_RM)

print str(n), " runs for ", str(num_of_sources), " random sources"


#log spaced bins

radii_list_logbins= np.logspace(1.,3.4,9)
radii_list_logbins[0] = 0.

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



for i in np.arange(2, n):
	if (i % 5 == 0) & (i!=0):
		print "On random run number " + str(i)
	np.random.shuffle(RM)

	#tessa's random edit

	l=np.random.rand(num_of_sources)*360.
	#why * 15 + 50??
	b=abs(np.random.randn(num_of_sources)*15+50)
	#generating random values between -1 and 1
	rrh=np.random.randn(num_of_sources)
	rrh=rrh/abs(rrh)
	#making some of the b's negative
	b=b*rrh
	#changing all values less than -90 and +90  to the proper value within range
	b[b<-90]=b[b<-90]+90
	b[b>90]=b[b>90]-90
	#masking out the galactic plane by placing all vals between -20 and 20 outside of that range
	b[(b<20)&(b>0)]=b[(b<20)&(b>0)]+20
	b[(b>-20)&(b<0)]=b[(b>-20)&(b<0)]-20

	coord_randg=SkyCoord(l*u.deg,b*u.deg,frame='galactic')
	coord_rand=coord_randg.fk5
	RA_RM=coord_rand.ra.wrap_at(180*u.degree).value
	Dec_RM = coord_rand.dec.degree

	coord_gal = SkyCoord(gal_RA*u.deg,gal_Dec*u.deg,frame='fk5')
	rra_gal=coord_gal.ra.wrap_at(180*u.degree).value


	'''
	l_rand = np.random.uniform(low=0.0, high=360.0, size=num_of_sources)
	#starting from 20 degrees so we don't include the galactic plane
	b_rand = np.random.uniform(low= 20.0, high=90.0, size=num_of_sources)
	#creating a negative array
	neg_array = np.random.randint(2, size = num_of_sources)
	#making all zeros negative ones
	np.place(neg_array, neg_array == 0, -1)
	#making some random b vals negative:
	b_rand = b_rand*neg_array

	#creaing l and b skycoord object
	lb_skycoord = SkyCoord(l = l_rand*u.degree, b = b_rand*u.degree, frame = 'galactic')
	#print "lb_skycoord: ", lb_skycoord
	#transforming to RA/Dec
	radec_skycoord = lb_skycoord.icrs
	RA_RM = radec_skycoord.ra.degree
	#print "RA_RM: ", RA_RM
	Dec_RM = radec_skycoord.dec.degree
	#because the Hammond catalogue doesnt have any sources below Dec = -40 degrees
	#Dec_RM[Dec_RM<-40]=Dec_RM[Dec_RM<-40]*(-1)
	#print "Dec_RM: ", Dec_RM
	'''

	#now ready to go with the random stuff

	#saving the RA/Dec/RM using index i
	np.savetxt('random_RM_sources/hammond_random_sources/random_RM_RA_DEC_run_'+str(i)+'.txt', np.c_[RA_RM, Dec_RM, RM[:num_of_sources]])
	print "txt file output: ", np.c_[RA_RM, Dec_RM, RM[:num_of_sources]]

	#dist_matrix_deg_z, dist_matrix_kpc_z, dist_indeces_z, dist_z_split_z = dmc.distance_func_zbins(RA_RM, Dec_RM, kpc2deg_list, radii_list_logbins, gal_RA_split, gal_Dec_split, z_bin_list, z_mean)

	#dist_matrix_deg_z_max, dist_matrix_kpc_z_max, dist_indeces_z_max, dist_z_split_z_max = dmc.distance_func_zbins_max(RA_RM, Dec_RM, kpc2deg_list, radii_list_logbins, gal_RA_split, gal_Dec_split, z_bin_list, z_mean)

	dist_matrix_deg_tessa, dist_matrix_kpc_tessa, dist_indeces_tessa, dist_z_split_tessa = dmcT.distance_func_tessa3(coord_rand,coord_gal,RA_RM,rra_gal, kpc2deg_list, radii_list_logbins, z_bin_list)

	#dist_matrix_deg_tessa, dist_matrix_kpc_tessa, dist_indeces_tessa, dist_z_split_tessa = bf.distance_func_tessa(RA_RM, Dec_RM, gal_RA, gal_Dec, kpc2deg_list, radii_list_logbins, z_bin_list)
	'''

	bins1 = np.linspace(0,np.amax(dist_matrix_kpc_tessa[0]), 20)


	plt.figure()

	plt.hist(dist_matrix_kpc_tessa[0],bins=bins1, histtype='bar', facecolor = 'r', alpha = 0.5, label = 'distance func')
	plt.hist(dist_matrix_kpc_z_max[0],bins=bins1, histtype='bar', facecolor = 'y', alpha = 0.5, label = 'distance func zbin - lower z')
	plt.hist(dist_matrix_kpc_z[0], bins=bins1,histtype='bar', facecolor = 'b', alpha = 0.5, label = 'distance func zbin - zmean')
	plt.ylabel('number of galaxies')
	plt.xlabel('Distance from the random RM source [kpc]')
	plt.title('Looking at the difference between the two distance matrix functions')
	plt.legend(loc ='upper left')
	plt.show()


	plt.figure()

	plt.hist(dist_matrix_kpc_tessa[0],bins=radii_list_linbins, histtype='bar', facecolor = 'r', alpha = 0.5, label = 'distance func')
	plt.hist(dist_matrix_kpc_z_max[0], bins=radii_list_linbins,histtype='bar', facecolor = 'y', alpha = 0.5, label = 'distance func zbin - lower z')
	plt.hist(dist_matrix_kpc_z[0], bins=radii_list_linbins,histtype='bar', facecolor = 'b', alpha = 0.5, label = 'distance func zbin - z_mean')

	plt.ylabel('number of galaxies')
	plt.xlabel('Distance from the random RM source [kpc]')
	plt.title('Looking at the difference between the two distance matrix functions')
	plt.legend(loc = 'upper left')
	plt.show()
	'''


	
	np.save("random_matrices/hammond_matrices/dist_matrix_deg_run_"+str(i)+".npy", dist_matrix_deg_tessa)
	np.save('random_matrices/hammond_matrices/dist_matrix_kpc_run_'+str(i)+'.npy',dist_matrix_kpc_tessa)
	np.save('random_matrices/hammond_matrices/dist_indeces_new_run_'+str(i)+'.npy',dist_indeces_tessa)
	np.save('random_matrices/hammond_matrices/dist_z_split_new_run_'+str(i)+'.npy',dist_z_split_tessa)
	

print " "
print "ALL DONE generating random distance matrices and RM sources :)"
print " "





