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
import scipy as scipy
import time

import sys
sys.path.insert(0, '/Users/arielamaral/Documents/AST430/Code/Tessa_code')

import bfield_functions_cat as bft

hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)

############################################
#import data

#data directory
pwd_data = '/Users/arielamaral/Documents/AST430/Data/'

print ' '
print "Uploading Catalogues...."

#RM catalogue Taylor et al.
RA_hms, err_RA, Dec_hms, err_DEC, b, I, S_I, S_I_err, ave_P, ave_P_err, pol_percent, pol_percent_err,RM, RM_err = np.genfromtxt(pwd_data + 'RMcat_taylor.csv', delimiter = ',', unpack=True, dtype=None)

#WISExsuperCOSMOS
WISE_cat = fits.open(pwd_data+'WISExSCOS.photoZ.MAIN.fits')
WISE_header = WISE_cat[1].header
WISE_data = WISE_cat[1].data

gal_RA = WISE_data['ra_WISE ']
gal_Dec = WISE_data['dec_WISE']
gal_photo_z = WISE_data ['zANNz   ']



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



#########################################################
#specifying healpy resolution

res=12 #SWITCH THIS TO CHANGE UP RESOLUTION

nside=2**res

npix=12*nside**2



print "total number of galaxy sources: ", len(gal_RA)

gal_RA, gal_Dec, gal_photo_z = z_reduc(gal_RA, gal_Dec, gal_photo_z)

print "total number of galaxy sources after filtering out 0.1 < z < 0.5: ", len(gal_RA)
print " "

print " "
print "HEALPY RESOLUTION: ", res
print " "



############################################################

print ' '
print "* Converting RM catalogue coordinates from HMS to degrees...."
print ' '

RA_hms = RA_hms[1:]
Dec_hms = Dec_hms[1:]
RM = RM[1:]
RM = RM.astype(np.float) #changing the read in RM vals to floats
RM_err = RM_err[1:]
RM_err = RM_err.astype(np.float)

#randomly shuffling the RMs
#np.random.shuffle(RM)


RA_RM, Dec_RM = np.loadtxt('RA_DEC_degrees.txt',unpack=True)

#get rid of RM sources not in the specific area we want

'''
RA_RM, Dec_RM, RM = RM_reduc(RA_RM, Dec_RM, RM)
'''

#reduce RM


print ' '
print "** Subtracting foreground....."
print ' '

#RM = foreground_subtract(RA_RM,Dec_RM,RM)

RM, RM_err = foreground_subtract_niels(RA_RM,Dec_RM,RM,RM_err)

#we want to use the absolute value of the RM for this calculation
#now or later?
RM = abs(RM) 


print ' '
print "*** Spliting up galaxies into redshift bins....."
print ' '

z_split, z_mean, gal_RA_split, gal_Dec_split, z_bin_list = photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)
#print "zmean is: ", z_mean

print ' '
print "**** Calculating galaxy number density grid....."
print ' '

#need to calculate 4 number density grids

#can create a function for this!
#but i'll do it here first

num_dens_total = healpy_grid2(res,gal_RA,gal_Dec)#np.zeros(npix)
num_dens_zbin = np.zeros((4,npix))

#setting all zero pixels (the galactic plane) to NaN
num_dens_total[num_dens_total==0.] = np.nan
num_dens_total_masked = hp.pixelfunc.ma(num_dens_total)
num_dens_total_masked.mask = np.isnan(num_dens_total_masked)


num_dens_zbin_masked=[]

num_dens_zbin_masked_a=[]


for i in np.arange(0,len(z_split)):
	#dont want 'nan's' though right now
	num_dens_grid_split = healpy_grid2(res,gal_RA_split[i],gal_Dec_split[i])
	num_dens_grid_split_a = healpy_grid2(res,gal_RA_split[i],gal_Dec_split[i])

	#subtract off the mean
	#plt.figure()
	num_dens_grid_split = hp.ma(num_dens_grid_split)
	num_dens_grid_split.mask = num_dens_total_masked.mask
	num_dens_grid_split -= np.mean(num_dens_grid_split) #this line makes it equal to Tessa's num_dens_zbinaves
	num_dens_zbin_masked.append(num_dens_grid_split)
	#hp.mollview(num_dens_grid_split, sub = (2,1,1))
	#plt.title('Total Number density mask for zbin = '+str(i))

	#MASKING USING ZBINS TO MASK
	num_dens_grid_split_a = hp.ma(num_dens_grid_split_a)
	num_dens_grid_split_a[num_dens_grid_split_a == 0] = np.nan
	num_dens_grid_split_a.mask = np.isnan(num_dens_grid_split_a)
	num_dens_grid_split_a -= np.mean(num_dens_grid_split_a)
	num_dens_zbin_masked_a.append(num_dens_grid_split_a)
	#hp.mollview(num_dens_grid_split_a, sub = (2,1,2))
	#plt.title('Number density zbin mask = ' + str(i))
	#plt.savefig('/Users/arielamaral/Documents/AST430/Plots/image_cross_corr/CCI_num_dens_mask/res'+str(res)+'/CCI_zbin'+str(i)+'_RES'+str(res)+'.png')



#setting all zero pixels (the galactic plane) to NaN
num_dens_total[num_dens_total==0.] = np.nan
num_dens_total_masked = hp.pixelfunc.ma(num_dens_total)
num_dens_total_masked.mask = np.isnan(num_dens_total_masked)


print ' '
print "**** Deleting RM sources which don't fall within WISE ....."
print ' '

radii_list = np.concatenate([[0.],np.logspace(1.3,3.3,15)])
arcsec_per_kpc =cosmo1.arcsec_per_kpc_comoving(z_mean).value
r_deg = (radii_list[np.newaxis,:] * arcsec_per_kpc[:,np.newaxis])/3600.


print ' '
print "****** Calculating the RM grid....."
print ' '

RM_healpy = RM_grid(RM,res,RA_RM,Dec_RM,subave=False)


print ' '
print "******** Healpy masking RM and WISE num dens grids....."
print ' '

#masking nan pixels
RM_healpy_masked = hp.pixelfunc.ma(RM_healpy)
RM_healpy_masked.mask = np.isnan(RM_healpy_masked)

RM_healpy_masked -= np.mean(RM_healpy_masked)


print ' '
print "********** Cross Correlating the two healpy grids...."
print ' '

cross_corr_z = []
cl_theta_z = []

cross_corr_z_a = []
cl_theta_z_a = []


for k in np.arange(0,len(z_mean)):
	print "...... Now correlating for zbin = ", k
	#ZTOT MASK
	cross_corr_z.append(hp.sphtfunc.anafast(num_dens_zbin_masked[k], RM_healpy_masked))
	ctheta,cl_theta_i = CCF_theta(cross_corr_z[k],l_start=0,end_theta=r_deg[k,:].max(),theta_samples=radii_list.size)
	cl_theta_z.append(cl_theta_i)
	#ZBIN MASK
	cross_corr_z_a.append(hp.sphtfunc.anafast(num_dens_zbin_masked_a[k], RM_healpy_masked))
	ctheta_a,cl_theta_i_a = bft.CCF_theta(cross_corr_z_a[k],l_start=0,end_theta=r_deg[k,:].max(),theta_samples=radii_list.size)
	cl_theta_z_a.append(cl_theta_i_a)
	


z_weight = (1.+np.array(z_mean))**(-2.)



cl_theta = (cl_theta_z*z_weight[:,np.newaxis]).sum(axis=0)
cl_theta_a = (cl_theta_z_a*z_weight[:,np.newaxis]).sum(axis=0)

plt.figure()
plt.plot(ctheta, cl_theta, 'b-', linewidth = 2, label = "num dens total")
plt.plot(ctheta_a, cl_theta_a, 'm--', linewidth = 2, label = "num dens zbin indv. mask")
plt.xlabel('Degrees')
plt.ylabel('$C_\ell$')
plt.legend(loc = 'upper left')
plt.grid()
plt.title('Image Cross Correlation, hp res: ' + str(res))
plt.savefig('/Users/arielamaral/Documents/AST430/Plots/image_cross_corr/CCI_testing/CCI_RES'+str(res)+'.png')
plt.show()
#plt.figtext(0,0,"healpy masking nan pixels on Taylor AND nan pixels resepectively on total WISE z-bin")





print ' '
print "************* DONE! :)"
print ' '



