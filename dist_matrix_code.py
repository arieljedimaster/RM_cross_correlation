import numpy as np
import astropy.cosmology as cosmo
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import stats as st
import astropy as ast
import matplotlib.pyplot as plt
import time
from math import *
import pylab as pyl
import pyfits as pf
import healpy as hp
from astropy.io import fits
from scipy.special import legendre
from scipy.special import eval_legendre
import numpy.ma as ma
import scipy.special as sp

from bfield_functions_cat import *


pwd_data = '/Users/arielamaral/Documents/AST430/Data/'


hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)

pwd_plots = '/Users/arielamaral/Documents/AST430/Plots/'



def distance_func_tessa(RM_RA, RM_Dec, gal_RA, gal_Dec, kpc2deg_list, radii_list, z_bin_list):
    #maximum radial bin:
    r = np.amax(radii_list)#np.sort(radii_list)[-1] #measured in kpc - want rmax

    print "Maximum value of radial bins (should be about 2,500 kpc): ", r

    arcsec_per_kpc = cosmo1.arcsec_per_kpc_comoving(0.1).value #number of arcseconds per kpc

    #something is wrong with this kpc conversion

    r_deg = (r * arcsec_per_kpc)/3600.

    coord_RM=SkyCoord(RM_RA*u.deg,RM_Dec*u.deg,frame='fk5')
    coord_gal=SkyCoord(gal_RA*u.deg,gal_Dec*u.deg,frame='fk5')

    rra_RM=coord_RM.ra.wrap_at(180*u.degree).value
    rra_gal=coord_gal.ra.wrap_at(180*u.degree).value

    dist_matrix_deg = []
    dist_matrix_kpc = []
    dist_indeces = [] #keeps tracck of which sources in the original gal index are in the dist matrix
    dist_z_split = [] #which redshift slice is each in?

    print "number of sources we have to make a distance matrix for: ", len(RM_RA)
    t1=time.time()
    for i in range(len(RM_RA)):
        #print i,len(RM_RA)
        if (i % 500 == 0) & (i!=0):
            print "RM source we're on... ", i,(time.time()-t1)/60.,np.concatenate(np.array(dist_matrix_kpc)).size

        RA_min = RM_RA[i] - r_deg
        RA_max = RM_RA[i] + r_deg
        Dec_min = RM_Dec[i] - r_deg
        Dec_max = RM_Dec[i] + r_deg
        
        #ind_gal_RA = np.where((RA_min < gal_RA) & (gal_RA <  RA_max))[0] #all galaxies which fall in the RA range
        #ind_gal_Dec = np.where((Dec_min < gal_Dec) & (gal_Dec < Dec_max))[0] #all galaxies which fall in the Dec range
        #gal_in_r_ind = np.union1d(ind_gal_RA,ind_gal_Dec)  #all galaxies which fall in both RA and Dec range
        if RA_min<0:
            gal_in_r_ind = np.argwhere((RA_min < rra_gal) & (rra_gal <  RA_max) & (Dec_min < gal_Dec) & (gal_Dec < Dec_max))[:,0]  #all galaxies which fall in both RA and Dec range
        if RA_max>360:
            RA_min = rra_RM[i] - r_deg
            RA_max = rra_RM[i] + r_deg
            gal_in_r_ind = np.argwhere((RA_min < rra_gal) & (rra_gal <  RA_max) &(Dec_min < gal_Dec) & (gal_Dec < Dec_max))[:,0]
        else:
            gal_in_r_ind = np.argwhere((RA_min < gal_RA) & (gal_RA <  RA_max) &(Dec_min < gal_Dec) & (gal_Dec < Dec_max))[:,0]

        #lists of all the RA/Dec of the galaxies which fall within this postage stamp
        gal_RA_in_r = gal_RA[gal_in_r_ind]
        gal_Dec_in_r = gal_Dec[gal_in_r_ind]
        gal_sep_in_r = kpc2deg_list[gal_in_r_ind]
        #now we want to calculate the distances between every galaxy in here and the RM source
        
        
        #using skycoord values makes it easier to calculate separations
        #distance in degrees
        separation = np.array(coord_RM[i].separation(coord_gal[gal_in_r_ind]).value, dtype = float)
        dist_matrix_deg.append(separation) #distances from each galaxy to that RM source
        dist_matrix_kpc.append(separation/gal_sep_in_r)
        
        dist_indeces.append(gal_in_r_ind)
        
        dist_z_split.append(z_bin_list[gal_in_r_ind])

    return np.array(dist_matrix_deg), np.array(dist_matrix_kpc), np.array(dist_indeces), np.array(dist_z_split)


def distance_func_tessa_zbin(RM_RA, RM_Dec, gal_RA, gal_Dec, kpc2deg_list, radii_list, z_bin_list):
    #maximum radial bin:
    r = np.amax(radii_list)#np.sort(radii_list)[-1] #measured in kpc - want rmax

    r_deg = (r * kpc2deg_list)/3600.

    print "r_deg: ", r_deg

    coord_RM=SkyCoord(RM_RA*u.deg,RM_Dec*u.deg,frame='fk5')
    coord_gal=SkyCoord(gal_RA*u.deg,gal_Dec*u.deg,frame='fk5')

    rra_RM=coord_RM.ra.wrap_at(180*u.degree).value
    rra_gal=coord_gal.ra.wrap_at(180*u.degree).value

    dist_matrix_deg = []
    dist_matrix_kpc = []
    dist_indeces = [] #keeps tracck of which sources in the original gal index are in the dist matrix
    dist_z_split = [] #which redshift slice is each in?

    print "number of sources we have to make a distance matrix for: ", len(RM_RA)

    t1=time.time()
    for i in range(len(RM_RA)):
        #print i,len(RM_RA)
        if (i % 500 == 0) & (i!=0):
            print "RM source we're on... ", i,(time.time()-t1)/60.,np.concatenate(np.array(dist_matrix_kpc)).size

        RA_min = RM_RA[i] - r_deg
        RA_max = RM_RA[i] + r_deg
        Dec_min = RM_Dec[i] - r_deg
        Dec_max = RM_Dec[i] + r_deg

        if RA_min<0:
            gal_in_r_ind = np.argwhere((RA_min < rra_gal) & (rra_gal <  RA_max) & (Dec_min < gal_Dec) & (gal_Dec < Dec_max))[:,0]  #all galaxies which fall in both RA and Dec range
        if RA_max>360:
            RA_min = rra_RM[i] - r_deg
            RA_max = rra_RM[i] + r_deg
            gal_in_r_ind = np.argwhere((RA_min < rra_gal) & (rra_gal <  RA_max) &(Dec_min < gal_Dec) & (gal_Dec < Dec_max))[:,0]
        else:
            gal_in_r_ind = np.argwhere((RA_min < gal_RA) & (gal_RA <  RA_max) &(Dec_min < gal_Dec) & (gal_Dec < Dec_max))[:,0]

        #lists of all the RA/Dec of the galaxies which fall within this postage stamp
        gal_RA_in_r = gal_RA[gal_in_r_ind]
        gal_Dec_in_r = gal_Dec[gal_in_r_ind]
        gal_sep_in_r = kpc2deg_list[gal_in_r_ind]
        #now we want to calculate the distances between every galaxy in here and the RM source
        
        
        #using skycoord values makes it easier to calculate separations
        #distance in degrees
        separation = np.array(coord_RM[i].separation(coord_gal[gal_in_r_ind]).value, dtype = float)
        dist_matrix_deg.append(separation) #distances from each galaxy to that RM source
        dist_matrix_kpc.append(separation/gal_sep_in_r)
        
        dist_indeces.append(gal_in_r_ind)
        
        dist_z_split.append(z_bin_list[gal_in_r_ind])

    return np.array(dist_matrix_deg), np.array(dist_matrix_kpc), np.array(dist_indeces), np.array(dist_z_split)



####### lets try to make this faster

def distance_func_zbins(RM_RA, RM_Dec, kpc2deg_list, radii_list,  gal_RA_split, gal_Dec_split,z_bin_list, z_mean):
    #maximum radial bin:
    r = np.amax(radii_list)#np.sort(radii_list)[-1] #measured in kpc - want rmax

    zbins = np.linspace(0.1,0.4,4)

    dist_matrix_deg = [[] for i in range(len(RM_RA))]
    dist_matrix_kpc = [[] for i in range(len(RM_RA))]
    dist_matrix_kpc_alt = [[] for i in range(len(RM_RA))]
    dist_indeces = [[] for i in range(len(RM_RA))] #keeps tracck of which sources in the original gal index are in the dist matrix
    dist_z_split = [[] for i in range(len(RM_RA))] #which redshift slice is each in?

    print "number of sources we have to make a distance matrix for: ", len(RM_RA)

    for z in np.arange(0, len(zbins)):

        print "z_mean for this bin: ", z_mean

        #using the mean redshfit of the redshift bin

        arcsec_per_kpc = cosmo1.arcsec_per_kpc_comoving(z_mean[z]).value #number of arcseconds per kpc

        print "arcsec_per_kpc: ", arcsec_per_kpc

        #galaxies in that zbin range:

        gal_RA_in_z = gal_RA_split[z]
        print gal_RA_in_z

        gal_Dec_in_z = gal_Dec_split[z]
        print gal_Dec_in_z

        r_deg = (r * arcsec_per_kpc)/3600.

        coord_RM=SkyCoord(RM_RA*u.deg,RM_Dec*u.deg,frame='fk5')
        coord_gal=SkyCoord(gal_RA_in_z*u.deg,gal_Dec_in_z*u.deg,frame='fk5')

        rra_RM=coord_RM.ra.wrap_at(180*u.degree).value
        rra_gal=coord_gal.ra.wrap_at(180*u.degree).value

        t1=time.time()
        for i in range(len(RM_RA)):

            RA_min = RM_RA[i] - r_deg
            RA_max = RM_RA[i] + r_deg
            Dec_min = RM_Dec[i] - r_deg
            Dec_max = RM_Dec[i] + r_deg

            if RA_min<0:
                gal_in_r_ind = np.argwhere((RA_min < rra_gal) & (rra_gal <  RA_max) & (Dec_min < gal_Dec_in_z) & (gal_Dec_in_z < Dec_max))[:,0]  #all galaxies which fall in both RA and Dec range
            if RA_max>360:
                RA_min = rra_RM[i] - r_deg
                RA_max = rra_RM[i] + r_deg
                gal_in_r_ind = np.argwhere((RA_min < rra_gal) & (rra_gal <  RA_max) &(Dec_min < gal_Dec_in_z) & (gal_Dec_in_z < Dec_max))[:,0]
            else:
                gal_in_r_ind = np.argwhere((RA_min < gal_RA_in_z) & (gal_RA_in_z <  RA_max) &(Dec_min < gal_Dec_in_z) & (gal_Dec_in_z < Dec_max))[:,0]

            #lists of all the RA/Dec of the galaxies which fall within this postage stamp
            gal_RA_in_r = gal_RA_in_z[gal_in_r_ind]
            gal_Dec_in_r = gal_Dec_in_z[gal_in_r_ind]
            gal_sep_in_r = kpc2deg_list[gal_in_r_ind]

            #now we want to calculate the distances between every galaxy in here and the RM source
            
            
            #using skycoord values makes it easier to calculate separations
            #distance in degrees
            #appending to that specific RM source location in the distance matrices
            separation = np.array(coord_RM[i].separation(coord_gal[gal_in_r_ind]).value, dtype = float)
            dist_matrix_deg[i]+= list(separation) #distances from each galaxy to that RM source
            dist_matrix_kpc[i] += list(separation/gal_sep_in_r)

            
            dist_indeces[i] += list(gal_in_r_ind)
            
            dist_z_split[i] += list(z_bin_list[gal_in_r_ind])

    return np.array(dist_matrix_deg), np.array(dist_matrix_kpc), np.array(dist_indeces), np.array(dist_z_split)




def distance_func_zbins_max(RM_RA, RM_Dec, kpc2deg_list, radii_list,  gal_RA_split, gal_Dec_split,z_bin_list, z_mean):
    #maximum radial bin:
    r = np.amax(radii_list)#np.sort(radii_list)[-1] #measured in kpc - want rmax

    zbins = np.linspace(0.1,0.4,4)

    dist_matrix_deg = [[] for i in range(len(RM_RA))]
    dist_matrix_kpc = [[] for i in range(len(RM_RA))]
    dist_matrix_kpc_alt = [[] for i in range(len(RM_RA))]
    dist_indeces = [[] for i in range(len(RM_RA))] #keeps tracck of which sources in the original gal index are in the dist matrix
    dist_z_split = [[] for i in range(len(RM_RA))] #which redshift slice is each in?

    print "number of sources we have to make a distance matrix for: ", len(RM_RA)

    for z in np.arange(0, len(zbins)):

        print "z_bins[z]: ", zbins[z]

        print "LOWER bound of redshift bin: ", zbins[z]

        #using the lower bound of the redshift bin

        arcsec_per_kpc = cosmo1.arcsec_per_kpc_comoving(zbins[z]).value #number of arcseconds per kpc

        print "arcsec_per_kpc: ", arcsec_per_kpc

        #galaxies in that zbin range:

        gal_RA_in_z = gal_RA_split[z]
        print gal_RA_in_z

        gal_Dec_in_z = gal_Dec_split[z]
        print gal_Dec_in_z

        r_deg = (r * arcsec_per_kpc)/3600.

        coord_RM=SkyCoord(RM_RA*u.deg,RM_Dec*u.deg,frame='fk5')
        coord_gal=SkyCoord(gal_RA_in_z*u.deg,gal_Dec_in_z*u.deg,frame='fk5')

        rra_RM=coord_RM.ra.wrap_at(180*u.degree).value
        rra_gal=coord_gal.ra.wrap_at(180*u.degree).value

        t1=time.time()
        for i in range(len(RM_RA)):

            RA_min = RM_RA[i] - r_deg
            RA_max = RM_RA[i] + r_deg
            Dec_min = RM_Dec[i] - r_deg
            Dec_max = RM_Dec[i] + r_deg

            if RA_min<0:
                gal_in_r_ind = np.argwhere((RA_min < rra_gal) & (rra_gal <  RA_max) & (Dec_min < gal_Dec_in_z) & (gal_Dec_in_z < Dec_max))[:,0]  #all galaxies which fall in both RA and Dec range
            if RA_max>360:
                RA_min = rra_RM[i] - r_deg
                RA_max = rra_RM[i] + r_deg
                gal_in_r_ind = np.argwhere((RA_min < rra_gal) & (rra_gal <  RA_max) &(Dec_min < gal_Dec_in_z) & (gal_Dec_in_z < Dec_max))[:,0]
            else:
                gal_in_r_ind = np.argwhere((RA_min < gal_RA_in_z) & (gal_RA_in_z <  RA_max) &(Dec_min < gal_Dec_in_z) & (gal_Dec_in_z < Dec_max))[:,0]

            #lists of all the RA/Dec of the galaxies which fall within this postage stamp
            gal_RA_in_r = gal_RA_in_z[gal_in_r_ind]
            gal_Dec_in_r = gal_Dec_in_z[gal_in_r_ind]
            gal_sep_in_r = kpc2deg_list[gal_in_r_ind]

            #now we want to calculate the distances between every galaxy in here and the RM source
            
            
            #using skycoord values makes it easier to calculate separations
            #distance in degrees
            #appending to that specific RM source location in the distance matrices
            separation = np.array(coord_RM[i].separation(coord_gal[gal_in_r_ind]).value, dtype = float)
            dist_matrix_deg[i]+= list(separation) #distances from each galaxy to that RM source
            dist_matrix_kpc[i] += list(separation/gal_sep_in_r)

            
            dist_indeces[i] += list(gal_in_r_ind)
            
            dist_z_split[i] += list(z_bin_list[gal_in_r_ind])

    return np.array(dist_matrix_deg), np.array(dist_matrix_kpc), np.array(dist_indeces), np.array(dist_z_split)



################################################################

nside= 128
npix = 12*nside**2
res = int(np.log(nside)/np.log(2))

#Taylor Catalogue

RA_hms, err_RA, Dec_hms, err_DEC, b, I, S_I, S_I_err, ave_P, ave_P_err, pol_percent, pol_percent_err,RM, RM_err = np.genfromtxt(pwd_data + 'RMcat_taylor.csv', delimiter = ',', unpack=True, skip_header=1)

RA_RM, Dec_RM = np.loadtxt('RA_DEC_degrees.txt',unpack=True)



#WISE/superCOSMOS

WISE_cat = fits.open(pwd_data+'WISExSCOS.photoZ.MAIN.fits')
WISE_header = WISE_cat[1].header
WISE_data = WISE_cat[1].data

gal_RA = WISE_data['ra_WISE ']
gal_Dec = WISE_data['dec_WISE']
gal_photo_z = WISE_data ['zANNz   ']

gal_RA, gal_Dec, gal_photo_z = z_reduc(gal_RA, gal_Dec, gal_photo_z)

z_split, z_mean, gal_RA_split, gal_Dec_split, z_bin_list = photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)

num_dens_grid = healpy_grid(res,gal_RA,gal_Dec)


#log spaced bins

radii_list_logbins= np.logspace(1.,3.4,9)
radii_list_logbins[0] = 0.

#linear spaced bins

radii_list_linbins = np.linspace(0., 2511.88643150958, 9)




RA_RM, Dec_RM, RM = mask2(num_dens_grid, nside, RA_RM, Dec_RM, RM, radii_list_logbins)

total_gal_in_z = total_num_gals_in_z(z_bin_list)

kpc2deg_list = gal_conv_fact(z_bin_list,z_mean)


RA_RM = [RA_RM[1000]]
Dec_RM = [Dec_RM[1000]]

print RA_RM,Dec_RM



#dist_matrix_deg_z, dist_matrix_kpc_z, dist_indeces_z, dist_z_split_z = distance_func_zbins(RA_RM, Dec_RM, kpc2deg_list, radii_list_logbins, gal_RA_split, gal_Dec_split, z_bin_list, z_mean)

#dist_matrix_deg_z_max, dist_matrix_kpc_z_max, dist_indeces_z_max, dist_z_split_z_max = distance_func_zbins_max(RA_RM, Dec_RM, kpc2deg_list, radii_list_logbins, gal_RA_split, gal_Dec_split, z_bin_list, z_mean)

dist_matrix_deg_tessa, dist_matrix_kpc_tessa, dist_indeces_tessa, dist_z_split_tessa = distance_func_tessa(RA_RM, Dec_RM, gal_RA, gal_Dec, kpc2deg_list, radii_list_logbins, z_bin_list)

dist_matrix_deg_tessa_zbin, dist_matrix_kpc_tessa_zbin, dist_indeces_tessa_zbin, dist_z_split_tessa_zbin = distance_func_tessa_zbin(RA_RM, Dec_RM, gal_RA, gal_Dec, kpc2deg_list, radii_list_logbins, z_bin_list)

bins1 = np.linspace(0,np.amax(dist_matrix_kpc_tessa[0]), 20)

plt.figure()

#plt.hist(dist_matrix_kpc_tessa[0],bins=bins1, normed =True, histtype='bar', facecolor = 'r', alpha = 0.5, label = 'distance func')
plt.hist(dist_matrix_kpc_tessa_zbin[0],bins=bins1, normed = True,histtype='bar', facecolor = 'b', alpha = 0.5, label = 'distance func zbin')
#plt.hist(dist_matrix_kpc_z_max[0],bins=bins1, histtype='bar', facecolor = 'y', alpha = 0.5, label = 'distance func zbin - lower z')
#plt.hist(dist_matrix_kpc_z[0], bins=bins1,histtype='bar', facecolor = 'b', alpha = 0.5, label = 'distance func zbin - zmean')
plt.ylabel('number of galaxies')
plt.xlabel('Distance from the random RM source [kpc]')
plt.title('Looking at the difference between the two distance matrix functions')
plt.legend(loc ='upper left')
plt.show()

'''
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

