#autocorrelation

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

NVSS_pwd = '/Users/arielamaral/Documents/AST430/Code/Code_new_cat/auto_correlation/NVSS/dist_matrices_ariel/'



#should I even be including redshift info??

kpc2deg_list = bf.gal_conv_fact(z_bin_list,z_mean)

#should I do one for the random ones as well?

z_split_rand, z_mean_rand, gal_RA_split_rand, gal_Dec_split_rand, z_bin_list_rand = bf.photo_z_split(gal_photo_z_rand, gal_RA_rand, gal_Dec_rand, bin_width =0.1)

kpc2deg_list_rand = bf.gal_conv_fact(z_bin_list_rand,z_mean_rand)

def distance_func_gal(gal_RA1, gal_Dec1, gal_RA2, gal_Dec2, kpc2deg_list2, theta_list):

    #we are calculating the distances to from each gal1 source to all the gal2 sources
    #ex: if we want to find DR(theta), all distances of all the random sources to each data source:
    #gal 1 = data
    #gal 2 = random
    #kpc2deg_list2 = kpc2_deg_list_rand

    z_mean = 0.2165125508812035 
    #maximum radial bin:
    r_deg = np.amax(theta_list) #in degrees

    arcsec_per_kpc = cosmo1.arcsec_per_kpc_comoving(0.1).value #number of arcseconds per kpc

    coord_gal1=SkyCoord(gal_RA1*u.deg,gal_Dec1*u.deg,frame='fk5')
    coord_gal2=SkyCoord(gal_RA2*u.deg,gal_Dec2*u.deg,frame='fk5')
    rra_gal1=coord_gal1.ra.wrap_at(180*u.degree).value
    rra_gal2=coord_gal2.ra.wrap_at(180*u.degree).value

    dist_matrix_deg = []
    dist_matrix_kpc = []
    dist_indeces = [] #keeps tracck of which sources in the original gal index are in the dist matrix
    dist_z_split = [] #which redshift slice is each in?

    print "number of sources we have to make a distance matrix for: ", len(gal_RA1)
    t1=time.time()
    for i in range(len(gal_RA1)):
        #print i,len(RM_RA)
        if (i % 500 == 0) & (i!=0):
            print "RM source we're on... ", i," ---------- time: ", (time.time()-t1)/60.

        RA_min = gal_RA1[i] - r_deg
        RA_max = gal_RA1[i] + r_deg
        Dec_min = gal_Dec1[i] - r_deg
        Dec_max = gal_Dec1[i] + r_deg

        if RA_min<0:
            gal2_in_r_ind = np.argwhere((RA_min < rra_gal2) & (rra_gal2 <  RA_max) &(Dec_min < gal_Dec2) & (gal_Dec2 < Dec_max))[:,0]  #all galaxies which fall in both RA and Dec range
        if RA_max>360:
            RA_min = rra_gal1[i] - r_deg
            RA_max = rra_gal1[i] + r_deg
            gal2_in_r_ind = np.argwhere((RA_min < rra_gal2) & (rra_gal2 <  RA_max) &(Dec_min < gal_Dec2) & (gal_Dec2 < Dec_max))[:,0]
        else:
            gal2_in_r_ind = np.argwhere((RA_min < gal_RA2) & (gal_RA2 <  RA_max) &(Dec_min < gal_Dec2) & (gal_Dec2 < Dec_max))[:,0]

        #lists of all the RA/Dec of the galaxies which fall within this postage stamp
        gal2_RA_in_r = gal_RA[gal2_in_r_ind]
        gal2_Dec_in_r = gal_Dec[gal2_in_r_ind]
        gal2_sep_in_r = kpc2deg_list[gal2_in_r_ind]
        #now we want to calculate the distances between every galaxy in here and the RM source
        
        
        #using skycoord values makes it easier to calculate separations
        #distance in degrees
        separation = np.array(coord_gal1[i].separation(coord_gal2[gal2_in_r_ind]).value, dtype = float)
        dist_matrix_deg.append(separation) #distances from each galaxy to that RM source
        dist_matrix_kpc.append(separation/gal2_sep_in_r)
        
        dist_indeces.append(gal2_in_r_ind)
        
        dist_z_split.append(z_bin_list[gal2_in_r_ind])

    return np.array(dist_matrix_deg)#, np.array(dist_matrix_kpc), np.array(dist_indeces), np.array(dist_z_split)


#save these because they're gonna take a while to run

distances_DD = distance_func_gal(gal_RA, gal_Dec, gal_RA, gal_Dec, kpc2deg_list, theta_list)
distances_DR = distance_func_gal(gal_RA, gal_Dec, gal_RA_rand, gal_Dec_rand, kpc2deg_list_rand, theta_list)
distances_RR = distance_func_gal(gal_RA_rand, gal_Dec_rand, gal_RA_rand, gal_Dec_rand, kpc2deg_list_rand, theta_list)

np.save(dist_pwd+"dist_matrix_deg_DD_NVSS.npy", distances_DD)
np.save(dist_pwd+'dist_matrix_deg_DR_NVSS.npy',distances_DR)
np.save(dist_pwd+'dist_matrix_deg_RR_NVSS.npy',distances_RR)



#load them whenever we want to run:

distances_DD = np.load(dist_pwd + "dist_matrix_deg_DD.npy")
distances_DR = np.load(dist_pwd + "dist_matrix_deg_DR.npy")
distances_RR = np.load(dist_pwd + "dist_matrix_deg_RR.npy")



def pair_estimator(theta_list, distances_DD, distances_DR, distances_RR):

    DD_theta = np.zeros(len(theta_list)-1)
    DR_theta = np.zeros(len(theta_list)-1)
    RR_theta = np.zeros(len(theta_list)-1)

    for i in np.arange(0,len(distances_DD)):
        #data-data histogram

        DD_theta += np.histogram(distances_DD[i], bins = theta_list)[0]

        #data-random histogram

        DR_theta += np.histogram(distances_DR[i], bins = theta_list)[0]

        #random-random histogram

        RR_theta += np.histogram(distances_RR[i], bins = theta_list)[0]

    #w_theta calculation

    w_theta = (DD_theta - 2.*DR_theta + RR_theta)/(RR_theta.astype(float))

    return w_theta


w_theta = pair_estimator(theta_list, distances_DD, distances_DR, distances_RR)

#plotting the test cases:

theta_list = [(a + b) /2. for a, b in zip(theta_list[::], theta_list[1::])]

plt.figure()
plt.plot(theta_list, w_theta, 'm-', linewidth = 2)
plt.title('Angular correlation function for test area: ' + str(gal_RA_min)+ ' < RA < ' + str(gal_RA_max) +' and ' + str(gal_Dec_min) + ' < Dec < ' + str(gal_Dec_max))
plt.xlabel('Distance [Degrees]') 
plt.ylabel('Angular Correlation Function')
plt.show()