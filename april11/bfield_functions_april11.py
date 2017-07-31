#from scipy import *
import numpy as np
#import pyfits as pf
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

import numpy.ma as ma
import scipy.special as sp

# Tessa's edits to my code, done on april 11th 2017


pwd_data = '/Users/arielamaral/Documents/AST430/Data/'


hknot=70.2
cosmo1=cosmo.FlatLambdaCDM(H0=hknot,Om0=.308)

pwd_plots = '/Users/arielamaral/Documents/AST430/Plots/'

def z_reduc(gal_RA, gal_Dec, gal_photo_z):
	z_reduc_indeces = np.where((0.1 < gal_photo_z)&(gal_photo_z < 0.5))
	gal_RA_new = gal_RA[z_reduc_indeces]
	gal_Dec_new = gal_Dec[z_reduc_indeces]
	gal_photo_z_new = gal_photo_z[z_reduc_indeces]
	return gal_RA_new, gal_Dec_new, gal_photo_z_new

def RM_reduc(RA_RM, Dec_RM, RM):

	new_RA_RM = []
	new_Dec_RM = []
	new_RM = []

	#I should put this into a function

	for j in np.arange(0,len(RA_RM)):
		if (120. <= RA_RM[j] <= 240.) and ( 0. <= Dec_RM[j] <= 60.):
			new_RA_RM.append(RA_RM[j])
			new_Dec_RM.append(Dec_RM[j])
			new_RM.append(RM[j])
		else:
			continue

	RA_RM = new_RA_RM
	Dec_RM = new_Dec_RM
	RM = new_RM

	return RA_RM, Dec_RM, RM



def foreground_subtract_niels(RA_RM,Dec_RM,RM,RM_err):
	nside_niels = 128
	npix_niels = 12*nside_niels**2
	res_niels = int(np.log(nside_niels)/np.log(2))

	niels_cat = fits.open(pwd_data+'faraday.fits')
	niels_header = niels_cat[1].header

	niels_data = niels_cat[3].data

	niels_err = niels_cat[4].data
	niels_err = niels_err.astype(np.float)

	niels_data = niels_data.astype(np.float)

	niels_pix = np.arange(0,196607+1)

	#degrees to radians
	cc1=SkyCoord(RA_RM*u.deg,Dec_RM*u.deg,frame='fk5')
	#degrees to radians
	Dec_RMr=(np.pi/180.)*(90.-cc1.galactic.b.deg)
	RA_RMr=(np.pi/180.)*(cc1.galactic.l.deg)
	pix_RM = hp.ang2pix(nside_niels,Dec_RMr,RA_RMr)

	RRM = np.zeros(len(RM))
	RRM_err = np.zeros(len(RM))

	for i in np.arange(0, len(pix_RM)):
		if niels_err[i] <= ((1.22*RM_err[i])**2. + 6.9**2.):
			pix = pix_RM[i]
			RRM[i] = RM[i] - niels_data[pix]
			RRM_err[i] = niels_err[i]
		else:
			RRM[i] = RM[i]
			RRM_err[i] = RM_err[i]

	return RRM, RRM_err

def foreground_subtract_niels1(RA_RM,Dec_RM,RM):

    #not including the errors right now

    

    nside_niels = 128

    npix_niels = 12*nside_niels**2

    res_niels = int(np.log(nside_niels)/np.log(2))

    

    niels_cat = fits.open(pwd_data+'faraday.fits')

    niels_header = niels_cat[1].header

    niels_data = niels_cat[3].data

    

    niels_data = niels_data.astype(np.float)

    

    niels_pix = np.arange(0,196607+1)

    cc1=SkyCoord(RA_RM*u.deg,Dec_RM*u.deg,frame='fk5')

    #degrees to radians

    Dec_RMr=(np.pi/180.)*(90.-cc1.galactic.b.deg)

    RA_RMr=(np.pi/180.)*(cc1.galactic.l.deg)

    

    pix_RM = hp.ang2pix(nside_niels,Dec_RMr,RA_RMr)

    

    niels_at_RM = niels_data[pix_RM] #niels foreground at the RM sources

    

    RRM = RM - niels_at_RM



    return RRM


def foreground_subtract(RM_RA_list,RM_Dec_list,RM_vals):
	#OLD METHOD

	RM_avg = np.zeros(len(RM_vals))
	for i in np.arange(0,len(RM_RA_list)): #looping through the RM sources
		#go to every RM source, RM_RA[i], RM_Dec[i]
		RM_3deg = np.where(((RM_RA_list[i] - 3.) < RM_RA_list) &  (RM_RA_list < (RM_RA_list[i] + 3.)))  #all sources within 3 RM degrees
		Dec_3deg = np.where(((RM_Dec_list[i] - 3.) < RM_Dec_list) & (RM_Dec_list < (RM_Dec_list[i] + 3.)))#all sources within 3 Dec degrees
		#I need distance ^ for the above
		all_3deg = np.intersect1d(RM_3deg,Dec_3deg)#better way to do this?
		RM_3deg = np.zeros(len(all_3deg))
		for j in np.arange(0,len(RM_3deg)):
			RM_3deg[j] = RM_vals[all_3deg[j]]
		RM_3deg = [float(k) for k in RM_3deg]
		#take average inside 3 degree circ
		avg_3deg = np.median(RM_3deg)
		#append avg to big list
		RM_avg[i] = avg_3deg

	new_RM = RM_vals - RM_avg

	return new_RM


def grid_split(gal_RA_list, gal_Dec_list):

	#exports the number density on a grid
	num_dens_grid, xedges, yedges = np.histogram2d(gal_RA_list, gal_Dec_list, bins = (2048, 2048))
	
	return num_dens_grid, xedges, yedges


def healpy_grid(res,ra,dec):
	dec=(np.pi/180.)*(90.-dec)
	ra=(np.pi/180.)*(ra)
	nside=2**res
	npix=12*nside**2
	#nn=ra.size #length of RA array?

	vals=np.zeros(npix)
	pixs=hp.ang2pix(nside,dec,ra)

	bins =np.bincount(pixs)
	num_bins=bins.size
	xxs=np.arange(num_bins)
	vals[xxs]=bins

	vals[vals==0.]=np.nan

	#vals_ang = hp.pix2ang(nside,vals,lonlat=True)

	return vals

def healpy_grid2(res,ra,dec):
	dec=(np.pi/180.)*(90.-dec)
	ra=(np.pi/180.)*(ra)
	nside=2**res
	npix=12*nside**2
	#nn=ra.size #length of RA array?

	vals=np.zeros(npix)
	pixs=hp.ang2pix(nside,dec,ra)

	bins =np.bincount(pixs)
	num_bins=bins.size
	xxs=np.arange(num_bins)
	vals[xxs]=bins

	#vals[vals==0.]=np.nan

	#vals_ang = hp.pix2ang(nside,vals,lonlat=True)

	return vals




def photo_z_split(photo_z_list, gal_RA_list, gal_Dec_list,bin_width): 

	gal_RA_list = np.array(gal_RA_list)
	gal_Dec_list = np.array(gal_Dec_list)
	num_bins = np.arange(0.1,0.5,bin_width)
	num_bins_len = len(num_bins)
	z_split = [[]for _ in range(num_bins_len)]
	RA_split = [[]for _ in range(num_bins_len)]
	Dec_split = [[]for _ in range(num_bins_len)]
	z_bin_list = np.zeros(len(photo_z_list)) #which redshift bin each source is in
	z_mean = []

	for i in np.arange(0, len(z_split)):
		min_z = num_bins[i]
		max_z = num_bins[i] + bin_width
		sources_index = np.where((min_z < photo_z_list)  & (photo_z_list < max_z))
		sources = photo_z_list[sources_index]
		z_bin_list[sources_index] = i #or z_split[i]????

		'''
		sources = np.zeros(len(sources_index))
		for j in np.arange(0,len(sources)):
			sources[j] = photo_z_list[sources_index[j]]
		'''

		z_split[i] = sources
		RA_split[i] = gal_RA_list[sources_index]
		Dec_split[i] = gal_Dec_list[sources_index]
		z_mean.append(np.mean(photo_z_list[sources_index]))

	return z_split, z_mean, RA_split, Dec_split, z_bin_list


def gal_conv_fact(z_bin_list,z_mean):
	#returns a list of the conversion factor for each galaxy source given by it's redshift bins
	kpc2deg_list = np.zeros(len(z_bin_list))

	for i in np.arange(0,len(z_mean)):
		#locations in zbin list where all the sources are in that desired zbin
		zbin_ind = np.where(z_bin_list == i)
		#kpc to degrees or degrees to kpc
		kpc2deg_list[zbin_ind] = (cosmo1.arcsec_per_kpc_comoving(z_mean[i]).value)/3600.

	return kpc2deg_list


def distance_func_tessa(RM_RA, RM_Dec, gal_RA, gal_Dec, kpc2deg_list, radii_list, z_bin_list):
    #maximum radial bin:
    r = np.sort(radii_list)[-1] #measured in kpc - want rmax

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
            gal_in_r_ind = np.argwhere((RA_min < rra_gal) & (rra_gal <  RA_max) &(Dec_min < gal_Dec) & (gal_Dec < Dec_max))[:,0]  #all galaxies which fall in both RA and Dec range
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

def distance_func2(nside,RA_RM,Dec_RM,gal_RA,gal_Dec,kpc2deg_list, radii_list, z_bin_list):
    
    r = np.sort(radii_list)[-1] #measured in kpc - want rmax
    arcsec_per_kpc = cosmo1.arcsec_per_kpc_comoving(0.1).value #number of arcseconds per kpc
    r_deg = (r * arcsec_per_kpc)/3600.
        
    coord_gal=SkyCoord(gal_RA*u.deg,gal_Dec*u.deg,frame='fk5')
    coord_RM=SkyCoord(RA_RM*u.deg,Dec_RM*u.deg,frame='fk5')
    
    lrad_gal=(np.pi/180.)*coord_gal.ra.deg
    brad_gal=(np.pi/180.)*(90-coord_gal.dec.deg)
    lpix_gal=hp.ang2pix(nside,brad_gal,lrad_gal)
    
    dist_matrix_deg=[]
    dist_matrix_kpc=[]
    dist_indeces=[]
    dist_z_split=[]
    t1=time.time()
    for i in np.arange(0,len(Dec_RM)):
        if (i % 500 == 0) & (i!=0):
            print "RM source we're on... ", i,(time.time()-t1)/60.,np.concatenate(np.array(dist_matrix_kpc)).size
        RA_min = RA_RM[i] - r_deg
        RA_max = RA_RM[i] + r_deg
        Dec_min = Dec_RM[i] - r_deg
        Dec_max = Dec_RM[i] + r_deg
        Dec_RM_rad=(np.pi/180.)*(90.-Dec_RM[i])
        RA_RM_rad=(np.pi/180.)*(RA_RM[i])
        Dec_min=(np.pi/180.)*(90.-Dec_min)
        RA_min=(np.pi/180.)*(RA_min)
        Dec_max=(np.pi/180.)*(90.-Dec_max)
        RA_max=(np.pi/180.)*(RA_max)
        vect1 = hp.ang2vec(Dec_min,RA_min)
        vect2 = hp.ang2vec(Dec_min,RA_max)
        vect3 = hp.ang2vec(Dec_max,RA_max)
        vect4 = hp.ang2vec(Dec_max,RA_min)
        vertices = np.array([vect1,vect2,vect3,vect4])
        pix_matrix = hp.query_polygon(nside, vertices, inclusive=True)
        gal_in_r_ind=[]
        for j in range(len(pix_matrix)):
            gal_in_r_ind.append(np.argwhere(lpix_gal==pix_matrix[j])[:,0])
    
        gal_in_r_ind=np.concatenate(np.array(gal_in_r_ind))

        separation = np.array(coord_RM[i].separation(coord_gal[gal_in_r_ind]).value, dtype = float)
        dist_matrix_deg.append(separation) #distances from each galaxy to that RM source
        gal_sep_in_r = kpc2deg_list[gal_in_r_ind]
        dist_matrix_kpc.append(separation/gal_sep_in_r)
        dist_indeces.append(gal_in_r_ind)
        dist_z_split.append(z_bin_list[gal_in_r_ind])
        
    return np.array(dist_matrix_deg), np.array(dist_matrix_kpc), np.array(dist_indeces), np.array(dist_z_split)



def mask(grid, RM_RA, RM_Dec, RM, radii_list):

	grid_size_RA = (240. - 120.)/2048.
	grid_size_Dec = 60./2048.

	RM_RA = np.array(RM_RA)
	RM_Dec = np.array(RM_Dec)

	#will both be arrays, faster this way
	RM_RA_grid_ind = ((RM_RA - 120.)/grid_size_RA)
	#changing every element to an int
	RM_RA_grid_ind = RM_RA_grid_ind.astype(int)
	RM_Dec_grid_ind = RM_Dec/grid_size_Dec
	RM_Dec_grid_ind = RM_Dec_grid_ind.astype(int)

	#maximum radial bin:
	r = np.sort(radii_list)[-1] #measured in kpc - want rmax

	arcsec_per_kpc = cosmo1.arcsec_per_kpc_comoving(0.1).value #number of arcseconds per kpc

	#something is wrong with this kpc conversion

	r_deg = (r * arcsec_per_kpc)/3600.

	
	R_max_ind = int(r_deg/grid_size_RA)

	#all indeces which have zero pixels

	zero_pix_ind = np.where(grid==0)

	#if postage stamp around RA contains a zero pixel get rid of that RA value
	#also if grid stamp is far away from the edges!

	new_RM = []
	new_RM_RA =[]
	new_RM_Dec = []

	for i in np.arange(0,len(RM_Dec)):
		#check to make sure RM source isnt too close to the edges:
		#if (((RM_RA[i] - r_deg) >= 120.) and ((RM_RA[i] + r_deg) <= 240.) and ((RM_Dec[i] - r_deg) >= 0.) and ((RM_Dec[i] + r_deg) <= 60.)):
		if (((RM_RA_grid_ind[i] - R_max_ind) >= 0.) and ((RM_RA_grid_ind[i] + R_max_ind) <= 2048.) and ((RM_Dec_grid_ind[i] - R_max_ind) >= 0.) and ((RM_Dec_grid_ind[i] + R_max_ind) <= 2048.)):

			grid_stamp = grid[RM_RA_grid_ind[i] - R_max_ind: RM_RA_grid_ind[i] + R_max_ind, RM_Dec_grid_ind[i] - R_max_ind: RM_Dec_grid_ind[i] + R_max_ind]
			
			if len(np.where(grid_stamp==0)[0]) == 0:
				#print "keep"
				#keep that RM source
				new_RM_Dec.append(RM_Dec[i])
				new_RM.append(RM[i])
				new_RM_RA.append(RM_RA[i])
			else:
				continue
		else:
			continue

	return new_RM_RA, new_RM_Dec, new_RM


def mask2(grid_vals, nside, RA_RM, Dec_RM, RM, radii_list):
	#grid_vals are in healpix format
	vals = grid_vals #number density grid values

	r = np.sort(radii_list)[-1] #measured in kpc - want rmax
	arcsec_per_kpc = cosmo1.arcsec_per_kpc_comoving(0.1).value #number of arcseconds per kpc
	r_deg = (r * arcsec_per_kpc)/3600.

	RM_delete = []
	RM_keep = []

	for i in np.arange(0,len(Dec_RM)):

		#edge RA/Dec
		#we get 4 combos from this
		RA_min = RA_RM[i] - r_deg
		RA_max = RA_RM[i] + r_deg
		Dec_min = Dec_RM[i] - r_deg
		Dec_max = Dec_RM[i] + r_deg

		#degrees to radians

		Dec_RM_rad=(np.pi/180.)*(90.-Dec_RM[i])
		RA_RM_rad=(np.pi/180.)*(RA_RM[i])

		Dec_min=(np.pi/180.)*(90.-Dec_min)
		RA_min=(np.pi/180.)*(RA_min)
		Dec_max=(np.pi/180.)*(90.-Dec_max)
		RA_max=(np.pi/180.)*(RA_max)

		#get vertice vectors for the postage stamp from the RA/Dec
		vect1 = hp.ang2vec(Dec_min,RA_min)
		vect2 = hp.ang2vec(Dec_min,RA_max)
		vect3 = hp.ang2vec(Dec_max,RA_max)
		vect4 = hp.ang2vec(Dec_max,RA_min)

		#put into format for other function
		vertices = np.array([vect1,vect2,vect3,vect4])

		#get polygon of pixels arround RM pix
		pix_matrix = hp.query_polygon(nside, vertices, inclusive=True)

		vals_matrix = vals[pix_matrix]


		if np.any(np.isnan(vals_matrix)) == False: #if len(np.where(vals_matrix == 0.)[0]) < len(np.where(vals_matrix != 0.)[0]):
			RM_keep.append(i)
		else:
			#continue
			RM_delete.append(i)

	RA_RM_keep = RA_RM[RM_keep]
	Dec_RM_keep = Dec_RM[RM_keep]
	RM_return = RM[RM_keep]


	return np.array(RA_RM_keep), np.array(Dec_RM_keep), np.array(RM_return)#,RM_keep #new_RM_RA, new_RM_Dec, new_RM, RM_delete, RM_keep



def total_num_gals_in_z(z_bin_list):
	total_gal_in_z = np.zeros(4)
	for i in np.arange(4):
		total_gal_in_z[i] = (np.where(z_bin_list == i)[0]).size
	return total_gal_in_z


#this will be a function so we'll return these new values 


def cross_corr(RA_RM, Dec_RM, RM_vals, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split):
	#input the galaxy catalogue number density (the 2048 grid one), for each redshift
	#input the RM catalogue 

	#I have to loop through each photo z slice to weight them?
	cross_corr_func = np.zeros(len(radii_list))

	radial_bins = radii_list 

	#calculating the average galaxy counts

	gal_counts_avg = np.zeros(len(radial_bins)-1)

	#list of all rho_r vals (for each RM source) for this input r
	rho_r = np.zeros((len(RA_RM),len(radial_bins)-1))

	for k in np.arange(0,len(RA_RM)): #can change this to only look at the first few

		#getting the distance from each cell in the galaxy num dens grid to the grid which the RM source reside
		#make a histogram of radial distances for each source

		RM_dist_grid = dist_matrix_kpc[k] 

		gal_counts_radial = np.zeros(len(radial_bins)-1)

		for i in np.arange(0, len(z_mean)):

			#which galazies in this distance stamp are in this specific redshift slice
			z_ind = np.where(dist_z_split[k] == i)

			RM_dist_grid_in_z = RM_dist_grid[z_ind]

			gal_counts_radial_z, bin_edges = np.histogram(RM_dist_grid_in_z, bins=radial_bins)


			#weight it myself (it is a bit sketchy inside the histogram)
			gal_counts_radial_z = gal_counts_radial_z*((1+z_mean[i])**(-2.))*(1./float(total_gal_in_z[i]))

			gal_counts_radial += gal_counts_radial_z
			
		#gal_counts_avg += gal_counts_radial


		rho_r[k] = gal_counts_radial

	#total galaxies around each RM source subtracted by the average of the total # of galaxies around each RM source
	#how galaxy density around each RM source deviates from the average density around all RM souces

	delt_rho_r = rho_r - np.average(rho_r, axis = 0) # 2dlist - 1dlist

	#the RM values minus the average of all RM values
	#how much do each RM values deviate from the average of all RM values
	delt_RM = RM_vals - np.average(RM_vals) # list - float

	rho_RM = np.zeros(len(np.average(rho_r, axis = 0)))

	for j in np.arange(0, len(rho_RM)):
		rho_RM[j] = np.average(delt_rho_r[:,j]*delt_RM, axis = 0) #numerator of cross corr func
		#rho_RM[j] = np.average(delt_rho_r[:,j], axis = 0)

	cross_corr_func = rho_RM/np.average(rho_r,axis=0) #numerator/denom, appending to a list for various r-values
	#print rho_RM,np.average(rho_r,axis=0),RM_vals.mean()
	return cross_corr_func, np.average(rho_r,axis=0)


def CCF_theta(cln,l_start=3,end_theta=2.,theta_samples=50):
    #print cln
    #theta_samples: number of samples between 0 and end_theta
    #more samples will require more computations

    cl_theta=np.zeros(theta_samples, dtype =float)
    ctheta=np.linspace(0.,end_theta,num=theta_samples)
    
    for x in range(0,len(cl_theta),1):
        #cl_arb is used in the loop to keep the summation values
        cl_arb=0.
        #Will run through the starting multipole value to the last
        for y in range(l_start,len(cln),1):
          #Legendre Polynomial 
            Pl=sp.lpmn(0,y,np.cos(np.radians(ctheta[x])))
	    ln=len(Pl[0][0][:])-1	
            arb=((2.*y+1.)/4*np.pi)*cln[y]*Pl[0][0][ln]
            cl_arb+= arb
            
            cl_theta[x]=cl_arb
    #return theta values used and respective CCF_theta values 
    return ctheta,cl_theta


def predict_numbdense(total_gal_in_z,z_mean,radii_list,sky_frac):

    kpcdeg = (1./cosmo1.arcsec_per_kpc_comoving(z_mean).value)*3600. #kpc per deg
    sqdegsky = 41253. #number of square degrees in sky
    areabins = np.pi*radii_list**2
    areabins1 = areabins[1:]-areabins[:-1]

    avegalz_sqkpc = total_gal_in_z/(sky_frac*kpcdeg**2*sqdegsky) #average number of galaxies per square kilopc per z bin
    avegalz = avegalz_sqkpc[:,np.newaxis]*areabins1[np.newaxis,:] #average number of galaxies per z and radial bin

    zweight = ((1+np.array(z_mean))**(-2.))*(1./(total_gal_in_z))
    zweight=(zweight/zweight.sum())*4.
    avegalzwt=avegalz*zweight[:,np.newaxis]
    avegalzwt_sqkpc=avegalz_sqkpc*zweight
    
    avegal_sqkpc=avegalz_sqkpc.sum() #average number of galaxies per square kilopc over all z bin
    avegal=avegalz.sum(axis=0)#average number of galaxies per radial bin over all z bin
    avegal_sqkpc_wt=(avegalzwt_sqkpc).sum() #weighted average number of galaxies per square kilopc over all z bin
    avegal_wt=(avegalzwt).sum(axis=0)#weight average number of galaxies per radial bin over all z bin

    #you can change this to return whatever you want
    return avegal_wt,avegalzwt#np.array([avegalz_sqkpc,avegalzwt_sqkpc]),np.array([avegalz,avegalzwt]),np.array([avegal_sqkpc,avegal_sqkpc_wt]),np.array([avegal,avegal_sqkpc_wt])



def cross_corr_predict(RA_RM, Dec_RM, RM_vals, dist_matrix_kpc, radii_list, z_mean, total_gal_in_z, dist_z_split):
    #input the galaxy catalogue number density (the 2048 grid one), for each redshift
    #input the RM catalogue 
    
    #I have to loop through each photo z slice to weight them?
    cross_corr_func = np.zeros(len(radii_list))
    
    radial_bins = radii_list 
    zwt=((1+np.array(z_mean))**(-2.))*(1./(np.array(total_gal_in_z)))
    print "inside cross_corr_predict april 11:"
    print "z-weighting 1 : ", zwt
    zwt=zwt/zwt.sum()*4.
    print "z-weighting 2 : ", zwt

    print "inside cross_corr_predict april 11:"
    print "z-weighting: ", zwt

    #calculating the average galaxy counts
    areabins = np.pi*radii_list**2
    areabins1 = areabins[1:]-areabins[:-1]
    print "areabins: ", areabins1
    gal_counts_avg = np.zeros(len(radial_bins)-1)

    #list of all rho_r vals (for each RM source) for this input r
    rho_r = np.zeros((len(RA_RM),len(radial_bins)-1))
    aa,bb=predict_numbdense(total_gal_in_z,z_mean,radii_list,0.75) # = avegal_wt,avegalzwt

    print "calling predict_numbdense here :"
    print  "avegal_wt, avegalzwt: ",  aa, bb

    gal_counts_radial = np.zeros((len(RA_RM),len(radial_bins)-1))
    gal_counts_radial_z = np.zeros((len(RA_RM),len(radial_bins)-1,4))
    
    for k in np.arange(0,len(RA_RM)): #can change this to only look at the first few

    #getting the distance from each cell in the galaxy num dens grid to the grid which the RM source reside
    #make a histogram of radial distances for each source

        RM_dist_grid = dist_matrix_kpc[k] #already a numpy array?

        for i in np.arange(0, len(z_mean)):

        #which galazies in this distance stamp are in this specific redshift slice
            z_ind = np.where(dist_z_split[k] == i)

            RM_dist_grid_in_z = RM_dist_grid[z_ind]

            gal_counts_radial_z[k,:,i], bin_edges = np.histogram(RM_dist_grid_in_z, bins=radial_bins)

    
            #weight it myself (it is a bit sketchy inside the histogram)
            gal_counts_radial_z[k,:,i] = (gal_counts_radial_z[k,:,i]*zwt[i])#-bb[i,:]

            gal_counts_radial[k,:] += gal_counts_radial_z[k,:,i]
		
            #gal_counts_avg += gal_counts_radial


        rho_r[k,:] = gal_counts_radial[k,:]
    
    #total galaxies around each RM source subtracted by the average of the total # of galaxies around each RM source
    #how galaxy density around each RM source deviates from the average density around all RM souces

    gal_counts_radial_z1=gal_counts_radial_z-bb.transpose()[np.newaxis,:,:]
    delt_rho_r = gal_counts_radial_z.sum(axis=2)/areabins1[np.newaxis,:]#rho_r #- np.average(rho_r, axis = 0) # 2dlist - 1dlist

    #the RM values minus the average of all RM values
    #how much do each RM values deviate from the average of all RM values
    delt_RM = RM_vals - np.average(RM_vals) # list - float

    rho_RM = np.zeros(len(np.average(rho_r, axis = 0)))

    for j in np.arange(0, len(rho_RM)):
        rho_RM[j] = np.average(delt_rho_r[:,j]*delt_RM, axis = 0) #numerator of cross corr func
        #rho_RM[j] = np.average(delt_rho_r[:,j], axis = 0)

    cross_corr_func = rho_RM#/np.average(rho_r,axis=0) #numerator/denom, appending to a list for various r-values

    return cross_corr_func,np.average(rho_r,axis=0)



def cross_corr_plot(radii_list, cross_corr_func):
	#plotting the cross correlation function
	radii_list = [(a + b) /2. for a, b in zip(radii_list[::], radii_list[1::])]
	plt.figure()
	plt.plot(radii_list, cross_corr_func, 'r-', markersize = 30)
	plt.plot(radii_list, cross_corr_func, 'b+', markersize = 30)
	plt.title("Cross Correlation Function with old foreground")
	plt.xlabel('Radii [kpc]')
	plt.ylabel('Cross-Correlation')
	plt.grid()
	plt.xscale('log')
	#put an x lim on this!!
	plt.figtext(0,0,"Subtracting galactic foreground with old method (avg RM around each source)")
	plt.savefig(pwd_plots+"cross_corr_plot_wise_taylor_old_foreground.png")
	#plt.show()
	return

def cross_corr_plot_scramble(radii_list, cross_corr_func, cross_corr_real,n):
	#plotting the cross correlation function
	radii_list = [(a + b) /2. for a, b in zip(radii_list[::], radii_list[1::])]
	plt.figure()
	plt.plot(radii_list, cross_corr_func, 'r-', linewidth = 2, markersize = 30, label = 'Scrambeled RM for '+str(n)+' runs')
	plt.plot(radii_list, cross_corr_func, 'b+', markersize = 30)
	plt.plot(radii_list, cross_corr_real, 'g-', linewidth = 2, markersize = 30, label = 'Real RM')
	plt.plot(radii_list, cross_corr_real, 'm+', markersize = 30)
	plt.title("Average Scrambled Cross Correlation Function for "+str(n)+" times")
	plt.xlabel('Radii [kpc]')
	plt.ylabel('Cross-Correlation')
	plt.legend()
	plt.grid()
	plt.xscale('log')
	#put an x lim on this!!
	plt.figtext(0,0,"Subtracting galactic foreground with Niels' foreground, scrambling RM sources")
	plt.savefig(pwd_plots+"cross_corr_plot_wise_taylor_scrambled.png")
	#plt.show()
	return

def num_dens_plot(num_dens_grid, xedges, yedges):
	#plotting the galaxy num dens grid
	fig = plt.figure(figsize=(7, 3))
	ax = fig.add_subplot('111',title='Galaxy Number Density Grid',aspect='equal')
	X, Y = np.meshgrid(xedges, yedges)
	GRID = ax.pcolormesh(X, Y, num_dens_grid)
	plt.colorbar(GRID, ax=ax) 
	plt.savefig(pwd_plots+"num_dens_grid_wise_taylor_logbins.png")
	return
    






