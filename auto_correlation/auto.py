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


#WISE number density map 
#dont need RM maps at all

################################################################################################################################
################################################################################################################################


#upload WISE catalogue

pwd_data = '/Users/arielamaral/Documents/AST430/Data/'

WISE_cat = fits.open(pwd_data+'WISExSCOS.photoZ.MAIN.fits')
WISE_header = WISE_cat[1].header
WISE_data = WISE_cat[1].data

gal_RA = WISE_data['ra_WISE ']
gal_Dec = WISE_data['dec_WISE']
gal_photo_z = WISE_data ['zANNz   ']


##################################


#what resolution should we use?

res=10 #SWITCH THIS TO CHANGE UP RESOLUTION
nside=2**res
npix=12*nside**2

radii_list = np.logspace(1.,3.4,9)
radii_list[0] = 0.


print "total number of galaxy sources: ", len(gal_RA)

gal_RA, gal_Dec, gal_photo_z = bf.z_reduc(gal_RA, gal_Dec, gal_photo_z)

print "total number of galaxy sources after filtering out 0.1 < z < 0.5: ", len(gal_RA)
print " "

z_split, z_mean, gal_RA_split, gal_Dec_split, z_bin_list = bf.photo_z_split(gal_photo_z, gal_RA, gal_Dec, bin_width =0.1)

'''
############################################
#using only a subset of WISE to test pair_estimator function


gal_Dec_min = 50.
gal_Dec_max = 60.
gal_RA_min = 230.
gal_RA_max = 240.


gal_ind = np.argwhere((gal_RA_min < gal_RA) & (gal_RA <  gal_RA_max) &(gal_Dec_min < gal_Dec) & (gal_Dec < gal_Dec_max))[:,0]


print "Number of galaxies after cropping a small testing region: ", len(gal_ind)

gal_RA = gal_RA[gal_ind]
gal_Dec = gal_Dec[gal_ind]
gal_photo_z = gal_photo_z[gal_ind]




theta_list = np.linspace(0., 4., 8) #in degrees


##################################

#randomized galaxy RA and Dec

gal_RA_rand = np.copy(gal_RA)
gal_Dec_rand = np.copy(gal_Dec)
gal_photo_z_rand = np.copy(gal_photo_z)


np.random.shuffle(gal_RA_rand)
np.random.shuffle(gal_Dec_rand)
np.random.shuffle(gal_photo_z_rand)
'''



'''
###########################################

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


#save these guys
dist_pwd = '/Users/arielamaral/Documents/AST430/Code/Code_new_cat/simulated_data/distances/'


'''
'''
#save these because they're gonna take a while to run

distances_DD = distance_func_gal(gal_RA, gal_Dec, gal_RA, gal_Dec, kpc2deg_list, theta_list)
distances_DR = distance_func_gal(gal_RA, gal_Dec, gal_RA_rand, gal_Dec_rand, kpc2deg_list_rand, theta_list)
distances_RR = distance_func_gal(gal_RA_rand, gal_Dec_rand, gal_RA_rand, gal_Dec_rand, kpc2deg_list_rand, theta_list)

np.save(dist_pwd+"dist_matrix_deg_DD.npy", distances_DD)
np.save(dist_pwd+'dist_matrix_deg_DR.npy',distances_DR)
np.save(dist_pwd+'dist_matrix_deg_RR.npy',distances_RR)
'''



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




#####################################################


#making number density grid

num_dens_total = bf.healpy_grid2(res,gal_RA,gal_Dec)#np.zeros(npix)
num_dens_zbin = np.zeros((4,npix))

#setting all zero pixels (the galactic plane) to NaN
num_dens_total[num_dens_total==0.] = np.nan
num_dens_total_masked = hp.pixelfunc.ma(num_dens_total)
num_dens_total_masked.mask = np.isnan(num_dens_total_masked)


num_dens_zbin_masked=[]


for i in np.arange(0,len(z_split)):
    #dont want 'nan's' though right now
    num_dens_grid_split = bf.healpy_grid2(res,gal_RA_split[i],gal_Dec_split[i])
    num_dens_grid_split_a = bf.healpy_grid2(res,gal_RA_split[i],gal_Dec_split[i])

    #subtract off the mean
    num_dens_grid_split = hp.ma(num_dens_grid_split)
    num_dens_grid_split.mask = num_dens_total_masked.mask
    num_dens_grid_split -= np.mean(num_dens_grid_split) #this line makes it equal to Tessa's num_dens_zbinaves
    num_dens_zbin_masked.append(num_dens_grid_split)


num_dens_total[num_dens_total==0.] = np.nan
num_dens_total_masked = hp.pixelfunc.ma(num_dens_total)
num_dens_total_masked.mask = np.isnan(num_dens_total_masked)


'''
#######################################################
# generating a random relatization of the number map?
#shuffling the number density map:

num_dens_shuffle = np.random.shuffle(num_dens_total_masked)
#does it accurately ignore the masked out cells
#does the numpy masking carry over to the healpy masking?
num_dens_shuffle_masked = np.ma.array(num_dens_shuffle, mask = num_dens_total_masked.mask)
'''

print " "
print "Completed generating number density grids"
print " "



########################################################


def cell_estimator(theta_list, num_dens_data,res):
    nside=2**res
    npix=12*nside**2
    print "nside is for cell estimator 1 is", nside

    w_theta = np.zeros(len(theta_list))

    n_avg = np.average(num_dens_data)

    #looping through each theta (radial) bin
    for alpha in np.arange(0,len(theta_list)):
        print "auto-correlating for theta = ", theta_list[alpha]

        #should this be in the alpha loop or the "i" loop?
        numerator = 0.

        #the theta denominator of the equation
        Theta_ij_sum = 0.

        #looping through each cell
        for i in [30]:#np.arange(0, len(num_dens_data)):
            #calculates how the number density differs from the average for that cell
            delta_g_i = np.float(num_dens_data[i] - n_avg)/np.float(n_avg)
            i_ang = hp.pixelfunc.pix2ang(nside, i)


            #looping through every other cell to compare to the cell "i" in the fist loop
            for j in np.arange(0, len(num_dens_data)):
                #getting the angular distances between the pixels
                #you need to calculate these vector thingys to do that
                j_ang = hp.pixelfunc.pix2ang(nside, j)
                #getting the distances between each cell and the reference cell
                dist_ij = hp.rotator.angdist(i_ang, j_ang)
                #Theta_ij is 1 if the sep between cells is within angular bin theta_alpha, and zero otherwise
                Theta_ij = 1. if np.absolute(dist_ij) <= (theta_list[alpha]*(np.pi/180.)) else 0.
                if Theta_ij == 1.:
                    print "pixel number within theta[alpha]: ",j
                else:
                    continue

                # should this maybe be num_dens_rand[j]???? - Ask Tessa
                #calculates how the number density differs from the average for that cell
                delta_j_i = np.float(num_dens_data[j] - n_avg)/np.float(n_avg)

                Theta_ij_sum += Theta_ij

            print "Theta_ij_sum: ", Theta_ij_sum

        w_theta[alpha] = numerator/Theta_ij_sum
        print "it's working, w_[",alpha,"] = ", w_theta[alpha]

    return w_theta

def cell_estimator2(theta_list, num_dens_data,res):
    nside=2**res
    npix=12*nside**2
    print "nside is for cell estimator 2 is", nside

    w_theta = np.zeros(len(theta_list))

    n_avg = np.average(num_dens_data)
    #n_avg_rand = np.average(num_dens_rand)

    #looping through each theta (radial) bin
    for alpha in np.arange(0,len(theta_list)):
        print "auto-correlating for theta = ", theta_list[alpha]
        #which loop should this be in?
        numerator = 0.
        Theta_ij_sum = 0.
        #looping through each cell

        for i in [30]:#np.arange(0, len(num_dens_data)):
            #find all pixels within the angular bin, alpha
            radius = theta_list[alpha]*(np.pi/180.)
            vec = hp.pix2vec(nside,i)
            print "nside going into query disc: ", nside
            pix_in_alpha = hp.query_disc(nside, vec, radius, inclusive=True)
            print "number of pixels inside theta[alpha], ", len(pix_in_alpha)
            print "the pixel indeces: ", pix_in_alpha

            Theta_ij = len(pix_in_alpha)
            Theta_ij_sum += Theta_ij
            print "Theta_ij_sum: ", Theta_ij_sum
            #calculates how the number density differs from the average for that cell
            delta_g_i = np.float(num_dens_data[i] - n_avg)/np.float(n_avg)

            #looping through every other cell to compare to the cell "i" in the fist loop
            for j in pix_in_alpha:
                delta_j_i = np.float(num_dens_data[j] - n_avg)/np.float(n_avg)
                #print "delta_j_i: ", delta_j_i
                numerator += delta_g_i*delta_j_i*1. #because Theta_ij = 1.
                #print "numerator: ", numerator
        w_theta[alpha] = numerator/Theta_ij_sum
        print "it's working, w_[",alpha,"] = ", w_theta[alpha]

    return w_theta


#GENERATING A FAKE HEALPY MAP OF EASY NUMBERS TO TEST THE PAIR ESTIMATOR
#fake resolution

res_fake=2 

nside_fake=2**res_fake
npix_fake=12*nside_fake**2
#setting all values to 1
fake_values=np.zeros(npix_fake) + 1.

print fake_values

print "degree resolution of an individual pixel: ", hp.nside2resol(nside_fake, arcmin = True)/60.

#lets try running it just for 2 values
#does theta_list have to be in degrees? I think so...

theta_list = np.array([30.]) #units? I dont think this is in degrees
w_theta_pair1 = cell_estimator(theta_list, fake_values, res_fake)
w_theta_pair2 = cell_estimator2(theta_list, fake_values, res_fake)

print "Output of the pair estimator 1 for theta list: ", theta_list, " is: ", w_theta_pair1
print "Output of the pair estimator 2 for theta list: ", theta_list, " is: ", w_theta_pair2

'''
plt.figure()
plt.plot(theta_list, w_theta_pair1, 'ro', label = 'method 1')
plt.plot(theta_list, w_theta_pair2, 'bo', label = 'method 2')
plt.title("Test output for w_theta for fake map and theta_list: " + str(theta_list))
plt.xlabel("Radial Bins [degrees]")
plt.ylabel("Autocorrelation w(theta)")
plt.legend()
plt.show()
'''



