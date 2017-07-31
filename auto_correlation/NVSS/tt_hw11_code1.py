import numpy as np
import scipy.special

import matplotlib.pyplot as plt

def bayes(x, y, rho):
    n = N = x.size
    m = y.size
    mean_x = np.sum(x)/n
    mean_y = np.sum(y)/m
    
    r = np.sum( (x - mean_x) * (y - mean_y) )/np.sqrt( np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2) )

    likelihoods = (1-rho**2)**((N-1)/2) / (1- rho*r)**(N-3/2) * (1 + 1/(N-1/2) * (1 + rho*r)/8)
    return likelihoods / np.sum(likelihoods)
 
    
    
def Vmax_method(luminosities, bin_edges, flux_limit):
    V_max = 4*np.pi/3 * (luminosities / (4*np.pi*flux_limit))**(3/2)
    V_max_histogram = np.zeros(bin_edges.size - 1)
    for i in range(bin_edges.size-1):
        in_l_bin = np.logical_and(bin_edges[i] < luminosities, luminosities < bin_edges[i+1])
        V_max_histogram[i] = np.sum( 1/V_max[in_l_bin] )
        
    return V_max_histogram
    
def Vmax_bootstrap_errors(luminosities, bin_edges, flux_limit, trails):
    V_max_histogram_bootstrap = np.zeros((trails, bin_edges.size-1) )
    for i in range(trails):
        bootstrap_objects = np.random.choice(luminosities, size=luminosities.size, replace=True)
        V_max_histogram_bootstrap[i] = Vmax_method(bootstrap_objects, bin_edges, flux_limit)
        
    return np.std(V_max_histogram_bootstrap, axis=0)
 

def create_survey(luminosities, radii, bin_edges, flux_limit):
    #Objects with flux greater than S_lim
    above_flux_limit = flux_limit < luminosities/(4*np.pi*radii**2)

    survey_luminosities = luminosities[above_flux_limit]
    N_survey = survey_luminosities.size
    
    survey_histogram, survey_histogram_edges = np.histogram( survey_luminosities, bin_edges )
    objects_histogram, objects_histogram_edges = np.histogram( luminosities, bin_edges )
        
    return above_flux_limit, survey_histogram, objects_histogram, N_survey
    
def Anvi_estimator(n, u, M, normalization=1):
    #Upper limits -> reverse order
    u = u[::-1]
    n = n[::-1]
    
    p = np.zeros_like(n, dtype=float)

    p[0] = n[0]/(M-u[0])
    for k in range(1,survey_histogram.size):
        B = 1 - np.cumsum(p[:k])
        B = np.concatenate( (np.array([1]), B) )
        A = np.sum( u[:k+1]/B )
        p[k] = n[k] / (M - A )
        
    return p[::-1]*normalization
  
def Anvi_estimator_errors(luminosities, upper_limits, M, bin_edges, normalization, trails):
    anvi_bootstrap = np.zeros((trails, bin_edges.size-1) )
    for i in range(trails):
        bootstrap_luminosities = np.random.choice(luminosities, size=luminosities.size, replace=True)
        bootstrap_upper_limits = np.random.choice(upper_limits, size=upper_limits.size, replace=True)
        n, b = np.histogram(bootstrap_luminosities, bin_edges)
        u, b = np.histogram(bootstrap_upper_limits, bin_edges)
        anvi_bootstrap[i] = Anvi_estimator(n, u, M, normalization)
        
    return np.std(anvi_bootstrap, axis=0)
    
def sanson_flamsteed_projection(phi, theta):
    x = phi*np.cos(theta)
    y = theta
    
    return np.array([x, y])
    
def aitoff_projection(phi, theta):
    z = np.sqrt(2/(1 + np.cos(theta)*np.cos(phi/2)))
    
    x = 2*z * np.cos(theta)*np.sin(phi/2) / np.sin(z)
    y = z * np.sin(theta)/np.sin(z)
    
    return np.array([x, y])
    
R = 1.0
N = 10**6
density = N * 3/(4 * np.pi * R**3)

L_min = 1.0
L_max = 10**4
L_0 = (L_min**(-2) - L_max**(-2))/2

S_lim_min = L_max/(4*np.pi*R**2)
S_lim = S_lim_min * 0.01


log_bins = np.logspace(np.log10(L_min), np.log10(L_max), 50)
log_L = np.logspace(np.log10(L_min), np.log10(L_max), 1000)

log_bin_sizes = log_bins[1:] - log_bins[0:-1]
log_bin_coords = (log_bins[1:] + log_bins[0:-1])/2

print("Flux limit: ", S_lim)


objects = np.empty( (N, 4) )

#r
objects[:,0] = (np.random.rand(N) )**(1/3)
#luminosity set 1
objects[:,1] = (-2*L_0*np.random.rand(N) + L_min**(-2))**(-1/2)
#phi
objects[:,2] = 2*np.pi * (np.random.rand(N) - 0.5)
#theta
objects[:,3] = np.pi * (np.random.rand(N) - 0.5)



survey_object_indicies, survey_histogram, objects_histogram, N_survey = create_survey(objects[:,1], objects[:,0], log_bins, S_lim)

print("Sample size: ", N_survey)

objects_in_survey = objects[survey_object_indicies,:]

flux = objects_in_survey[:,1]/(4*np.pi*objects_in_survey[:,0]**2)

longitudes = np.linspace(-np.pi, np.pi, 7)
latitudes = np.linspace(-np.pi/2, np.pi/2, 7)

phi = np.linspace(-np.pi, np.pi, 100)
theta = np.linspace(-np.pi/2, np.pi/2, 100)
#Sanson-Flamsted
sfx, sfy = sanson_flamsteed_projection(objects_in_survey[:,2], objects_in_survey[:,3])
scv = np.zeros( (longitudes.size, 2, theta.size) )
sch = np.zeros( (latitudes.size, 2, phi.size) )
for i in range(longitudes.size):
    scv[i,:] = sanson_flamsteed_projection(longitudes[i]*np.ones_like(theta), theta )
for i in range(latitudes.size):
    sch[i,:] = sanson_flamsteed_projection(phi, latitudes[i]*np.ones_like(phi) )
    
#Aitoff
ax, ay = aitoff_projection(objects_in_survey[:,2], objects_in_survey[:,3])
acv = np.zeros( (longitudes.size, 2, theta.size) )
ach = np.zeros( (latitudes.size, 2, phi.size) )
for i in range(longitudes.size):
    acv[i,:] = aitoff_projection(longitudes[i]*np.ones_like(theta), theta )
for i in range(latitudes.size):
    ach[i,:] = aitoff_projection(phi, latitudes[i]*np.ones_like(phi) )


fig1, (sanson_flamsteed_plot, aitoff_plot) = plt.subplots(2, 1)

sanson_flamsteed_plot.scatter(x=sfx, y=sfy, s=flux/4, marker='o', linewidth=0, alpha=0.5)
for i in range(longitudes.size):
    sanson_flamsteed_plot.plot(sch[i,0], sch[i,1], color="black")
for i in range(latitudes.size):
    sanson_flamsteed_plot.plot(scv[i,0], scv[i,1], color="black")
    
sanson_flamsteed_plot.set_axis_off()
sanson_flamsteed_plot.set_title("Sanson-Flamsteed projection")

aitoff_plot.scatter(x=ax, y=ay, s=flux/4, marker='o', linewidth=0, alpha=0.5)
for i in range(longitudes.size):
    aitoff_plot.plot(ach[i,0], ach[i,1], color="black")
for i in range(latitudes.size):
    aitoff_plot.plot(acv[i,0], acv[i,1], color="black")

aitoff_plot.set_axis_off()
aitoff_plot.set_title("Aitoff projection")

fig1.show()
fig1.savefig("tt_hw11_projections.png")
