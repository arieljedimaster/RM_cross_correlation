import numpy as np
import pylab as pyl
import pyfits as pf
import matplotlib.pyplot as plt
from scipy import *

num_clusters=500 # number of clusters

cluster_size=2 #degrees cluster size

max_sources=5000 #maximum number of sources per cluster

min_sources=500 #minimum number of sources per cluster

#picking random x and y positions to place the clusters 
#note this is not in RA/Dec
xpos_cluster=np.random.rand(num_clusters)*200-100
ypos_cluster=np.random.rand(num_clusters)*200-100

#randomly picking how big each cluster would be
sizes = np.random.randint(min_sources,high=max_sources,size=num_clusters)

#list of all x and y positions in each cluster
#each index refers to a different cluster
xx=[]
yy=[]

#bins for the histogram
bins=np.linspace(0,5.,10)
bins1=.5*(bins[1:]+bins[:-1])


dist_hist=np.zeros((num_clusters,len(bins1))) # why 9? is that the number of bins?

#calculating the clustering
for i in range(num_clusters):
    mean=[xpos_cluster[i],ypos_cluster[i]]
    cov= [[cluster_size/2.3548,0],[0,cluster_size/2.3548]]
    x1, y1 = np.random.multivariate_normal(mean, cov, sizes[i]).T
    xx.append(x1)
    yy.append(y1)
    #calculating the distance between everything in that cluster?
    dd=sqrt((xpos_cluster[i]-x1)**2+(ypos_cluster[i]-y1)**2) #calculating the actual distances in that cluster
    dist_hist[i,:],dhx=np.histogram(dd,bins=bins) #histogram of distances


#computing for all sources distance,not just those known to be in the right cluster
dist_hist2=np.zeros((num_clusters,len(bins1)))
for i in range(num_clusters):
    dd=sqrt((xpos_cluster[i]-np.concatenate(xx))**2+(ypos_cluster[i]-np.concatenate(yy))**2)
    dist_hist2[i,:],dhx=np.histogram(dd,bins=bins)

plt.figure()    
plt.plot(x1,y1,'kx')
plt.show()


#taking the average 
dist_hist_avg=distsh.mean(axis=0)
dist_hist2_avg=dist_hist2.mean(axis=0)

#mean subtracting so we can do the cross-correlation
dist_hist_mean_sub=dist_hist-dist_hist_avg[np.newaxis,:]
dist_hist2_mean_sub=dist_hist2-dist_hist2_avg[np.newaxis,:]


RMs=sizes.copy()
RMs_mean_sub=RMs-RMs.mean()

ccf=(dist_hist_mean_sub*RMs_mean_sub[:,np.newaxis]).mean(axis=0)
ccf2=(dist_hist2_mean_sub*RMs_mean_sub[:,np.newaxis]).mean(axis=0)

RMs_std=RMs.std()
dists_std=dist_hist.std(axis=0)
dists2_std=dist_hist2.std(axis=0)


##CCF traces shape of gaussian with positive correlation

plt.figure()
plt.subplot(211)
plt.title('Un-Normalized CCF')
plt.plot(bins1,ccf,'rd-', label = 'distances for all clusters')
plt.plot(bins1,ccf2,'ko-', label = 'distances within one cluster')
plt.legend()
## properly normalized CCF
plt.subplot(212)
plt.title('Normalized CCF')
plt.plot(bins1,ccf/(RMs_std*dists_std),'ko-', label = 'distances within one cluster')
plt.plot(bins1,ccf2/(RMs_std*dists2_std),'rd-', label = 'distances for all clusters')
plt.legend()
plt.show()






### from here you can make it more complicated
###  -- add in some background points everywhere not assocaited with the clusters
###  -- move the "RM" positions randomly off center from the clusters
###  -- have clusters with slightly varying sizes
###  -- use a different tracer than the size as the "RM" value (something slightly less correlated
###  -- add in some RM points that are not in clusters