import numpy as np
import pylab as pyl
import pyfits as pf
from scipy import 8
#from astroML.correlation import bootstrap_two_point_angular


##simplest thing I can think of is generate random points where we know the clustering shape, i.e. a 2D gaussian.
#Give them all a characteristic cluster scale size and a different number of points in each
#(things cluster on the same scale but have different amounts of clustering, ie number density per cluster).
#Then choose your "RM" point as the center of the cluster, calculate distance from each "nearby" or cluster point to the RM point.
#Compute CCF using the cluster size as an RM value, that should give a near perfect positive correlation with the shape of the Gaussian.

nclust=500 # number of clusters

clsize=2 #degrees cluster size

maxsrcs=5000 #maximum number of sources per cluster

minsrcs=500 #minimum number of sources per cluster

#picking random x and y positions to place the clusters 
#note this is not in RA/Dec
xposcl=np.random.rand(nclust)*200-100
yposcl=np.random.rand(nclust)*200-100

#randomly picking how big each cluster would be
sizes=np.random.randint(minsrcs,high=maxsrcs,size=nclust)

xx=[]
yy=[]

#what are these bins?
bins=np.linspace(0,5.,10)
bins1=.5*(bins[1:]+bins[:-1])


distsh=np.zeros((nclust,9))
for i in range(nclust):
    mean=[xposcl[i],yposcl[i]]
    cov= [[clsize/2.3548,0],[0,clsize/2.3548]]
    x1, y1 = np.random.multivariate_normal(mean, cov, sizes[i]).T
    xx.append(x1)
    yy.append(y1)
    #da=sqrt((x1.reshape(1,x1.size)-x1.reshape(x1.size,1))**2+(y1.reshape(1,x1.size)-y1.reshape(x1.size,1))**2)
    dd=sqrt((xposcl[i]-x1)**2+(yposcl[i]-y1)**2)
    distsh[i,:],dhx=np.histogram(dd,bins=bins)


#computing for all sources distance,not just those known to be in the right cluster
distsh2=np.zeros((nclust,9))
for i in range(nclust):
    dd=sqrt((xposcl[i]-np.concatenate(xx))**2+(yposcl[i]-np.concatenate(yy))**2)
    distsh2[i,:],dhx=np.histogram(dd,bins=bins)

    
pyl.plot(x1,y1,'kx')
pyl.show()
    
distsh_avs=distsh.mean(axis=0)
distsh_nm=distsh-distsh_avs[np.newaxis,:]

distsh2_avs=distsh2.mean(axis=0)
distsh2_nm=distsh2-distsh2_avs[np.newaxis,:]


rmms=sizes.copy()
rmms_nm=rmms-rmms.mean()

ccf=(distsh_nm*rmms_nm[:,np.newaxis]).mean(axis=0)
ccf2=(distsh2_nm*rmms_nm[:,np.newaxis]).mean(axis=0)

rmms_sd=rmms.std()
dists_sd=distsh.std(axis=0)
dists2_sd=distsh2.std(axis=0)


#pyl.plot(rmms,distsh[:,3],'k.')
#pyl.show()

##CCF traces shape of gaussian with positive correlation

pyl.plot(bins1,ccf,'ko-')
pyl.plot(bins1,ccf2,'rd-')
pyl.show()


## properly normalized CCF

pyl.plot(bins1,ccf/(rmms_sd*dists_sd),'ko-')
pyl.plot(bins1,ccf2/(rmms_sd*dists2_sd),'rd-')
pyl.show()






### from here you can make it more complicated
###  -- add in some background points everywhere not assocaited with the clusters
###  -- move the "RM" positions randomly off center from the clusters
###  -- have clusters with slightly varying sizes
###  -- use a different tracer than the size as the "RM" value (something slightly less correlated
###  -- add in some RM points that are not in clusters
