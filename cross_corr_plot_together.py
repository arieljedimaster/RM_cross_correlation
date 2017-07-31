#plotting all of the cross_corr runs all together

import numpy as np 
import matplotlib.pyplot as plt 

# upload cross_corr1
#norm
cross_corr1, radii_list1 = np.loadtxt('cross_corr1.txt')
#reg
cross_corr1_reg, radii_list1_reg = np.loadtxt('cross_corr_reg1.txt')
#un-norm
cross_corr1_unnorm, radii_list1_unnorm = np.loadtxt('cross_corr_unnorm1.txt')


#cross_corr_2
#norm
cross_corr2, radii_list2 = np.loadtxt('cross_corr2.txt')
#reg
cross_corr2_reg, radii_list2_reg = np.loadtxt('cross_corr_reg2.txt')
#unnorm
cross_corr2_unnorm, radii_list2_unnorm = np.loadtxt('cross_corr_unnorm2.txt')


#cross_corr_3
#norm
cross_corr3, radii_list3 = np.loadtxt('cross_corr3.txt')
#reg
cross_corr3_reg, radii_list3_reg = np.loadtxt('cross_corr_reg3.txt')
#unnorm
cross_corr3_unnorm, radii_list3_unnorm = np.loadtxt('cross_corr_unnorm3.txt')
'''
#cross_corr_4
#norm
cross_corr4, radii_list4 = np.loadtxt('cross_corr4.txt')
#reg
cross_corr4_reg, radii_list4_reg = np.loadtxt('cross_corr4_reg.txt')
#unnorm
cross_corr4_unnorm, radii_list4_unnorm = np.loadtxt('cross_corr4_unnorm.txt')

#cross_corr_5
#norm
cross_corr5, radii_list5 = np.loadtxt('cross_corr5.txt')
#reg
cross_corr5_reg, radii_list5_reg = np.loadtxt('cross_corr5_reg.txt')
#unnorm
cross_corr5_unnorm, radii_list5_unnorm = np.loadtxt('cross_corr5_unnorm.txt')
'''

####################################################


plt.figure()
plt.plot(radii_list1,cross_corr1, linewidth = 2, label = "bins = np.logspace(1,3.04, 9)")
plt.plot(radii_list2,cross_corr2, linewidth = 2, label = "bins = np.logspace(1,3.04, 5)")
plt.plot(radii_list3,cross_corr3, linewidth = 2, label = "bins = np.logspace(1,3.04,14)")
#plt.plot(radii_list4,cross_corr4, linewidth = 2)
#plt.plot(radii_list5,cross_corr5, linewidth = 2)
plt.title("Testing different radial bins for fully normalized Cross Corr")
plt.xlabel('Radial Bin [Mpc/h]')
plt.ylabel('Cross Correlation')
plt.xscale('log')
plt.legend()
plt.show()

plt.figure()
plt.plot(radii_list1_reg,cross_corr1_reg, linewidth = 2, label = "bins = np.logspace(1,3.04, 9)")
plt.plot(radii_list2_reg,cross_corr2_reg, linewidth = 2, label = "bins = np.logspace(1,3.04, 5)")
plt.plot(radii_list3_reg,cross_corr3_reg, linewidth = 2, label = "bins = np.logspace(1,3.04,14)")
#plt.plot(radii_list4_reg,cross_corr4_reg, linewidth = 2)
#plt.plot(radii_list5_reg,cross_corr5_reg, linewidth = 2)
plt.title("Testing different radial bins for Cross Corr (div by rho only)")
plt.xlabel('Radial Bin [Mpc/h]')
plt.ylabel('Cross Correlation')
plt.xscale('log')
plt.legend()
plt.show()

plt.figure()
plt.plot(radii_list1_unnorm,cross_corr1_unnorm, linewidth = 2, label = "bins = np.logspace(1,3.04, 9)")
plt.plot(radii_list2_unnorm,cross_corr2_unnorm, linewidth = 2, label = "bins = np.logspace(1,3.04, 5)")
plt.plot(radii_list3_unnorm,cross_corr3_unnorm, linewidth = 2, label = "bins = np.logspace(1,3.04,14)")
#plt.plot(radii_list4_unnorm,cross_corr4_unnorm, linewidth = 2)
#plt.plot(radii_list5_unnorm,cross_corr5_unnorm, linewidth = 2)
plt.title("Testing different radial bins for un-normalized Cross Corr")
plt.xlabel('Radial Bin [Mpc/h]')
plt.ylabel('Cross Correlation')
plt.xscale('log')
plt.legend()
plt.show()


