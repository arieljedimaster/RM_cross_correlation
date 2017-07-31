import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
#import scipy.weave

c_code = """
int d;

for(int i=0; i<Nx; ++i)
{
    for(int j=i+1; j<Nx; ++j)
    {
        d = sqrt( (x[i]-x[j])*(x[i]-x[j]) + (y[i]-y[j])*(y[i]-y[j]) );
        for(int k=0; k<Nbins-1; ++k)
        {
            if(d > bins[k] && d < bins[k+1])
            {
                histogram[k] += 1;
                break;
            }
        }
    }
}
"""

def angular_auto_correlation_function_binned_weave(x, y, bins):
    histogram = np.zeros(bins.size-1)
    
    #scipy.weave.inline(
    
def angular_auto_correlation_function_binned(x, y, bins):
    N = x.size
    max_shift = 0
    histogram = np.zeros(bins.size-1)
    if N%2 == 0:
        max_shift = int(N/2 - 1)
    else:
        max_shift = int((N-1)/2)
        
    for i in range(1, max_shift + 1):
        distances = np.sqrt((x-np.roll(x,i))**2 + (y-np.roll(y,i))**2)
#         for d in distances:
#             mask = np.logical_and(d > bins[:-1], d < bins[1:])
#             histogram[mask] += 1
        h, b = np.histogram(distances, bins)
        histogram += h
     
    if N%2 == 0:
        shift = max_shift + 1
        distances = np.sqrt((x[:shift]-np.roll(x,shift)[:shift])**2 + (y[:shift]-np.roll(y,shift)[:shift])**2)
        h, b = np.histogram(distances, bins)
        histogram += h
    
    return histogram
    
def angular_cross_correlation_function_binned(x1, y1, x2, y2, bins):
    #Assume x1.size = x2.size
    N = x1.size
    histogram = np.zeros(bins.size-1)
        
    for i in range(0, N):
        distances = np.sqrt((x1-np.roll(x2,i))**2 + (y1-np.roll(y2,i))**2)
#         for d in distances:
#             mask = np.logical_and(d > bins[:-1], d < bins[1:])
#             histogram[mask] += 1
        h, b = np.histogram(distances, bins)
        histogram += h
    
    return histogram

def w3_bootstrap_errors(ra, dec, r_ra, r_dec, bins, trails):
    bootstrap_w3 = np.zeros( (trails, bins.size-1) )
    rr_histogram = angular_auto_correlation_function_binned(r_ra, r_dec, d_bins)
    for i in range(trails):
        print("Trail ", i)
        bootstrap_indices = np.random.choice(ra.size, size=ra.size, replace=True)
        b_ra = ra[bootstrap_indices]
        b_dec = dec[bootstrap_indices]
        dd_histogram = angular_auto_correlation_function_binned(b_ra, b_dec, d_bins)
        dr_histogram = angular_cross_correlation_function_binned(b_ra, b_dec, r_ra, r_dec, d_bins)
    
        bootstrap_w3[i] = R*(R-1)/(N*(N-1))*dd_histogram/rr_histogram - (R-1)/N*dr_histogram/rr_histogram + 1

    return np.std(bootstrap_w3, axis=0)
    
ra, dec, flux = np.loadtxt("hw11_2013.dat", unpack=True)

N = ra.size
R = N
m = 10

r_ra = 170 + 20*np.random.rand(m, R)
r_dec = -10 + 20*np.random.rand(m, R)

d_bins = np.logspace(-2,1, 30)
d_bin_sizes = d_bins[1:] - d_bins[0:-1]
d_bin_coords = (d_bins[1:] + d_bins[0:-1])/2 

dd_histogram = angular_auto_correlation_function_binned(ra, dec, d_bins)
rr_histograms = np.zeros((m, d_bins.size-1) )
dr_histograms = np.zeros((m, d_bins.size-1) )

for i in range(m):
    rr_histograms[i] = angular_auto_correlation_function_binned(r_ra[i], r_dec[i], d_bins)
    dr_histograms[i] = angular_cross_correlation_function_binned(ra, dec, r_ra[i], r_dec[i], d_bins)

rr_histogram_a = np.mean(rr_histograms, axis=0)
dr_histogram_a = np.mean(dr_histograms, axis=0)

w3 = R*(R-1)/(N*(N-1))*dd_histogram/rr_histogram_a - (R-1)/N*dr_histogram_a/rr_histogram_a + 1
w31 = R*(R-1)/(N*(N-1))*dd_histogram/rr_histograms[1] - (R-1)/N*dr_histograms[1]/rr_histograms[0] + 1

#w3_errors = w3_bootstrap_errors(ra, dec, r_ra, r_dec, d_bins, 100)
#np.savetxt("w3_errors.dat", w3_errors)

print(np.sum(dd_histogram), ra.size*(ra.size-1)/2)
w3_errors = np.loadtxt("/home/tilman/Documents/ASTR509/hw11/w3_errors.dat")

mask = w3>0
fit1 = scipy.stats.linregress(np.log10(d_bin_coords[mask][2:9]), np.log10(w3[mask][2:9]))
fit2 = scipy.stats.linregress(np.log10(d_bin_coords[mask][9:]), np.log10(w3[mask][9:]))

fit1_graph = d_bin_coords**(fit1[0])* 10**(fit1[1])
fit2_graph = d_bin_coords**(fit2[0])* 10**(fit2[1])

fig, (data_field_plot, random_field_plot) = plt.subplots(1,2)
fig.subplots_adjust(wspace=0.4)
data_field_plot.scatter(ra, dec, s=1, linewidth=0)
data_field_plot.set_xlim(170, 190)
data_field_plot.set_ylim(-10, 10)
data_field_plot.set_aspect("equal")
data_field_plot.set_xlabel("RA [deg]")
data_field_plot.set_ylabel("DE [deg]")
data_field_plot.set_title("Data")
random_field_plot.scatter(r_ra[0], r_dec[0], s=1, linewidth=0)
random_field_plot.set_xlim(170, 190)
random_field_plot.set_ylim(-10, 10)
random_field_plot.set_aspect("equal")
random_field_plot.set_xlabel("RA [deg]")
random_field_plot.set_ylabel("DE [deg]")
random_field_plot.set_title("Random field")
fig.show()
fig.savefig("tt_hw11_fields.png")

#fig1, (w_plot, cor_f_plot) = plt.subplots(2,1)
fig1, w_plot = plt.subplots(1,1)
w_plot.errorbar(x=d_bin_coords, y=w3, yerr=w3_errors, linestyle="None", marker='.', label="bootstrap errors")
w_plot.plot(d_bin_coords, fit1_graph, label="Slope %.2f" % (fit1[0]) )
w_plot.plot(d_bin_coords, fit2_graph, label="Slope %.2f" % (fit2[0]) )
w_plot.set_xscale("log")
w_plot.set_yscale("log", nonposy='clip')
w_plot.set_ylim(0.0001, 10)
#w_plot.set_aspect("equal")
w_plot.set_xlabel(r"$\theta$ [deg]")
w_plot.set_ylabel(r"$w(\theta)$")
w_plot.legend(fontsize="small")


fig1.show()
fig1.savefig("tt_hw11_ang_cor.png")

fig1, w_plot = plt.subplots(1,1)
w_plot.errorbar(x=d_bin_coords, y=w3, yerr=w3/np.sqrt(dd_histogram), linestyle="None", marker='.', label=r"$1/\sqrt{N}$ errors")
w_plot.plot(d_bin_coords, fit1_graph, label="Slope %.2f" % (fit1[0]) )
w_plot.plot(d_bin_coords, fit2_graph, label="Slope %.2f" % (fit2[0]) )
w_plot.set_xscale("log")
w_plot.set_yscale("log", nonposy='clip')
w_plot.set_ylim(0.0001, 10)
w_plot.set_xlabel(r"$\theta$ [deg]")
w_plot.set_ylabel(r"$w(\theta)$")
w_plot.legend(fontsize="small")
fig1.show()
fig1.savefig("tt_hw11_ang_cor2.png")

fig1, w_plot = plt.subplots(1,1)
w_plot.errorbar(x=d_bin_coords, y=w31, yerr=w31/np.sqrt(dd_histogram), linestyle="None", marker='.', label=r"$1/\sqrt{N}$ errors")

w_plot.set_xscale("log")
w_plot.set_yscale("log", nonposy='clip')
w_plot.set_ylim(0.0001, 10)
w_plot.set_xlabel(r"$\theta$ [deg]")
w_plot.set_ylabel(r"$w(\theta)$")
w_plot.legend(fontsize="small")
fig1.show()
fig1.savefig("tt_hw11_ang_cor3.png")

fig1, cor_f_plot = plt.subplots(1,1)
cor_f_plot.loglog(d_bin_coords, dd_histogram, linewidth=0, marker='.', label="DD")
cor_f_plot.loglog(d_bin_coords, rr_histogram_a, linewidth=0, marker='.', label="RR")
cor_f_plot.loglog(d_bin_coords, dr_histogram_a, linewidth=0, marker='.', label="DR")
cor_f_plot.legend(loc="upper left", fontsize="small")
cor_f_plot.set_xlabel(r"$\theta$ [deg]")
cor_f_plot.set_ylabel("N")
fig1.show()
fig1.savefig("tt_hw11_ang_cor4.png")