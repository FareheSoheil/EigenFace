
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import warnings
import statsmodels.api as sm 
import seaborn as sns
import pylab as py 
# ()
def standarise(array,pct,pct_lower):
    sc = StandardScaler() 
    y = array
    y.sort()
    len_y = len(y)
    y = y[int(pct_lower * len_y):int(len_y * pct)]
    len_y = len(y)
    yy=([[x] for x in y])
    sc.fit(yy)
    y_std =sc.transform(yy)
    y_std = y_std.flatten()
    return y_std,len_y,y

# ()
def fit_distribution(array,pct,pct_lower):
	chi_square_statistics = []
	y_std,size,y_org = standarise(array,pct,pct_lower)
	percentile_bins = np.linspace(0,100,11)
	percentile_cutoffs = np.percentile(y_std, percentile_bins)
	observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
	cum_observed_frequency = np.cumsum(observed_frequency)
	dist_names = ['weibull_min','norm','weibull_max','beta','invgauss','uniform','gamma','expon','lognorm','pearson3','triang']
	# print(len(y_std))
	# print('percentile_bins ',percentile_bins)
	# print('percentile_cutoffs ',percentile_cutoffs)
	# print('observed_frequency ',observed_frequency)
	# print('bins ',bins)
	# print('cum_observed_frequency ',cum_observed_frequency)

	# Loop through candidate distributions
	for distribution in dist_names:
	    # Set up distribution and get fitted distribution parameters
	    dist = getattr(scipy.stats, distribution)
	    param = dist.fit(y_std)
	    print("{}\n{}\n".format(dist, param))
	        # Get expected counts in percentile bins
	    # cdf of fitted sistrinution across bins
	    cdf_fitted = dist.cdf(percentile_cutoffs, *param)
	    expected_frequency = []
	    for bin in range(len(percentile_bins)-1):
	        expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
	        expected_frequency.append(expected_cdf_area)

	    # Chi-square Statistics
	    expected_frequency = np.array(expected_frequency) * size
	    cum_expected_frequency = np.cumsum(expected_frequency)
	    ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
	    chi_square_statistics.append(ss)
	#Sort by minimum ch-square statistics
	results = pd.DataFrame()
	results['Distribution'] = dist_names
	results['chi_square'] = chi_square_statistics
	results.sort_values(['chi_square'], inplace=True)

	print ('\nDistributions listed by Betterment of fit:')
	print ('............................................')
	print (results)


# ()
arr=[11,12,23,23,23,1000,342,23,24,25,56,57,76,45,124,125,145,765,875,3,4,5,6]
fit_distribution(arr, 1, 0)
y_std,len_y,y = standarise(arr,1,0)
plt.hist(y)
plt.xlabel('num')
plt.ylabel('Frequency')
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
axes[0].hist(y)
axes[0].set_xlabel('Price\n\nHistogram plot of Oberseved Data')
axes[0].set_ylabel('Frequency')
axes[1].plot(y,pearson3.pdf(y_std,3.3145421275666136, 1.9073291431666877e-17, 0.9302955990221242))
# (3.3145421275666136, 1.9073291431666877e-17, 0.9302955990221242)
axes[1].set_xlabel('Price\n\npearson3 Distribution')
axes[1].set_ylabel('pdf')
axes[2].plot(y,weibull_min.pdf(y_std,0.6676819297959757, 2.896119846018192, 1.2543783623999674))
axes[2].set_xlabel('Price\n\nweibull_min Distribution')
axes[2].set_ylabel('pdf')
fig.tight_layout()
plt.show()