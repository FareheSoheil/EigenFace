import scipy.io
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from PIL import Image
import scipy.stats
from scipy.stats import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import warnings
import statsmodels.api as sm 
import seaborn as sns
import pylab as py 
# load non-face data
def readCifar(file):
	dict=None
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='latin1')
	rgb_weights = [0.2989, 0.5870, 0.1140]
	X = np.reshape(dict['data'], (10000,3, 32, 32))
	X = X.transpose(0, 2, 3, 1)
	GRI=np.zeros((5000,1024))
	for i in range(0,5000):
		sm=X[i]
		# print(sm.shape)
		grayscale_image = np.dot(sm[...,:3], rgb_weights)
		GRI[i]=np.reshape(grayscale_image, (1024))
	return GRI
# load face data
def readSamples(file):
	mat = scipy.io.loadmat(file)
	indicies=np.random.randint(1,5000,100)
	X=mat['X']
	# showing some samples
	samples =  np.zeros(shape=(100,1024))
	for i in range(0,100):
		samples[i]=X[indicies[i]]

	sample_data = np.zeros(shape=(320,320))
	for i in range(0,10):
		for j in range(0,10):
			sample_data[i*32:(i+1)*32,j*32:(j+1)*32]= np.reshape(samples[i*10+j,:],(32,32))
	# plotImg("100 face samples",sample_data,'gray',0)

	return X
	# end of showing some samples
# compute eigen faces
def eigenFace(X):
	# input face samples
	# output eigen values and vectors
	# computing covariance matrix
	mean_x = np.mean(X, axis=0)
	centered_x = X-mean_x
	cov_matrix = np.zeros(shape=(len(centered_x[0]),len(centered_x[0])))
	for i in range(0, len(centered_x)):
		a=centered_x[i]
		dot=np.dot(a[:,None],a[None,:])
		cov_matrix = cov_matrix + dot

	cov_matrix_len=len(cov_matrix)
	for i in range(0, cov_matrix_len):
		for j in range(0, len(cov_matrix[0])):
			cov_matrix[i][j]=cov_matrix[i][j]/cov_matrix_len

	# get eigenvalues and eigenvectors of cov-matrix
	[VALS,VECS] = LA.eig(cov_matrix);

	# sort eigenvalues while preserving the correspondance between eigenvalues and eigenvectors
	sorted_values_indices = np.argsort(VALS)
	sorted_VALS=VALS[::-1].sort()
	ordered_VECS = np.zeros(shape=(len(VECS),len(VECS)))
	for i in range(0,len(sorted_values_indices)):
		ordered_VECS[:,len(sorted_values_indices)-i-1]=VECS[:,sorted_values_indices[i]]

	# show eigen faces
	l=10
	eigf_sample = VECS[:,1:l*l+1];
	eigf = np.zeros(shape=(320,320))
	for i in range(0,l):
		for j in range(0,l):
			eigf[i*32:(i+1)*32,j*32:(j+1)*32]= np.reshape(eigf_sample[:,i*10+j],(32,32))

	# plotImg("eigf_sample",eigf,'gray',0)
	

	# histogram for eigenvalues
	eig_vals_hist = VALS[:]/sum(VALS)
	# plotBars("magnitude of each eigen value in %",eig_vals_hist,'pink',70)
	return(VALS,VECS)

def plotImg(title,img,cmap,reshaped):
	# plots and saves image
	plt.title(title)
	if reshaped==1:
		plt.imshow(np.reshape(img,(32,32)),cmap=cmap)
	else:
		plt.imshow(img,cmap=cmap)
	plt.savefig('_'+title+'.png')	
	plt.show()
	block=False
	plt.pause(1)
	plt.close()

def plotBars(title,arr,color,xlim):
	# plots and saves bars
	y_posW = np.arange(len(arr))
	plt.bar(y_posW,arr,width = 0.4,color=color)
	plt.title(title)
	plt.xlim(0,xlim)
	plt.savefig('_'+title+'.png')
	plt.show(block=False)
	plt.pause(1)
	plt.close()


def eigenVectorWeightCalculator(img,VECS):
	# computs weight of eigen vector constructing image
	weights = np.dot(img,VECS);
	weightT= np.array([weights]).T
	recons_img = np.dot(VECS,weightT);
	return weights,recons_img

def eigenVectorWeightSeperator(X,VECS,numOfev,numOfSamples):
	r=numOfev
	c=numOfSamples
	ev_weights=np.zeros((r,c))
	for i in range(0,c):
		weights,recons_img=eigenVectorWeightCalculator(X[i,:],VECS)
		ev_weights[:,i]=weights[0:numOfev]
	return ev_weights

def statisticalCalculator(ev_weights,r):
	deviations=np.zeros(r)
	avgs=np.zeros(r)
	for i in range(0,r):
		deviations[i]=np.std(ev_weights[i,:])
		avgs[i]=sum(ev_weights[i,:])/len(ev_weights[i,:])
	return deviations,avgs
# def subPlotBars(xlabel,arr,color,xlim,row,col,index)
def imageProcess(X,VECS,type,numOfSamples):
	random_indicies=np.random.randint(1,5000,numOfSamples)
	ev_weights=np.zeros((10,numOfSamples))
	for i in range(0,numOfSamples):
		# decompose an image to eigenfaces
		# num_of_eigenfaces = 30;
		# img shape = (1024,)
		img = X[random_indicies[i],:]
		# weights shape = (1024,)
		weights = np.dot(img,VECS)
		weightT= np.array([weights]).T
		recons_img = np.dot(VECS,weightT)
		weights,recons_img=eigenVectorWeightCalculator(X[random_indicies[i],:],VECS)
		# histogram for weights
		ev_weights[:,i]=weights[0:10]
		# just for my checking
		plotBars(str(type)+'_'+str(i)+'_first 10 eigen vectors',weights,'red',10)
	for i in range(0,10):
		plotBars(str(numOfSamples)+'_'+str(type)+' weights of the '+str(i)+'th eigen vector',ev_weights[i,:],'red',numOfSamples)

# standarise best distribution
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

# find best distribution
def fit_distribution(array,pct,pct_lower):
	chi_square_statistics = []
	y_std,size,y_org = standarise(array,pct,pct_lower)
	percentile_bins = np.linspace(0,100,11)
	percentile_cutoffs = np.percentile(y_std, percentile_bins)
	observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
	cum_observed_frequency = np.cumsum(observed_frequency)
	dist_names = ['weibull_min','norm','weibull_max','beta','invgauss','uniform','gamma','expon','lognorm','pearson3','triang']
	# Loop through candidate distributions
	# minimum=1000000
	# minimumParam=
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
	# print (results[0]['Distribution'])
	# return(results[0]['Distribution'])

def egienValueElicitor():
	pass

# non-faces
face_files='face-data\\faces.mat'
non_face_files='cifar-data\\data_batch_1'
faces=readSamples(face_files)
non_faces=readCifar(non_face_files)
values,vectors=eigenFace(faces)
numOfsamples=5000
numpOfev=1024
avgFace=stdFace=avgNon=stdNon=np.zeros(numpOfev)
# calculating weights
allFaceWeights=eigenVectorWeightSeperator(faces,vectors,numpOfev,numOfsamples)
allNonFaceWeights=eigenVectorWeightSeperator(non_faces,vectors,numpOfev,numOfsamples)
allFaceWeightsAppended=allNonFaceWeightsAppended=np.zeros(1)
for i in range(0,numpOfev):
	allFaceWeightsAppended=np.append(allFaceWeightsAppended,allFaceWeights[i,:])
	allNonFaceWeightsAppended=np.append(allNonFaceWeightsAppended,allNonFaceWeights[i,:])

print(allFaceWeightsAppended.shape)
print(allNonFaceWeightsAppended.shape)
fig, axes = plt.subplots(figsize=(12, 7))
plt.title('on '+str(numOfsamples)+'samples')
axes.hist(allFaceWeightsAppended,alpha=0.5,bins=30, color='pink', label='face data')
axes.hist(allNonFaceWeightsAppended,alpha=0.5,bins=30, color='blue', label='non-face data')
axes.legend(loc='upper right')
fig.tight_layout()
plt.show()


# numpOfev=16
# faceWeights=eigenVectorWeightSeperator(faces,vectors,numpOfev,numOfsamples)
# nonFaceWeights=eigenVectorWeightSeperator(non_faces,vectors,numpOfev,numOfsamples)

# # avg and std of face and non-face data
# stdFace,avgFace=statisticalCalculator(faceWeights,numpOfev)
# stdNon,avgNon=statisticalCalculator(nonFaceWeights,numpOfev)
# print('--------------------------------')
# print('----------- Averages of ev wieghts -----------')
# for i in range(0,numpOfev):
# 	print('EV '+str(i)+'-> Face: '+str(avgFace[i])+'     Non Face: '+str(avgNon[i]))

# print('----------- Standard deviations of ev wieghts -----------')
# for i in range(0,numpOfev):
# 	print('EV '+str(i)+'-> Face: '+str(stdFace[i])+'     Non Face: '+str(stdNon[i]))
# # print('Face avgs',avgFace)
# # print('Face stds',stdFace)
# # print('Non Face avgs',avgNon)
# # print('Non Face stds',stdNon)
# for i in range(0,2):
# 	# y_std,len_y,y = standarise(faceWeights[i,:],1,0)
# 	fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 5))
# 	plt.title('on '+str(numOfsamples)+'samples')
# 	for j in range(0,4):
# 		for k in range(0,2):
# 			y1=faceWeights[(8*i+2*j+k),:]
# 			y2=nonFaceWeights[(8*i+2*j+k),:]
# 			axes[j][k].hist(y1,alpha=0.5,bins=20, color='green', label='face data')
# 			axes[j][k].hist(y2,alpha=0.5,bins=20, color='orange', label='non-face data')
# 			# axes[j][k].set_xlabel(str(i)+' ev weights for faces\n\nHistogram plot of Oberseved Data')
# 			axes[j][k].set_ylabel('Freq: '+str(8*i+2*j+k+1))
			
# 			# x_std,len_x,x = standarise(nonFaceWeights[i,:],1,0)
# 			# axes[j][k].hist(y2, color='orange')
# 			# axes[j][k].set_xlabel(str(i)+' ev weights for non faces\n\nHistogram plot of Oberseved Data')
# 			# axes[j][k].set_ylabel('Frequency')
# 			axes[j][k].legend(loc='upper right')
# 	fig.tight_layout()
# 	plt.show()

