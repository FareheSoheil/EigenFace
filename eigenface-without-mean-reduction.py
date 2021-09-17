import scipy.io
import random
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from copy import copy, deepcopy

# https://www.cs.toronto.edu/~kriz/cifar.html -- more date sets
# loading data
mat = scipy.io.loadmat('faces.mat')
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
plt.title("samples")
imgplot2 = plt.imshow(sample_data,cmap='gray')
plt.show()
# end of showing some samples

# computing covariance matrix
mean_x = np.mean(X, axis=0)
# modified here
# centered_x = X-mean_x
centered_x = X
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

plt.title("eigf_sample")
imgplot3 = plt.imshow(eigf,cmap='gray')
plt.show()

# histogram for eigenvalues
eig_vals_hist = VALS[:]/sum(VALS)
y_pos = np.arange(len(eig_vals_hist))
print('max',max(VALS))
plt.bar(y_pos,eig_vals_hist,width = 0.3,color='pink')
plt.title("Percent of effectivness- Number of the eigen value")
plt.xlim(0,70)
plt.show()

# decompose an image to eigenfaces
num_of_eigenfaces = 50;
img = X[500,:];
weights = np.dot(img,VECS);
weightT= np.array([weights]).T
print(img.shape)
print(VECS.shape)
print(weights.shape)
print(weightT.shape)
recons_img = np.dot(VECS,weightT);
plt.title("main image")
plt.imshow(np.reshape(img,(32,32)),cmap='gray')
plt.show()
plt.title("reconstructed image with all comps")
plt.imshow(np.reshape(recons_img,(32,32)),cmap='gray')
plt.show()
# weights2 = img*v_d(:,1:numofcomp);
# recons_img2 = v_d(:,1:numofcomp)*weights2';
# figure;subplot(131);imshow(reshape(img,32,32),[]);title('main image')
# subplot(132);imshow(reshape(recons_img,32,32),[]);title('reconstructed image with all comps')
# subplot(133);imshow(reshape(recons_img2,32,32),[]);title('reconstructed image with 20 comps')

# print('VECS[0]', VECS[0])
# print('VALS[0,1,2,3]', VALS[0], VALS[1], VALS[2], VALS[3])
# print('VAL[0,1,2,3] sorted', VALS[0], VALS[1], VALS[2], VALS[3])
# print('len(VEC)', len(VECS))
# print('len(VEC[0])', len(VECS[0]))
# print(len(VALS))
