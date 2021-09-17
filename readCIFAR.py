import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
# dict=None
# with open('data_batch_2', 'rb') as fo:
#     dict = pickle.load(fo, encoding='latin1')
# print(dict['data'][0]) 
# rgb_weights = [0.2989, 0.5870, 0.1140]
# X = np.reshape(dict['data'], (10000,3, 32, 32))
# X = X.transpose(0, 2, 3, 1)
b=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
a=np.zeros((3,4))
c=np.array([9,8,7,6,5,4])
print('a shape', a.shape)
print('b shape', b.shape)
print('c shape', c.shape)
# for i in range(0,5000):
# 	sm=X[i]
# 	# print(sm.shape)
# 	grayscale_image = np.dot(sm[...,:3], rgb_weights)
# 	GRI[i]=np.reshape(grayscale_image, (1024))
# print(GRI[3].shape)
# print(len(X[0][0]))
# print row 1
print(b[1,:])
# print col 1
print(b[:,1])
# print from index 2 to 5
print(c[2:6])
print(os.path.abspath('face-data\\face.mat'))
# print('------------------------------------------------')
# print(X[0][0])
# print(X[0][1])
# print(X[0][2])
# print(len(X[0][0]))
# ,cmap='gray'
# plt.title("main image")
# plt.imshow(np.reshape(GRI[2],(32,32)),cmap='gray')
# plt.show()
# plt.imshow(X[0][1])
# plt.show()
# plt.imshow(X[0][2])
# plt.show()