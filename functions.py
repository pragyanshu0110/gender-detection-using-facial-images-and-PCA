import numpy as np
from numpy import * 
import os
import cv2

# sklearn
from sklearn.utils import shuffle


#==========================[1]:DATA===============================
def training_data():
	path='training_set'

	imlist=os.listdir(path)
	#print(imlist[0])
	num_samples=np.size(imlist)

	img0=cv2.imread('training_set'+'/'+imlist[0],0)
	print(img0.shape)
	#cv2.imshow('img0',img0)
	#cv2.waitKey(0)

	img1=np.array(img0) 
	#print('img1',img1)
	m,n=img1.shape[0:2]
	no_of_img=len(imlist)
	
	print('m,n,no_img',m,n,no_of_img)

	img_matrix=np.array([np.array(cv2.imread('training_set'+'/'+img2,0)).flatten()
               for img2 in imlist],'F')
	
	print('img_matrix',img_matrix.shape)
	label=np.ones((num_samples,),dtype=int)
	label[0:237]=0
	label[237:474]=1

	data,Label=shuffle(img_matrix,label,random_state=2)
	train_data=[data, Label]
	#print(train_data[0].shape,train_data[1].shape)

	return train_data





#===================== test data
def test_data():
	path='test_set'

	imlist=os.listdir(path)
	#print(imlist[0])
	num_samples=np.size(imlist)

	img0=cv2.imread('test_set'+'/'+imlist[0])
	#print(img0.shape)
	#cv2.imshow('img0',img0)
	#cv2.waitKey(0)

	img1=np.array(img0) 
	#print('img1',img1)
	m,n=img1.shape[0:2]
	no_of_img=len(imlist)
	
	print('m,n,no_img',m,n,no_of_img)

	img_matrix=np.array([np.array(cv2.imread('test_set'+'/'+img2)).flatten()
               for img2 in imlist],'f')
	
	label=np.ones((num_samples,),dtype=int)
	label[0:235]=0
	

	data,Label=shuffle(img_matrix,label,random_state=2)
	test_data=[data, Label]
	print(test_data[0].shape,test_data[1].shape)

	return test_data









#=========================[2]:normalization ==============================

def normalization(x):
	x=np.divide(x,255)
	return x


#========================









