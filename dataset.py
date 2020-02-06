from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
import tensorflow as tf
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import time

class Dataset(object):
	"""
	images_train = np.array([])
	images_test = np.array([])
	labels_train = np.array([])
	labels_test = np.array([])
	unique_train_label = np.array([])
	map_train_label_indices = dict()
	"""
	def _get_siamese_similar_pair(self):

		label =np.random.choice(self.unique_train_label)
		l, r = np.random.choice(self.map_train_label_indices[label], 2, replace=False)
		return l, r, 1

	def _get_siamese_dissimilar_pair(self):

		label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
		l = np.random.choice(self.map_train_label_indices[label_l])
		r = np.random.choice(self.map_train_label_indices[label_r])
		return l, r, 0

	def _get_siamese_pair(self):
		z = np.random.uniform()
		if  z < 0.5:
			#print("zzzzzz: ", z)
			
			l,r, x = self._get_siamese_similar_pair()
			#print( l, r)
			#print(self.images_train[l,:].shape)
			return l,r, x#self.images_train[l,:], self.images_train[r, :], x, l,r
		else:
			#print("eeeeee: ", z)
			
			l,r, x = self._get_siamese_dissimilar_pair()
			#print( l, r)
			return l,r, x#self.images_train[l,:], self.images_train[r, :], x, l, r
	
	def _get_labels(self, train_size,enhance):
		i = 0
		while i<train_size:
			z = np.random.uniform()
			if  z < 0.1219:
				l,r, x, label_l, label_r = self._get_siamese_similar_pair()
				yield l,r, x, label_l, label_r
			else:
				l,r, x, label_l, label_r = self._get_siamese_dissimilar_pair()
				yield l,r, x, label_l, label_r
			i+=1



	def _get_train_pair_generator(self, train_size, enhance = True):
		
		i  = 0
		while i < train_size:
			z = np.random.uniform()
			if  z < 0.05: #0.1219: #1086


				l,r, x = self._get_siamese_similar_pair()
				left_im, right_im = self.images_train[l,:], self.images_train[r, :]
				if enhance: 
					left_im, right_im = self._image_resize_filter(left_im), self._image_resize_filter(right_im)
				
				yield  left_im, right_im,x #, label_l, label_r 
			else:
				l,r, x = self._get_siamese_dissimilar_pair()
				left_im, right_im = self.images_train[l,:], self.images_train[r, :]
				if enhance:
					left_im, right_im = self._image_resize_filter(left_im), self._image_resize_filter(right_im)
				yield left_im, right_im, x#, label_l, label_r #
			i+=1
	def _get_siamese_batch(self, n, enhance = True):
		idxs_left, idxs_right, labels = [], [], []
		for _ in range(n):
			l, r, x = self._get_train_pair_generator(enhance)
			idxs_left.append(l)
			idxs_right.append(r)
			labels.append(x)
		return idxs_left, idxs_right, labels#self.images_train[idxs_left,:], self.images_train[idxs_right, :], np.expand_dims(labels, axis=1)
	def _test_dataset_generator(self, test_size, enhance, k_hot):
		i = 0
		while i<test_size:
			yield self._get_test_batch(k_hot, enhance)
			i+=1
	def _get_test_batch(self, n, enhance):
		idxs_left, idxs_right = [], []
		labels = np.random.choice(self.unique_test_label, n, replace = False)
		l, r = np.random.choice(self.map_test_label_indices[labels[0]], 2, replace=False)
		left_im, right_im = self.images_test[l,:], self.images_test[r, :]

		if enhance:
			left_im, right_im = self._image_resize_filter(left_im), self._image_resize_filter(right_im)

		idxs_left.append(left_im)
		idxs_right.append(right_im)

		for index in range(1,n):
			[r] = np.random.choice(self.map_test_label_indices[labels[index]], 1, replace=False)
			right_im = self.images_test[r, :]
			if enhance:
				right_im = self._image_resize_filter(right_im)
			idxs_left.append(left_im)
			idxs_right.append(right_im)
		
		return idxs_left, idxs_right #self.images_test[idxs_left,:], self.images_test[idxs_right, :]
	def _image_resize_filter(self, im,  height=128, width=128):
		im = Image.fromarray((im*255/np.max(im)).astype(np.uint8))
		im = im.resize((128, 128), resample = 3)
		im = ImageEnhance.Sharpness(im).enhance(3)
		im = ImageEnhance.Contrast(im).enhance(1.5)
		enhanced  = np.array(im)/255	
		im.close()
		return enhanced

class Image_Dataset(Dataset):
	def __init__(self):
		print("===Loading cifar100 Dataset===")

		(self.images_train, self.labels_train), (self.images_test, self.labels_test) = cifar100.load_data()
		#self.images_train = np.expand_dims(self.images_train, axis=3) / 255.0
		#self.images_test = np.expand_dims(self.images_test, axis=3) / 255.0
		self.images_train,self.images_test =self.images_train/255.0, self.images_test/255.0
		self.labels_train = np.expand_dims(self.labels_train, axis=1)

		self.unique_train_label = np.unique(self.labels_train)
		self.unique_test_label = np.unique(self.labels_test)

		self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in self.unique_train_label}
		self.map_test_label_indices = {label: np.flatnonzero(self.labels_test == label) for label in self.unique_test_label}

		self.n_classes,self.n_examples,self.w,self.h = self.images_train.shape
		self.n_val,self.n_ex_val,_,_ = self.images_test.shape
		print("Test shape: ", self.images_test.shape)

		
		
		print("Images train :", self.images_train.shape)
		print("Labels train :", self.labels_train.shape)
		print("Images test  :", self.images_test.shape)
		print("Labels test  :", self.labels_test.shape)
		print("Unique label :", self.unique_train_label)
# print("Map label indices:", self.map_train_label_indices)

if __name__ == "__main__":
	# Test if it can load the dataset properly or not. use the train.py to run the training
	a = Image_Dataset()
	batch_size = 4
	ls, rs, xs = a.get_siamese_batch(batch_size)
	f, axarr = plt.subplots(batch_size, 2)
	for idx, (l, r, x) in enumerate(zip(ls, rs, xs)):
		print("Row", idx, "Label:", "similar" if x else "dissimilar")
		print("max:", np.squeeze(l, axis=2).max())
		axarr[idx, 0].imshow(np.squeeze(l, axis=2))
		axarr[idx, 1].imshow(np.squeeze(r, axis=2))
		plt.show()

# SIZE OPENPOSE = 368 X 368
# IMAGE =112 X 112

