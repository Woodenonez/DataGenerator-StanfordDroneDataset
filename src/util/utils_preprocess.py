import cv2
import numpy as np

from typing import Union, List

def rot(image:np.ndarray, label:np.ndarray, k:int=1):
	'''
	Rotates image and coordinates counter-clockwise by k * 90° within image origin
	:param image: HxWxC or HxW
	:param label: [[x,y],[x,y],...]
	:param k: Number of times to rotate by 90°
	:return: Rotated Dataframe and image
	'''
	xy = label.copy()

	if image.ndim == 3:
		h0, w0, _ = image.shape
	else:
		h0, w0= image.shape
	xy[:,0] = xy[:,0] - w0 / 2
	xy[:,1] = xy[:,1] - h0 / 2

	R = np.array([[ np.cos(-k*np.pi/2), np.sin(-k*np.pi/2)], 
	              [-np.sin(-k*np.pi/2), np.cos(-k*np.pi/2)]])
	xy = np.dot(xy, R)
	for _ in range(k):
		image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

	if image.ndim == 3:
		h0, w0, _ = image.shape
	else:
		h0, w0 = image.shape
	xy[:,0] = xy[:,0] + w0 / 2
	xy[:,1] = xy[:,1] + h0 / 2
	return image, xy

def fliplr(image:np.ndarray, label:np.ndarray, k=0):
	'''
	Flip image and coordinates horizontally
	:param image: HxWxC or HxW
	:param label: [[x,y],[x,y],...]
	:return: Flipped Dataframe and image
	'''
	xy = label.copy()
	if image.ndim == 3:
		h0, w0, _ = image.shape
	else:
		h0, w0= image.shape
	xy[:,0] = xy[:,0] - w0 / 2
	xy[:,1] = xy[:,1] - h0 / 2

	R = np.array([[-1, 0], [0, 1]])
	xy = np.dot(xy, R)
	image = cv2.flip(image, 1) # 0-vertical, 1-horizontal

	if image.ndim == 3:
		h0, w0, _ = image.shape
	else:
		h0, w0= image.shape
	xy[:,0] = xy[:,0] + w0 / 2
	xy[:,1] = xy[:,1] + h0 / 2
	return image, xy

def rot_n_fliplr(image:np.ndarray, label:np.ndarray, k:int=1):
	if k>0:
		image, label = rot(image, label, k)
	image, xy = fliplr(image, label)
	return image, xy