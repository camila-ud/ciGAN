import glob
import numpy as np
import scipy.misc as misc
import scipy.ndimage.morphology as morph
import scipy.ndimage.filters as filters
from scipy import ndimage
import random
from config import *
import h5py
import hickle as hkl
import re
import pdb
import pickle
import os, os.path
import imageio
import skimage.transform as transform
import sys
import warnings
warnings.filterwarnings("ignore")


def normalize(img):
    return (img - img.min())/(img.max()-img.min())


def generate_cpatches(nsamples, patches_path ='./patches.npz'):
    print("g",nsamples)
    combined_dims = 4

    while True:
        counts = nsamples
        #n_samples -> n_experiments, probabity sample_rates
        X_all = None
        #load all images from each class (i)
        X_masks = np.load(patches_path)['x_mask']
        X_reals = np.load(patches_path)['x_real']
        try:
            # get (c) images
            shuffle_idx = np.random.choice(len(X_masks), counts, replace=False)
        except:
            pdb.set_trace()
        
        #get mask and reals
        X_masks = X_masks[shuffle_idx]
        X_reals = X_reals[shuffle_idx]
        
        X_ = []
        for i,_ in enumerate(X_masks):
            X_mask = X_masks[i].reshape((patch_size, patch_size))
            X_real = normalize(X_reals[i].reshape((patch_size, patch_size)))

            X_rand = np.random.uniform(0, 1, X_mask.shape)*X_mask
            X_corrupt = (np.multiply(X_real, np.logical_not(X_mask).astype(int))+X_rand)
            boundary = np.multiply(np.invert(morph.binary_erosion(X_mask)), X_mask)
            X_boundary = normalize(filters.gaussian_filter(255.0*boundary,10)).reshape((patch_size, patch_size, 1))

            X_combined = np.concatenate((X_corrupt.reshape(patch_size, patch_size, 1), 
                                        X_mask.reshape(patch_size, patch_size, 1),
                                        X_real.reshape(patch_size, patch_size, 1),
                                        X_boundary), 
                                        axis=-1)
            X_.append(X_combined)
        X_ = np.stack(X_)
        print("Patches size {}".format(X_.shape))
        shuffle_idx = np.random.choice(len(X_), len(X_), replace=False)
        yield X_[shuffle_idx]

     
    
def generate_patch_id(i, patches_path ='./patches.npz'):
    #generate a image with an id
    X_masks = np.load(patches_path)['x_mask']
    X_reals = np.load(patches_path)['x_real']
    
    X_ = []
    X_mask = X_masks[i].reshape((patch_size, patch_size))
    X_real = normalize(X_reals[i].reshape((patch_size, patch_size)))

    X_rand = np.random.uniform(0, 1, X_mask.shape)*X_mask
    X_corrupt = (np.multiply(X_real, np.logical_not(X_mask).astype(int))+X_rand)
    boundary = np.multiply(np.invert(morph.binary_erosion(X_mask)), X_mask)
    X_boundary = normalize(filters.gaussian_filter(255.0*boundary,10)).reshape((patch_size, patch_size, 1))

    X_combined = np.concatenate((X_corrupt.reshape(patch_size, patch_size, 1), 
                                    X_mask.reshape(patch_size, patch_size, 1),
                                    X_real.reshape(patch_size, patch_size, 1),
                                    X_boundary), 
                                    axis=-1)
    X_.append(X_combined)
    X_ = np.stack(X_)
    print("Patch id {} is generated, {}".format(i,X_.shape))
    return X_
    

def generate_nc_patches(nsamples, patches_path ='./patches.npz',
                        cancer_path = './non_cancer_patches.npz'):
    print(nsamples)
    combined_dims = 4

    while True:
        counts = nsamples
        #n_samples -> n_experiments, probabity sample_rates
        X_all = None
        #load all images from each class (i)
        X_masks = np.load(patches_path)['x_mask']
        X_reals = np.load(cancer_path)['x_real']
        try:
            # get (c) images
            shuffle_idx_mask = np.random.choice(len(X_masks), counts, replace=False)
            shuffle_idx_cancer  = np.random.choice(len(X_reals), counts, replace=False)
        except:
            pdb.set_trace()
        
        #get mask and reals
        X_masks = X_masks[shuffle_idx_mask]
        X_reals = X_reals[shuffle_idx_cancer]
        X_ = []
        for i,_ in enumerate(X_masks):
            X_mask = X_masks[i].reshape((patch_size, patch_size))
            X_real = normalize(X_reals[i].reshape((patch_size, patch_size)))

            X_rand = np.random.uniform(0, 1, X_mask.shape)*X_mask
            X_corrupt = (np.multiply(X_real, np.logical_not(X_mask).astype(int))+X_rand)
            boundary = np.multiply(np.invert(morph.binary_erosion(X_mask)), X_mask)
            X_boundary = normalize(filters.gaussian_filter(255.0*boundary,10)).reshape((patch_size, patch_size, 1))

            X_combined = np.concatenate((X_corrupt.reshape(patch_size, patch_size, 1), 
                                        X_mask.reshape(patch_size, patch_size, 1),
                                        X_real.reshape(patch_size, patch_size, 1),
                                        X_boundary), 
                                        axis=-1)
            X_.append(X_combined)
        X_ = np.stack(X_)
        print("Patches_nc_ size {}".format(X_.shape))
        shuffle_idx = np.random.choice(len(X_), len(X_), replace=False)
        yield X_[shuffle_idx]
