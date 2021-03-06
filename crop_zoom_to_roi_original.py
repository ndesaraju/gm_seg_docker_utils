#!/usr/bin/env python
from matplotlib.path import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import interpolate
import scipy.ndimage.interpolation as wezoom
import glob
import os
import sys
import argparse
import math
def get_dimension(msid):
    import pandas as pd
    from subprocess import Popen,PIPE
    cmd = ["fslinfo",msid]
    proc = Popen(cmd, stdout=PIPE)
    lines = [l.decode("utf-8").split() for l in proc.stdout.readlines()[5:]]
    d={}
    for i in lines:
        d[i[0]]=i[1]
    dim1=d['pixdim1']
    dim2=d['pixdim2']
    #tmp["a0"] = par_returner(x,ch[0][0],ch[0][1], ch[0][2], ch[0][3], ch[0][4], ch[0][5])[1]
    return float(dim1),float(dim2)
def create_zoomed_files(psir, roi,slic=-1):
    psir_affine, psir_data = load(psir)
    x_res = psir_affine[0,0]
    y_res = psir_affine[1,1]
    x_res,y_res=get_dimension(psir) #get dimensions of current dimensions
    #print(psir_data.shape)
    if(len(psir_data.shape) == 2):
        psir_data = psir_data[:,:,np.newaxis]
    x_dim, y_dim, z_dim = psir_data.shape

    # COMPUTE SCALE FACTOR AND CROP DISTANCE
    
    scalefac = int(round(math.sqrt((abs(x_res*y_res)/(.078125**2))))) #.078125 is the new dimensions that I'm aiming for
    cropdist = int(round(150/float(scalefac)))
    
    # settings
    numvox = cropdist*2*scalefac
    bsp_order=2

    pt_list = getpts(roi)
    #print(pt_list[2])
    # FLIP SIGN OF Y
    flip_ysign_matrix=np.ones([pt_list.shape[0],pt_list.shape[1]])
    flip_ysign_matrix[:,1]=-1
    flip_ysign_pt_list = pt_list*flip_ysign_matrix
    
    # TRANSFORM FROM JIM COORDINATES (0,0 at center of PSIR image) TO PSIR SPACE
    x_multiplicative_factor = 1/(math.fabs(x_res))
    y_multiplicative_factor = 1/(math.fabs(y_res))
    x_additive_factor = x_dim/2
    y_additive_factor = y_dim/2
    psir_pt_list = flip_ysign_pt_list*(x_multiplicative_factor, y_multiplicative_factor)+(x_additive_factor,y_additive_factor)
    # FIND CENTER OF MASS OF SPINAL CORD WM, THEN SET BORDERS
    center_of_mass_psir_space = np.floor(psir_pt_list.mean(0))

    xmin_psir_space = int(center_of_mass_psir_space[0]-cropdist)
    xmax_psir_space = int(center_of_mass_psir_space[0]+cropdist)
    ymin_psir_space = int(center_of_mass_psir_space[1]-cropdist)
    ymax_psir_space = int(center_of_mass_psir_space[1]+cropdist)

    xmin_JIM_space = (xmin_psir_space-x_additive_factor)/x_multiplicative_factor
    xmax_JIM_space = (xmax_psir_space-x_additive_factor)/x_multiplicative_factor
    ymin_JIM_space = (ymin_psir_space-y_additive_factor)/y_multiplicative_factor
    ymax_JIM_space = (ymax_psir_space-y_additive_factor)/y_multiplicative_factor
    #print 'xmin = %s; xmax = %s; ymin = %s; ymax = %s' % (xmin, xmax, ymin, ymax)

    # MAKE PSIR CROP IMAGE
    psir_crop = psir_data[xmin_psir_space:xmax_psir_space, ymin_psir_space:ymax_psir_space]
    crop_aff = np.diag([x_res*psir_crop.shape[0]/numvox,y_res*psir_crop.shape[1]/numvox,1,1])
    if slic!=-1:
        z = psir_crop[:,:,slic]
    else:
        z = psir_crop[:,:]
    
    
    
    # MAKE PSIR CROP INTERP IMAGE
    zoomed = wezoom.zoom(z, numvox/(xmax_psir_space-xmin_psir_space), order=bsp_order)
    zoomed_file = psir[:-7]+'_zoomed.nii.gz'
    #nib.save(nib.Nifti1Image(zoomed,crop_aff), zoomed_file)

    # MAKE CORD MASK IMAGE
    path = Path(flip_ysign_pt_list)
    X = np.linspace(xmin_JIM_space,xmax_JIM_space,zoomed.shape[0])
    Y = np.linspace(ymin_JIM_space,ymax_JIM_space,zoomed.shape[0])
    cord_mask = np.zeros([len(X),len(Y)])
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            cord_mask[i,j]=path.contains_point([x,y])
    cord_mask_file = psir[:-7]+'_zoomed_cord_mask.nii.gz'
    #cord_mask_img = nib.save(nib.Nifti1Image(cord_mask.astype("uint8"), crop_aff), cord_mask_file)

    # MAKE CORD IMAGE
    #print('cord_mask:{} zoomed: {}'.format(cord_mask.shape,zoomed.shape))
    cord = np.multiply(cord_mask, zoomed[:,:,0])
    
    cordsave=nib.save(nib.Nifti1Image(cord,crop_aff),os.path.dirname(psir)+'/only_cord'+os.path.basename(psir))
    cordpth=os.path.dirname(psir)+'/only_cord'+os.path.basename(psir)
    return cord,crop_aff,cordpth

def getpts(filepath):
    rois = 0
    cordlist=[]
    cordxylist = []
    roifile = open(filepath, 'r')
    lines = roifile.readlines()
    roifile.close()
    for line in lines:
        if "X=" in line:
            cordlist.append(line.strip())
    for pt in cordlist:
        parts = pt.split('=')
        x = parts[1][:-3]
        y = parts[-1]
        cordxylist += [[x, y]]
        cord = np.array(cordxylist, dtype='float64')
    return cord

def load(image):
    img = nib.load(image)
    img_affine = img.get_affine()
    img_data = np.array(img.get_data())
    return img_affine, img_data

