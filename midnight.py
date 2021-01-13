
from shutil import copyfile
from scipy.stats import norm
from avgim import avgim
import nibabel as nb
import numpy as np
import os
import sys
from glob import glob
import crop_zoom_to_roi_original as crop
import commonly as c
import General_reg_longi as GRL
from subprocess import Popen, PIPE
import xlrd
import time
from multiprocessing import Pool
from collections import defaultdict
import pandas as pd
#import cord_seg_clean as cord
import random
#used to individually correct from the gray matter algorithm most recent June 2019
def alter_ims(ima,mask,naming):
        print(ima)
        print(mask)
        tmp=nb.load(mask)
        roi=tmp.get_data()
        tmp1=np.where(roi>.45,roi,0)
        b_roi=np.where(tmp1<=.45,tmp1,1)
        raw=nb.load(ima)
        im=raw.get_data()
        gray=np.multiply(b_roi,im)
        nb.save(nb.Nifti1Image(gray,raw.affine),naming+'gray.nii.gz')
        #info=np.concatenate([gray[np.where(gray>0.1)],gray[np.where(gray<0.1)]])
        #avg=sum(info)/len(info)
        info=gray[np.where(gray>0.1)]
        avg=np.mean(info)
        final=np.where(b_roi<0.1,im,avg-35)
        nb.save(nb.Nifti1Image(final,raw.affine),naming+'altered.nii.gz')
def individual_correct(i,cord,target,naming,subject):
    working='/data/henry4/jjuwono/corrected/'
    mse=subject
    #cord_image=glob('/data/henry11/PBR/subjects/'+mse+'/nii/*only_cord*C2_3*psir*PSIR*')+glob('/data/henry7/PBR/subjects/'+mse+'/nii/*only_cord*C2_3*psir*PSIR*')
    #cord_image=glob('/data/henry6/PBR/H_cohort/*only_cord*'+mse+'*C2_3.nii.gz')
    
    try:
         alter_ims(cord,i,working+mse)
    except:
        errors.append(i)
        input('I errored 1')
        raise ValueError('could not alter')
        return ' '
    pth=[working+mse+'altered.nii.gz']
    try:
        os.mkdir('/data/henry10-w/jjuwono/'+naming)

    except:
        pass
    try:
        os.mkdir('/data/henry10-w/jjuwono/'+naming+'/'+mse)
    except:
        pass
    try:
        os.mkdir('/data/henry10-w/jjuwono/'+naming+'/'+mse+'/warped')
    except:
        pass
    
    output_path='/data/henry10-w/jjuwono/'+naming+'/'+mse+'/'
    static_path=target
    filt=None
    pth2=[i]
    print(pth)
    print(pth2)
    filt2=None
    return GRL.SimpleRegister(pth,output_path,static_path,mse,lamb=filt,pth2=pth2,lamb2=filt2).Syn()
def scanner(x):
    if 'SKYRA' in x:
        return '_SKYRA'
    if 'GE' in x:
        return '_GE'
    if 'PHILIPS' in x:
        return '_PHILIPS'
    else:
        return ''

#subjects=pass_on
#subjects=glob('/data/henry4/jjuwono/new_GM_method/*')
#subjects=[c.get_mse(x) for x in subjects]
#subjects=subjects[subjects.index('ms0245'):]
subjects=glob('/data/henry4/jjuwono/new_GM_method/ms*PSIR_*')
#exempt=['ms0064','ms0171','ms0184','ms0243','ms0245','ms0387','ms0405','ms0501','ms0511','ms0541','ms0575','ms0585','ms0587','ms0591','ms0620','ms0631','ms0645','ms0665','ms0678','ms0708','ms0725','ms0731','ms0738','ms0766','ms0779','ms0789','ms0790','ms0812','ms0820','ms0837','ms0899']
#exempt=['ms1954_GE','ms1954_PHILIPS','ms1954retest_PHILIPS','ms1954retest_GE']
exempt=[]
errors=[]
study_tag='test_retest_PSIR'
input(subjects)
for subject in subjects[6:]:
    if 'retest' in subject:
        subject=c.get_mse(subject)+'PSIR_retest'+scanner(subject)
    else:
        subject=c.get_mse(subject)+'PSIR_'+scanner(subject)
    print('#########{}######{}'.format(subject,subject))
    if subject in exempt or 'mse' in subject:
        continue
    registrations=sorted(glob('/data/henry4/jjuwono/data/henry4/jjuwono/{}/{}/warped/*.nii.gz'.format(study_tag,subject)))
    try:
        registrations.remove('/data/henry4/jjuwono/data/henry4/jjuwono/{}/{}/warped/synslice_avggmsegs.nii.gz'.format(study_tag,subject))
        registrations.remove('/data/henry4/jjuwono/data/henry4/jjuwono/{}/{}/warped/cor_synslice_avggmsegs.nii.gz'.format(study_tag,subject))
    except:
        pass
    cord_registrations=sorted(glob('/data/henry4/jjuwono/data/henry4/jjuwono/{}/{}/warped1/*.nii.gz'.format(study_tag,subject)))
    try:
        cord_registrations.remove('/data/henry4/jjuwono/data/henry4/jjuwono/{}/{}/warped1/synslice_avggmsegs.nii.gz'.format(study_tag,subject))
    except:
        pass
    t_map='/data/henry4/jjuwono/new_GM_method/{}/t_map.nii.gz'.format(subject)
    original_line_fit='/data/henry4/jjuwono/new_GM_method/{}/original_line_fit.nii.gz'.format(subject)
    target='/data/henry4/jjuwono/new_GM_method/{}/raw_im.nii.gz'.format(subject)
    olf=nb.load(original_line_fit).get_data()
    poop1=nb.load(target)
    poop1_dat=poop1.get_data()+1300
    poop2_dat=nb.load(t_map).get_data()
    x,y=poop1_dat.shape
    new=np.zeros(poop1_dat.shape)
    mean_white_interm=poop1_dat[np.where(olf<0.5)]
    mean_white=np.mean(mean_white_interm[np.where(mean_white_interm!=0)])
    print(mean_white)
    for q in range(x):
        for w in range(y):
           if poop2_dat[q,w]<=-1.5:
               new[q,w]=random.randint(int(mean_white-(mean_white*.03)),int(mean_white+(mean_white*.03)))
               #or pass? what happens with 0's?
           else:
               new[q,w]=poop1_dat[q,w]
    tmp=np.where(olf>=0.5)
    for i in range(len(tmp[0])):
        new[tmp[0][i],tmp[1][i]]=poop1_dat[tmp[0][i],tmp[1][i]]

    nb.save(nb.Nifti1Image(new,poop1.affine),'/data/henry4/jjuwono/new_GM_method/{}/cor_raw_im.nii.gz'.format(subject))
    target='/data/henry4/jjuwono/new_GM_method/{}/cor_raw_im.nii.gz'.format(subject)
    output=[]
    data_dict={}
    for dicting in range(max((len(registrations),len(cord_registrations)))):
        try:
            poop=int(os.path.basename(registrations[dicting])[0])
            print(poop)
            data_dict[os.path.basename(registrations[dicting]).split('_')[0]]=[registrations[dicting]]
            continue
        except:
            pass
        ide=c.get_mse(os.path.basename(registrations[dicting]))
        data_dict[ide]=[registrations[dicting]]
    for dicting in range(len(cord_registrations)):
        try:
            poop=int(os.path.basename(cord_registrations[dicting])[0])
            data_dict[os.path.basename(cord_registrations[dicting]).split('_')[0]].append(cord_registrations[dicting])
            continue
        except:
            pass

        ide=c.get_mse(os.path.basename(cord_registrations[dicting]))
        try:
            data_dict[ide].append(cord_registrations[dicting])
        except:
            pass

    for corrrecting in range(len(registrations)):
        try:
            idee=c.get_mse(os.path.basename(registrations[corrrecting]))
        except:
            pass
        try:
            poop=int(os.path.basename(registrations[corrrecting])[0])
            idee=os.path.basename(registrations[corrrecting]).split('_')[0]
        except:
            pass
        try:
            first=data_dict[idee][0]
            print('first:{}'.format(first))
        except:
            continue
        try:
            cord=data_dict[idee][1]
            print('cord:{}'.format(cord))
        except:
            continue
        try:
            output.append(individual_correct(first,cord,target,'test',subject))
        except:
            errors.append(subject)


