
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
import time
from multiprocessing import Pool
from collections import defaultdict
#import cord_seg_clean as cord
import random
#used to individually correct from the gray matter algorithm most recent June 2019
def alter_ims(ima,mask,naming):

        if 'control' in ima:
                prefix=os.path.basename(ima)[0:2]
        else:
            try:
                prefix=c.get_mse(os.path.basename(ima))
            except:
                prefix=os.path.basename(ima).split("_")[0]

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
        nb.save(nb.Nifti1Image(final,raw.affine),naming+prefix+'altered.nii.gz')
        return naming+prefix+'altered.nii.gz'
	
def individual_correct(i,cord,target,naming,subject,outputs_path,file_handl):
    try:
        os.mkdir(os.path.join(outputs_path, 'working'))
    except:
        pass
    working=os.path.join(outputs_path, 'working/')
    mse=subject
    #cord_image=glob('/data/henry11/PBR/subjects/'+mse+'/nii/*only_cord*C2_3*psir*PSIR*')+glob('/data/henry7/PBR/subjects/'+mse+'/nii/*only_cord*C2_3*psir*PSIR*')
    #cord_image=glob('/data/henry6/PBR/H_cohort/*only_cord*'+mse+'*C2_3.nii.gz')
    
    try:
         pth=[alter_ims(cord,i,working)]
         file_handl.write(str((cord,i)))
    except:
        #input('I errored 1')
        raise ValueError('could not alter')
        return ' '
    #pth=[working+'altered.nii.gz']
    try:
        os.mkdir(os.path.join(outputs_path, 'registrations2'))

    except:
        pass
    try:
        os.mkdir(os.path.join(outputs_path, 'registrations2/warped'))
    except:
        pass
    
    output_path=os.path.join(outputs_path, 'registrations2/')
    static_path=target
    filt=None
    pth2=[i]
    #print(pth)
    #print(pth2)
    filt2=None
    return (pth,output_path,static_path,mse,filt,pth2,filt2)
    #return GRL.SimpleRegister(pth,output_path,static_path,mse,lamb=filt,pth2=pth2,lamb2=filt2).Syn()
def scanner(x):
    if 'SKYRA' in x:
        return '_SKYRA'
    if 'GE' in x:
        return '_GE'
    if 'PHILIPS' in x:
        return '_PHILIPS'
    else:
        return ''

def run_this(static,outputs_path,prefix=0):
    file_handl=open(os.path.join(outputs_path, 'prints.txt'),'a')
    errors=[]
    if not(prefix): 
        if 'retest' in static:
            subject=c.get_mse(static)+'retest'+scanner(static)
        else:
            subject=c.get_mse(static)+scanner(static)
    else:
        subject=prefix
        
    print('#########{}######{}'.format(subject,subject))
    file_handl.write('#########{}######{}\n'.format(subject,subject))
    registrations=sorted(glob(os.path.join(outputs_path, 'registrations1/warped/*.nii.gz')))
    try:
        registrations.remove(os.path.join(outputs_path, 'registrations1/warped/synslice_avggmsegs.nii.gz'))
        registrations.remove(os.path.join(outputs_path, 'registrations1/warped/cor_synslice_avggmsegs.nii.gz'))
    except:
        pass
    cord_registrations=sorted(glob(os.path.join(outputs_path, 'registrations1/warped1/*.nii.gz')))
    try:
        cord_registrations.remove(os.path.join(outputs_path, 'registrations1/warped1/synslice_avggmsegs.nii.gz'))
    except:
        pass
    t_map=os.path.join(outputs_path, 'quality_assurance/t_map.nii.gz')
    original_line_fit=os.path.join(outputs_path, 'quality_assurance/original_line_fit.nii.gz')
    target=static
    olf=nb.load(original_line_fit).get_data()
    poop1=nb.load(target)
    poop1_dat=poop1.get_data()
    poop2_dat=nb.load(t_map).get_data()
    x,y=poop1_dat.shape
    new=np.zeros(poop1_dat.shape)
    mean_white_interm=poop1_dat[np.where(olf<0.5)]
    mean_white=np.mean(mean_white_interm[np.where(mean_white_interm!=0)])
    file_handl.write('mean_white:{}\n'.format(mean_white))
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

    nb.save(nb.Nifti1Image(new,poop1.affine),os.path.join(outputs_path,'quality_assurance/cor_raw_im.nii.gz'))
    target=os.path.join(outputs_path,'quality_assurance/cor_raw_im.nii.gz')
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
    argus=[]
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
            #print('first:{}'.format(first))
        except:
            continue
        try:
            cord=data_dict[idee][1]
            #print('cord:{}'.format(cord))
        except:
            continue
        try:
            import pdb
            #pdb.set_trace()
            argus.append(individual_correct(first,cord,target,'test',subject,outputs_path,file_handl))
            #output.append(individual_correct(first,cord,target,'test',subject,outputs_path))
        except:
            errors.append(os.path.basename(cord).split('.')[0])
        if len(errors)>10:
            return errors
    #try:
        #print(argus)
    GRL.Syn(argus,file_handl)
    file_handl.close()
    #except:
        #return ['error with registrations']

  



