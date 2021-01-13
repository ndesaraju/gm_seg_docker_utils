
import random
import copy
import commonly as c
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
import scipy.stats as statss
import os
from subprocess import Popen,PIPE
import sklearn.preprocessing
import nibabel as nb
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
from avgim import avgim
##just to apply warps to cord and mask iamges##
from scipy.optimize import curve_fit
import logging as log
from henrygce.logging import log_gm_job_status

def sigmoid(x,x0,k,y0):
    y = 1 / (1 + np.exp(-k*(x-x0))) + y0
    return y
def hist_j(image):
    d=defaultdict(int)
    for i in image.flatten():
        d[i]=d[i]+1
    return d
def quantile_transform(image):
    dat_array=image
    nonzer_dat=dat_array[np.where(dat_array>0)]
    unique_vals=sorted(set(nonzer_dat))
    
    step=1/len(unique_vals)
    dist=norm(0,1)
    normal_data=[dist.pdf(step*(-1*int(len(unique_vals)/2)+x)) for x in range(len(unique_vals))]
    
    new_nums=[]

    d={}
    for i in range(len(unique_vals)):
        d[unique_vals[i]]=normal_data[i]
    sh=dat_array.shape
    for i in range(sh[0]):
        for j in range(sh[1]):
            if dat_array[i,j]==0:
                continue
            dat_array[i,j]=d[dat_array[i,j]]
    return dat_array
def quantile_transform(image):
    dat_array=image
    new_dats=np.zeros(dat_array.shape)
    nonzer_dat=dat_array[np.where(dat_array>0)]
    unique_vals=sorted(set(nonzer_dat))
    #print(len(unique_vals))
    #input()
    new_vals=[]
    dist=stats.norm(0,1)
    d={}
    for i in unique_vals:
        perc=stats.percentileofscore(nonzer_dat, i, kind='strict')
        
        if perc==0:
            d[i]=-3
            continue
        
        d[i]=dist.ppf(perc*.01)
 
    sh=dat_array.shape
    for i in range(sh[0]):
        for j in range(sh[1]):
            if dat_array[i,j]==0:
                continue
            new_dats[i,j]=d[dat_array[i,j]]

            
    return new_dats

def z_score_a_dat(image): 

    nonzer=image[np.where(image>0)]
    median=np.median(nonzer)
    mod_dat=nonzer[np.where(nonzer>median)]
    new_std=np.std(mod_dat)/((1-(2/math.pi))**0.5)   
    sh=image.shape
    r_image=np.zeros(sh)
    for i in range(sh[0]):
        for j in range(sh[1]):
            if image[i,j]==0:
                continue
            r_image[i,j]=(image[i,j]-median)/new_std

    return r_image
def func_l(x,a,b):
    return (a*x)+b
def rounds(x):
    if x%1>=.75:
        return int(x)+1
    else:
        return int(x)
def create_prob_seg_iteration3(template_grays,templates,image,file_handl):
        a=nb.load(image)
        adat_raw=a.get_data()
        #adat=quantile_transform(a.get_data())
        adat=a.get_data()
        distributions_raw=[]
        distributions=[]
        fgs=[]
        data_dict={}
        for i in template_grays:
            data_dict[c.get_ms(os.path.basename(i))]=i

        for i in templates:
            temp=nb.load(i).get_data()
            #temp=quantile_transform(temp)
            try:
                z=z_score_a_dat(temp)
            except:
                
                file_handl.write(str(sys.exc_info())+'\n')
                continue
            try:
                #print(data_dict[c.get_ms(os.path.basename(i))],i)
                fg=nb.load(data_dict[c.get_ms(os.path.basename(i))]).get_data()
            except:
                
                file_handl.write(str(sys.exc_info())+'\n')
                continue
            distributions.append(z)
            fgs.append(fg)
        
        return fgs,distributions,a,adat,adat_raw
def scanner(x):
    if 'SKYRA' in x:
        return '_SKYRA'
    if 'GE' in x:
        return '_GE'
    if 'PHILIPS' in x:
        return '_PHILIPS'
    else:
        return ''

#loop through input#
def run_this(static,outputs_path,subj,sess,protocol,prefix=0):
    file_handl=open(os.path.join(outputs_path, 'papers.txt'),'a')
    apply_warps=False
    if not(prefix): 
        if 'retest' in static:
            subject=c.get_mse(static)+'retest'+scanner(static)
        else:
            subject=c.get_mse(static)+scanner(static)
    else:
         subject=prefix

    mse=subject
    

    try:
        os.remove(os.path.join(outputs_path, 'registrations2/warped/synslice_avggmsegs.nii.gz'))
    except:
        pass
    log_gm_job_status("final set of warps", subj, sess, protocol)

###run process to fit lines###

    
    aff=nb.load(static)
    adat_raw=nb.load(static).get_data()
    print(outputs_path)  


###run process to grab distributions##
    #quick control helper
    controls=glob(os.path.join(outputs_path,'registrations2/warped/[0-9]*.nii.gz'))
    controls1=glob(os.path.join(outputs_path,'registrations1/warped1/[0-9]*.nii.gz'))
    control_flag=False
    control1_flag=False
    for i in controls:
        if control_flag==False:
            control_par=os.path.dirname(i)
            control_flag=True
        il=control_par+'/ms'+os.path.basename(i)
        os.rename(i,il)
    for i in controls1:
        if control1_flag==False:
            control1_par=os.path.dirname(i)
            control1_flag=True
        il=control1_par+'/ms'+os.path.basename(i)
        os.rename(i,il)
    #control helper done

    template_grays=glob(os.path.join(outputs_path, 'registrations2/warped/ms*.nii.gz'))
    templates=glob(os.path.join(outputs_path, 'registrations1/warped1/ms*.nii.gz'))
    fgs,distributions,a,adat,adat_raw=create_prob_seg_iteration3(template_grays,templates,static,file_handl) 
    avgim(os.path.join(outputs_path, 'registrations2/warped/'))


#initialize images to write
    sh=adat.shape
    slope=np.zeros(sh)
    screwed=np.zeros(sh)
    intercept=np.zeros(sh)
    confidences=np.zeros(sh)
    confidences1=np.zeros(sh)
    confidences2=np.zeros(sh)
    t_map=np.zeros(sh)
    mean_templates=np.zeros(sh)
    new_image=np.zeros(sh)
    new_image_logi=np.zeros(sh)
    original_line_fit=np.zeros(sh)
    color_im=np.zeros(sh)


    file_handl.write(str(len(fgs))+'\n')
    file_handl.write(str(len(distributions))+'\n')
    distributions=np.asarray(distributions)
    print('distributions:{}'.format(distributions.shape))
    fgs=np.asarray(fgs)
    file_handl.close()
    adat_list=[z_score_a_dat(adat)]
    count=0
    for adat_z in adat_list:
        count+=1
        for i in range(sh[0]):
            for j in range(sh[1]):
        
                if adat_raw[i,j]==0:
                    continue
                ##insert A block here for polynomial degree fitting ###
                ##average method##
                a = np.array(distributions[:,i,j])[np.newaxis]
                try:
                    params=np.polyfit(distributions[:,i,j],fgs[:,i,j],1)
                except:
                    screwed[i,j]=1
                    continue
                    
                original_line_fit[i,j]=(adat_z[i,j]*params[0])+params[1]
                if len(np.where(fgs[:,i,j]==0)[0])<40 or len(np.where(fgs[:,i,j]==1)[0])<40:
                
                    assign=(adat_z[i,j]*params[0])+params[1]
                else:
                    #print('logi')
                    group0=np.mean(distributions[:,i,j][np.where(fgs[:,i,j]==0)])
                    group1=np.mean(distributions[:,i,j][np.where(fgs[:,i,j]==1)])
                    grouped=np.where(fgs[:,i,j]!=0,distributions[:,i,j],group0)
                    grouped=np.where(fgs[:,i,j]!=1,grouped,group1)
                    params=np.polyfit(grouped,fgs[:,i,j],1)
                    logi_slope=params[0]
                    logi_inter=params[1]
                    assign=(adat_z[i,j]*params[0])+params[1]
                    if 0<=assign<=1:
                        assign=assign
                    elif assign<0:
                        assign=0
                    else:
                        assign=1
                new_image_logi[i,j]=assign

                ##reverse method##
                if len(np.where(fgs[:,i,j]==0)[0])<25 or len(np.where(fgs[:,i,j]==1)[0])<25:
                    try:
                        params=np.polyfit(distributions[:,i,j],fgs[:,i,j],1)
                    except:
                        continue
                    new_image[i,j]=(adat_z[i,j]*params[0])+params[1]
                    if len(np.where(fgs[:,i,j]==0)[0])<60:
                        color_im[i,j]=1
                else:
                    try:
                        params,covs=np.polyfit(fgs[:,i,j],distributions[:,i,j],1,cov=True)
                    except:
                        continue

                    slope[i,j]=params[0]
                    intercept[i,j]=params[1]
                    if abs(params[0])<2*(covs[0,0]**0.5):
                        new_image[i,j]=statss.mode(fgs[:,i,j],axis=None)[0][0]
                        color_im[i,j]=2
                    else:
                        new_image[i,j]=(adat_z[i,j]-params[1])/params[0]
                        if np.isnan(covs[0,1]):
                            #input('whyyyyyyyyyyyyuyuyyyyyyyy')
                            covs[0,1]=0
                        confidence=(abs((covs[1,1]/((adat_z[i,j]-params[1])**2))+(covs[0,0]/((params[0])**2))-(2*((abs(covs[0,1]))**0.5)/((adat_z[i,j]-params[1])*params[0]))))**0.5
                    
                        confidences[i,j]=confidence/new_image[i,j]
                        if confidence>=2:
                            new_image[i,j]=-1000
                        color_im[i,j]=3
        
                mean_template=np.mean(distributions[:,i,j])
                std_template=np.std(distributions[:,i,j])
                mean_templates[i,j]=mean_template
                t_map[i,j]=(adat_z[i,j]-mean_template)/std_template

        try:
            os.mkdir(os.path.join(outputs_path, 'final_output'))
        except:
            pass
        #nb.save(nb.Nifti1Image(confidences,aff.affine),'/data/henry4/jjuwono/new_GM_method/'+mse+'/confidence.nii.gz')
        #nb.save(nb.Nifti1Image(new_image,aff.affine),'/data/henry4/jjuwono/new_GM_method/'+mse+'/new_image.nii.gz')
        #nb.save(nb.Nifti1Image(new_image_logi,aff.affine),'/data/henry4/jjuwono/new_GM_method/'+mse+'/new_image_logi.nii.gz')
        nb.save(nb.Nifti1Image(t_map,aff.affine), os.path.join(outputs_path, 'final_output/rereg_t_map.nii.gz'))
        #nb.save(nb.Nifti1Image(color_im,aff.affine),'/data/henry4/jjuwono/new_GM_method/'+mse+'/color_im.nii.gz')
        nb.save(nb.Nifti1Image(original_line_fit,aff.affine), os.path.join(outputs_path, 'final_output/rereg_original_line_fit.nii.gz'))
    return 1

