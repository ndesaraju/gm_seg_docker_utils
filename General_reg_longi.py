#import nipype.interfaces.ants as ants
#from nipype.pipeline.engine import Node, Workflow, MapNode


#import nipype.interfaces.fsl as fsl
#from nipype.utils.filemanip import load_json
import os
from subprocess import Popen
from subprocess import PIPE
import commonly as c
import numpy as np
import nibabel as nib
from collections import defaultdict
from glob import glob
from time import sleep
import logging as log
from subprocess import check_output

def check_finished(ind,cycle):
	log.info('cycle:{}'.format(cycle))
	printer=[]
	sleep(90)
	out = check_output(["ps","-aux"])
	# log.info(out)
	fin=0
	for i in ind:
		printer.append(i.poll())
		if i.poll()==None:
			fin=fin+1
	log.info(printer)
	log.info('{}/{} processes finished'.format(len(ind)-fin,len(ind)))
	if fin>0:
		return check_finished(ind,cycle+1)
	return True


def savenii(data, aff, path):
    img = nib.Nifti1Image(data, aff)
    img.get_data_dtype() == np.dtype(np.int16)
    img.set_qform(aff, 1)
    img.set_sform(aff, 1)
    img.to_filename(path)

def loadnii(path):
    im = nib.load(path)
    aff = im.affine
    data = im.get_data()
    return data, aff

def Syn(arg,file_handl,cycle_size):
	dim=2
	grad_step=0.1
	bins=32
	convergence='500x500x500'
	convrg_thresh='1e-5'
	shrink_factors='4x1x1'
	smoothing_sigmasr='1x1x1'
	smoothing_sigmas='1x1x1'
	import pdb
	#pdb.set_trace()
	job_array=[]
	job1_array=[]
	count = 0
	for argum in arg:
		files = argum[0] #altered with mse
		output_path=argum[1] #registrations2
		static_path=argum[2] #cor_raw_im
		mse=argum[3] #target mse
		files2=argum[5]#alteredmatching_mask

		if not os.path.exists(output_path):
			os.makedirs(output_path)
			#print(output_path)
			os.makedirs(os.path.join(output_path, 'warped'))

		for i in range(len(files)):
			if 'mse' not in os.path.basename(files[i]):
				mse2=os.path.basename(files[i])[0:2]
			else:
				mse2=c.get_mse(os.path.basename(files[i]))
			#print('file:{}'.format(i))
			metric_str='MI['+','.join([static_path,files[i],'1','32'])+']'
			metric_str1='CC['+','.join([static_path,files[i],'1','3'])+']'
			metric_str2='Mattes['+','.join([static_path,files[i],'1','32'])+']'

			cmd=['/opt/ants-2.3.1/antsRegistration','--dimensionality',str(dim),'--output',os.path.join(output_path,'warp'+ os.path.basename(files[i]).split('.')[0]),'--transform','Rigid['+str(grad_step)+']','--metric',metric_str2,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmasr),'--transform','Affine['+str(grad_step)+']','--metric',metric_str2,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmas),'--transform','SyN[0.1,2,1]','--metric',metric_str2,'--convergence','[200x200x200,1e-6,10]','--shrink-factors','1x1x1','--smoothing-sigmas','1x1x1','--transform','SyN[0.1,2,1]','--metric',metric_str1,'--convergence','[500x500x500,1e-6,10]','--shrink-factors','1x1x1','--smoothing-sigmas','1x1x1']
			cmd1=['/opt/ants-2.3.1/WarpImageMultiTransform',str(dim),files2[i],os.path.join(os.path.join(output_path, 'warped/'), os.path.basename(files2[i]).split('.')[0]+mse+mse2+'.nii.gz'), os.path.join(output_path,'warp'+os.path.basename(files[i]).split('.')[0]+'1Warp.nii.gz'), os.path.join(output_path, 'warp'+os.path.basename(files[i]).split('.')[0]+'0GenericAffine.mat'),'-R',static_path]
			job_array.append(Popen(cmd))
			count += 1
			job1_array.append((cmd1))
			if count>cycle_size:
				check_finished(job_array,0)
				count = 0
	
	# for cmd in job_array:
	# 	proc = Popen(cmd,stdout=PIPE)
	# 	proc.wait()


	if check_finished(job_array,0):
		for i in range(len(job1_array)):
			proc1=Popen(job1_array[i],stdout=PIPE)
			proc1.wait()
	file_handl.write(str(job1_array)+'\n')
			





	

	
