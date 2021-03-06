#import nipype.interfaces.ants as ants
#from nipype.pipeline.engine import Node, Workflow, MapNode


#import nipype.interfaces.fsl as fsl
#from nipype.utils.filemanip import load_json
import os
from subprocess import Popen
from subprocess import PIPE
from time import sleep
import numpy as np
import nibabel as nib
import multiprocessing as mp
import logging as log
from subprocess import check_output
from henrygce.logging import log_gm_job_status, log_gm_area

class SimpleRegister:
	def __init__(self,pth,output_path,static_path,file_handl,cycle_size, subj, sess, protocol, lamb=None,pth2=None,lamb2=None):
		if lamb != None:
			log.info(lamb)
			exec('from '+lamb+ ' import '+lamb,globals())
			log.info(every6('asdf'))
			exec('lambs='+lamb,globals())
			
			log.info(type(lambs))
		else:
			exec('lambs=lambda x:True',globals())
			#lambs=lambda x:True
			
		if lamb2 != None:
			exec('from '+lamb2+ ' import '+lamb2,globals())
			exec('lambs2 ='+lamb2)

		else:
			lambs2=lambda x:True

		self.pth=os.path.dirname(pth)
		# print("this value should be '/flywheel/v0/input/pth/':")
		# print(os.path.dirname(pth), self.pth)
		self.output_path=output_path
 #               if not os.path.exists(output_path):
#                        os.makedirs(self.output_path)
		self.static_path=static_path
		self.filter=lambs
		self.file_handl=file_handl
		if os.path.isfile(pth):
			self.files=[os.path.basename(pth)]
		else:
			self.files = sorted([f for f in os.listdir(pth) if self.filter(f)])
			self.files.remove(".DS_Store")
		if pth2==None:
			self.pth2=self.pth
			self.files2=self.files
		else:
			self.pth2=pth2
			self.files2=sorted([f for f in os.listdir(pth2) if lambs2(f)])

		self.subj = subj
		self.sess = sess
		self.protocol = protocol
		self.cycle_size = cycle_size
	def check_finished(self,ind,cycle):
		# import pdb
		# pdb.set_trace()
		log.info('cycle:{}'.format(cycle))
		printer=[]
		sleep(90)
		out = check_output(["ps","-aux"])
		log.info(out)
		fin=0
		# import pdb
		# pdb.set_trace()
		count = 0
		for i in ind:
			out = []
			if i.stdout:
				for n in np.arange(20):
					if n % 2 == 0:
						sleep(1)
					out.append(i.stdout.readline())
			# pdb.set_trace()
				f = open('/flywheel/v0/output/{}_error.txt'.format(count), 'w')
				# print(out)
				for line in out:
					# print(line)
					f.write(line.decode('utf-8'))
				f.close()
			printer.append(i.poll())
			if i.poll()==None:
				fin=fin+1
			count+=1
		log.info(printer)
		log_gm_job_status("in progress; first cycle {}/{} finished".format(len(ind)-fin, len(ind)), self.subj, self.sess, self.protocol)
		log.info('{}/{} processes finished'.format(len(ind)-fin,len(ind)))
		# pdb.set_trace()
		if fin>0:
			return self.check_finished(ind,cycle+1)
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


	def rigid(self,dim=2,grad_step=0.1,bins=32,convergence='500x500x500',convrg_thresh='1e-5',shrink_factors='1x1x1',smoothing_sigmas='1x1x1'):
		if not os.path.exists(self.output_path):
			os.makedirs(self.output_path)
			os.makedirs(os.path.join(self.output_path,'warped'))
		#create warped directory in outputh path 

		for i in range(len(self.files)):

			log.info(i)
			metric_str='MI['+','.join([self.static_path,os.path.join(self.pth, self.files[i]),'1','32'])+']'

			cmd=['/opt/ants-2.3.1/antsRegistration','--dimensionality',str(dim),'--output', os.path.join(os.path.join(self.output_path, 'warp'), self.files[i].split('.')[0]),'--transform','Rigid['+str(grad_step)+']','--metric',metric_str,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmas)]
			proc=Popen(cmd,stdout=PIPE)
			proc.wait()


			cmd=['/opt/ants-2.3.1/WarpImageMultiTransform',str(dim),os.path.join(self.pth2, self.files2[i]),os.path.join(os.path.join(self.output_path,'warped/'), self.files2[i].split('.')[0]+'.nii.gz'), os.path.join(os.path.join(self.output_path, 'warp'), self.files[i].split('.')[0]+'0GenericAffine.mat'),'-R',self.static_path]
			proc=Popen(cmd,stdout=PIPE)
			proc.wait()

	def affine(self,dim=2,grad_step=0.1,bins=32,convergence='500x500x500',convrg_thresh='1e-5',shrink_factors='1x1x1',smoothing_sigmasr='1x1x1',smoothing_sigmas='1x1x1'):
		if not os.path.exists(self.output_path):
			os.makedirs(self.output_path)
			os.makedirs(os.path.join(self.output_path,'warped'))


		#create warped directory in outputh path 

		for i in range(len(self.files)):

			log.info(i)
			metric_str='MI['+','.join([self.static_path,os.path.join(self.pth,self.files[i]),'1','32'])+']'
			#metric_str='CC['+','.join([self.static_path,self.pth+self.files[i],'1','2'])+']'


			cmd=['antsRegistration','--dimensionality',str(dim),'--output',os.path.join(os.path.join(self.output_path,'warp'),self.files[i].split('.')[0]),'--transform','Rigid['+str(grad_step)+']','--metric',metric_str,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmasr),'--transform','Affine['+str(grad_step)+']','--metric',metric_str,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmas)]
			proc=Popen(cmd,stdout=PIPE)
			proc.wait()


			cmd=['WarpImageMultiTransform',str(dim),os.path.join(self.pth2,self.files2[i]),os.path.join(os.path.join(self.output_path,'warped/'),self.files2[i].split('.')[0]+'.nii.gz'),os.path.join(os.path.join(self.output_path,'warp'), self.files[i].split('.')[0]+'0GenericAffine.mat'),'-R',self.static_path]
			proc=Popen(cmd,stdout=PIPE)
			proc.wait()
	def super_affine(self,subject_id,dim=3,grad_step=0.1,bins=32,convergence='400x300x100x500',convrg_thresh='1e-6',shrink_factors='8x4x2x1',smoothing_sigmasr='2x2x1x1',smoothing_sigmas='2x2x1x1'):
		if not os.path.exists(self.output_path):
			os.makedirs(self.output_path)
			os.makedirs(os.path.join(self.output_path,'warped'))

		job_array=[]
		job1_array=[]
		#create warped directory in outputh path 
		rawr=1
		o_path='/data/henry4/jjuwono/EPIC_jacobian_affine_low_res/{}/'.format(subject_id)
		for i in range(len(self.files)):
			if '.nii.gz' not in self.files[i]:
				continue
			if self.files[i]==os.path.basename(self.static_path):
				rawr=rawr+1
				continue
			log.info('poopity_scoop')
			log.info(self.files[i])
			metric_strm='MI['+','.join([self.static_path,os.path.join(self.pth,self.files[i]),'1','32','Regular','0.25'])+']'
			metric_str='CC['+','.join([self.static_path,os.path.join(self.pth, self.files[i]),'0.4','8','Regular','0.5'])+']'

			
			#cmd=['antsRegistration','--dimensionality',str(dim),'--output',self.output_path+'warp'+self.files[i].split('.')[0],'--winsorize-image-intensities','[0.02,0.97]','--initial-moving-transform','['+o_path+os.path.basename(self.static_path)+','+o_path+self.files[i]+',1]','--transform','Rigid['+str(grad_step)+']','--metric',metric_str,'--convergence','[50x50x50x100'+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmasr),'--transform','Affine['+str(grad_step)+']','--metric',metric_strm,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmas),'--transform','Affine['+str(grad_step)+']','--metric',metric_str,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmas)]
			#input(cmd)
	


			cmd1=['/opt/ants-2.3.1/WarpImageMultiTransform',str(dim),os.path.join(self.pth2,self.files2[i]),os.path.join(os.path.join(self.output_path,'warped/'),self.files2[i].split('.')[0]+'.nii.gz'),os.path.join(os.path.join(self.output_path, 'warp'), self.files[i].split('.')[0]+'0GenericAffine.mat'),'-R',self.static_path]
			cmd2=[]
			#job_array.append(Popen(cmd,stdout=PIPE))
			job1_array.append((cmd1,cmd2))
		#if self.check_finished(job_array,0):
		if True:
			submi=0
			for i in range(len(self.files)):
				if '.nii.gz' not in self.files[i]:
					submi=submi+1
					continue
				if self.files[i]==os.path.basename(self.static_path):
					submi=submi+1
					continue
				proc1=Popen(job1_array[i-submi][0],stdout=PIPE)
				#proc2=Popen(job1_array[i][1],stdout=PIPE)

	

	def Syn(self,gilroy=False,dim=2,grad_step=0.1,bins=32,convergence='500x500x500',convrg_thresh='1e-5',shrink_factors='4x1x1',smoothing_sigmasr='1x1x1',smoothing_sigmas='1x1x1'):
		if not os.path.exists(self.output_path):
			os.makedirs(self.output_path)
			os.makedirs(os.path.join(self.output_path, 'warped'))
		if not os.path.exists(os.path.join(self.output_path, 'warped1')):
			os.makedirs(os.path.join(self.output_path, 'warped1'))


		#create warped directory in outputh path 
		# log.info("made it")
		# instantiate array that will be filled with jobs
		job_array = []
		job1_array=[]
		cmd_array=[]
		count=0
		# log.info("made it here")
		for i in range(len(self.files)):
			if self.files[i] != '.DS_Store':
				fil=self.files[i]
				fil2=self.files2[i]
				# log.info("made it further")
				# self.static_path is fixed img, self.files[i] is moving img
				# print(self.pth)
				# print(os.path.join(self.pth,fil))
				metric_str='MI['+','.join([self.static_path,os.path.join(self.pth,fil),'1','32'])+']'
				metric_str1='CC['+','.join([self.static_path,os.path.join(self.pth,fil),'1','3'])+']'
				metric_str2='Mattes['+','.join([self.static_path,os.path.join(self.pth,fil),'1','32'])+']'
				# print(self.output_path,os.path.join(self.pth, fil),os.path.join(self.pth2,fil2))

				# cmd=['/opt/ants-2.3.1/antsRegistration', '-v','--dimensionality',str(dim),'--output',os.path.join(self.output_path,'warp'+fil.split('.')[0]),'--transform','Rigid['+str(grad_step)+']','--metric',metric_str2,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmasr),'--transform','Affine['+str(grad_step)+']','--metric',metric_str2,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmas),'--transform','SyN[0.1,2,1]','--metric',metric_str,'--convergence','[500x500x500,1e-6,10]','--shrink-factors','4x1x1','--smoothing-sigmas','1x1x1','--transform','SyN[0.1,2,1]','--metric',metric_str1,'--convergence','[500x500x500,1e-6,10]','--shrink-factors','4x1x1','--smoothing-sigmas','1x1x1']
				cmd=['/opt/ants-2.3.1/antsRegistration', '--dimensionality',str(dim),'--output',os.path.join(self.output_path,'warp'+fil.split('.')[0]),'--transform','Rigid['+str(grad_step)+']','--metric',metric_str2,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmasr),'--transform','Affine['+str(grad_step)+']','--metric',metric_str2,'--convergence','['+convergence+','+convrg_thresh+',10]','--shrink-factors',str(shrink_factors),'--smoothing-sigmas',str(smoothing_sigmas),'--transform','SyN[0.1,2,1]','--metric',metric_str,'--convergence','[500x500x500,1e-6,10]','--shrink-factors','4x1x1','--smoothing-sigmas','1x1x1','--transform','SyN[0.12,2,1.0]','--metric',metric_str1,'--convergence','[500x500x1000,1e-6,10]','--shrink-factors','4x1x1','--smoothing-sigmas','1x1x1']

				# log.info(cmd)
				# print(cmd)
				cmd1=['/opt/ants-2.3.1/WarpImageMultiTransform',str(dim),os.path.join(self.pth2,fil2),os.path.join(os.path.join(self.output_path,'warped/'), fil2.split('.')[0]+'.nii.gz'),os.path.join(self.output_path, 'warp'+fil.split('.')[0]+'1Warp.nii.gz'),os.path.join(self.output_path,'warp'+fil.split('.')[0]+'0GenericAffine.mat'),'-R',self.static_path]
				cmd2=['/opt/ants-2.3.1/WarpImageMultiTransform',str(dim),os.path.join(self.pth,fil),os.path.join(os.path.join(self.output_path,'warped1/'), fil.split('.')[0]+'.nii.gz'),os.path.join(self.output_path,'warp'+fil.split('.')[0]+'1Warp.nii.gz'),os.path.join(self.output_path, 'warp'+fil.split('.')[0]+'0GenericAffine.mat'),'-R',self.static_path]
				# print(cmd1)
				# print(cmd2)	
				#grid_job = self.grid_submit( cmd+cmd1+cmd2, '{}_{}_reg'.format(self.static_path.split('/')[-1][:4], i)) 
				# job_array.append(Popen(cmd, stdout=PIPE))
				job_array.append(Popen(cmd))
				job1_array.append((cmd1,cmd2))
				cmd_array.append(cmd)
				count+=1
				if gilroy:
					if count>self.cycle_size:
						self.check_finished(job_array,0)
						count=0

		if self.check_finished(job_array,0):
			for i in range(len(self.files)):
				proc1=Popen(job1_array[i][0],stdout=PIPE)
				proc2=Popen(job1_array[i][1],stdout=PIPE)
				proc1.wait()
				proc2.wait()
			self.file_handl.write(str(job1_array)+'\n')

