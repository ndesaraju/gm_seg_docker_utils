import nibabel as nb 
import numpy as np 
from scipy.ndimage import label as L
from glob import glob
import copy
import commonly as c
def distance_qualify(i,j,l,dis=4):
	dist=[]
	for x,y in l:
		dist.append(((x-i)**2+(y-j)**2)**0.5)
	if min(dist)>dis:
		return False
	else:
		return True
def distance(x,y):
	return (((x[0]-y[0])**2)+((x[1]-y[1])**2))**0.5
def damper(x,t):
	if t<0:
		return x
	m=0.5*1.5/0.8
	return x*m/t
def increase(x,t):
	#if t>0:
		#return x
	#m=0.5*1.5/0.8
	#return x*m/t
	return 0.5
def cut(x,t):
	#if t>0:
		#return x
	#m=0.5*1.5/0.8
	#return x*m/t
	return 0.3
	

def classify_pixel(raw_im,i,j,t_map):
	if raw_im[i,j]==0:
		if t_map[i+1,j]==0:
			return (True,3)
		elif t_map[i-1,j]==0:
			return (True,3)
		elif t_map[i,j+1]==0:
			return (True,3)
		elif t_map[i,j-1]==0:
			return (True,3)
		elif t_map[i+1,j+1]==0:
			return (True,3)
		elif t_map[i+1,j-1]==0:
			return (True,3)
		elif t_map[i-1,j+1]==0:
			return (True,3)
		elif t_map[i-1,j-1]==0:
			return (True,3)
	
	if raw_im[i,j]==0:
		return (True,0)

	if raw_im[i,j]!=0:
		if raw_im[i+1,j]==0:
			return (False,1)
		elif raw_im[i-1,j]==0:
			return (False,1)
		elif raw_im[i,j+1]==0:
			return (False,1)
		elif raw_im[i,j-1]==0:
			return (False,1)
		elif raw_im[i+1,j+1]==0:
			return (False,1)
		elif raw_im[i+1,j-1]==0:
			return (False,1)
		elif raw_im[i-1,j+1]==0:
			return (False,1)
		elif raw_im[i-1,j-1]==0:
			return (False,1)
		else:
			return (False,2)
def T_correct_pos(raw,t,soo,file_handl):

	raw_im_handl=nb.load(raw[0])
	t_map_handl=nb.load(t[0])
	raw_im=raw_im_handl.get_data()
	t_map=t_map_handl.get_data()
	im=np.where(raw_im>=0.5,raw_im,0)
	t_map_pos=np.where(t_map>=1.5,t_map,0)
	t_clust,n_clust=L(t_map_pos)
	print('needs {} corrections'.format(n_clust))
	new_im=copy.deepcopy(raw_im)

	for c in range(n_clust):
		edge_coords=[]
		working_clust=np.where(t_clust==c+1)
		work_coords=zip(working_clust[0],working_clust[1])
		work_dict={}
		for iii in list(work_coords):
			work_dict[iii]=1
		
		for i in range(len(working_clust[0])):
			claas=classify_pixel(im,working_clust[0][i],working_clust[1][i],t_map_pos)
			if claas[0]:
				del work_dict[(working_clust[0][i],working_clust[1][i])]
			elif claas[1]==1:
				edge_coords.append((working_clust[0][i],working_clust[1][i]))
				#del work_dict[(working_clust[0][i],working_clust[1][i])]
		if len(edge_coords)<=2:
			continue

		for pix in work_dict.keys():
			#print('here')
			if distance_qualify(pix[0],pix[1],edge_coords):
				#print('qualified{}{}'.format(pix[0],pix[1]))
				new_im[pix[0],pix[1]]=damper(im[pix[0],pix[1]],t_map_pos[pix[0],pix[1]])
	return new_im,raw_im_handl.affine
	nb.save(nb.Nifti1Image(new_im,raw_im_handl.affine),outputs_path+'/final_output/rereg_t_map_edit.nii.gz')
def T_correct_neg(raw,t,soo,file_handl):

	
	t_map_handl=nb.load(t[0])
	raw_im=raw
	t_map=t_map_handl.get_data()
	im=np.where(raw_im>=0.5,raw_im,0)
	t_map_neg=np.where(t_map<=-1.5,t_map,0)
	t_clust,n_clust=L(t_map_neg)
	print('needs {} corrections'.format(n_clust))
	new_im=copy.deepcopy(raw_im)


	for c in range(n_clust):
		edge_coords=[]
		edge_cluster_coords=[]

		working_clust=np.where(t_clust==c+1)
		if len(working_clust[0])>40:
			work_coords=zip(working_clust[0],working_clust[1])
			work_dict={}
			for iii in list(work_coords):
				work_dict[iii]=1
		
			for i in range(len(working_clust[0])):
				claas=classify_pixel(im,working_clust[0][i],working_clust[1][i],t_map_neg)
				if claas[1]==1:
					edge_coords.append((working_clust[0][i],working_clust[1][i]))
				if claas[1]==3:
					edge_cluster_coords.append((working_clust[0][i],working_clust[1][i]))
				if claas[0]:
					del work_dict[(working_clust[0][i],working_clust[1][i])]
			maxy=0
			distancey=0
			file_handl.write('edge_coords:{}'.format(len(edge_cluster_coords))+'\n')
			if edge_cluster_coords and edge_coords:
				edge_mean=np.mean(edge_coords,axis=0)
				edge_cluster_mean=np.mean(edge_cluster_coords,axis=0)
				file_handl.write(str(distance(edge_mean,edge_cluster_mean))+'\n')
				if distance(edge_mean,edge_cluster_mean)<10:
					continue
				#else:this part is just to mark where the centers are
					#print(int(edge_mean[0]),int(edge_mean[1]),int(edge_cluster_mean[0]),int(edge_cluster_mean[1]))
					#new_im[int(edge_mean[0]),int(edge_mean[1])]=9000
					#new_im[int(edge_cluster_mean[0]),int(edge_cluster_mean[1])]=9000
				

				



			if len(edge_coords)<=2:
				continue

			for pix in work_dict.keys():
				#print('here')
				if distance_qualify(pix[0],pix[1],edge_coords):
					#print('qualified{}{}'.format(pix[0],pix[1]))
					new_im[pix[0],pix[1]]=cut(im[pix[0],pix[1]],t_map_neg[pix[0],pix[1]])
		else:
			work_coords=zip(working_clust[0],working_clust[1])
			work_dict={}
			for iii in list(work_coords):
				work_dict[iii]=1
		
			for i in range(len(working_clust[0])):
				claas=classify_pixel(im,working_clust[0][i],working_clust[1][i],t_map_neg)
				if claas[1]==1:
					edge_coords.append((working_clust[0][i],working_clust[1][i]))
				#del work_dict[(working_clust[0][i],working_clust[1][i])]
			if len(edge_coords)<=2:
				continue

			for pix in work_dict.keys():
				#print('here')
				if distance_qualify(pix[0],pix[1],edge_coords):
					#print('qualified{}{}'.format(pix[0],pix[1]))
					new_im[pix[0],pix[1]]=increase(im[pix[0],pix[1]],t_map_neg[pix[0],pix[1]])
	return new_im

	nb.save(nb.Nifti1Image(new_im,t_map_handl.affine),outputs_path+'/final_output/rereg_t_map_edit.nii.gz')
def scanner(x):
	if 'SKYRA' in x:
		return '_SKYRA'
	if 'GE' in x:
		return '_GE'
	if 'PHILIPS' in x:
		return '_PHILIPS'
	else:
		return ''

def run_this(subject,outputs_path,prefix=0):
	file_handl=open(outputs_path+'/papers.txt','a')
	if not(prefix):
		if 'retest' in subject:
				subject=c.get_mse(subject)+'retest'+scanner(subject)
		else:
			subject=c.get_mse(subject)+scanner(subject)
	else:
		subject=prefix
	file_handl.write('t_map'+'\n')
	print(subject)
	raw=glob(outputs_path+'/final_output/raw_im.nii.gz')
	t=glob(outputs_path+'/final_output/rereg_t_map.nii.gz')
	raw=glob(outputs_path+'/final_output/rereg_original_line_fit.nii.gz')
	yes,affine=T_correct_pos(raw,t,subject,file_handl)
	nb.save(nb.Nifti1Image(T_correct_neg(yes,t,subject,file_handl),affine),outputs_path+'/final_output/rereg_t_map_edit.nii.gz')
	file_handl.close()





			


