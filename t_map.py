import nibabel as nb 
import matplotlib.pyplot as plt
import numpy as np 
from scipy.ndimage import label as L
from glob import glob
import copy
import commonly as c
from scipy.ndimage import binary_erosion,binary_dilation 
def distance_qualify(i,j,l,dis=4):
	dist=[]
	for x,y in l:
		dist.append(((x-i)**2+(y-j)**2)**0.5)
	if min(dist)>dis:
		return False
	else:
		return True
def distance_qualify_out(i,j,edge_cluster_coords,dis=4):
	coords=zip(*edge_cluster_coords)
	a=np.zeros(im.shape)
	b=np.zeros(im.shape)
	a[i,j]=1
	b[coords]=1
	a=binary_dilation(a,iterations=dis)
	c=a*b
	if np.sum(c):
		return True
	else:
		return False
	
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
def outside_coords(raw,working_clust,cord_im):
	total_area=len(np.where(cord_im!=0)[0])
	working_clust_im=np.zeros(raw.shape)
	for i in range(len(working_clust[0])):
		working_clust_im[working_clust[0][i],working_clust[1][i]]=1
	raw=raw+1
	count=np.sum(working_clust_im)
	raw=np.where(raw==1,raw,0)
	leftovers=working_clust_im*raw
	t_clust,n_clust=L(leftovers)
	return leftovers,count,t_clust,n_clust,total_area

def valid_for_neg(leftovers,count,t_clust,n_clust,total_area):
	num_to_return=[]
	if np.sum(leftovers)>=count:
		return []
	for q in range(n_clust):
		coords_of_clust=np.where(t_clust==q+1)
		if len(coords_of_clust[0])>=total_area/80:
			num_to_return.append((q+1,coords_of_clust))

	return num_to_return

def valid_for_pos(leftovers,count,t_clust,n_clust,total_area):
	num_to_return=0
	if np.sum(leftovers)>=count:
		return [0,0],0
	maxi=[0,0]
	for q in range(n_clust):
		measure=len(np.where(t_clust==q+1)[0])
		if measure>=maxi[0]:
			maxi[0]=measure
			maxi[1]=q+1
	t_clust=np.where(t_clust!=maxi[1],t_clust,0)
	return maxi,t_clust



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
		#elif raw_im[i+1,j+1]==0:
		#	return (False,1)
		#elif raw_im[i+1,j-1]==0:
		#	return (False,1)
		#elif raw_im[i-1,j+1]==0:
		#	return (False,1)
		#elif raw_im[i-1,j-1]==0:
		#	return (False,1)
		else:
			return (False,2)
def T_correct_pos(raw,t,soo,cord_im):

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
		clusts,t_clust_pos=valid_for_pos(*outside_coords(im,working_clust,cord_im))
		if not(clusts[1]):
			continue
		else:
			t_clust=(-1*(t_clust_pos-1))*t_clust
			working_clust=np.where(t_clust!=0)



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
			if distance_qualify(pix[0],pix[1],edge_coords,3):
				#print('qualified{}{}'.format(pix[0],pix[1]))
				new_im[pix[0],pix[1]]=damper(im[pix[0],pix[1]],t_map_pos[pix[0],pix[1]])
	return new_im,raw_im_handl.affine
	nb.save(nb.Nifti1Image(new_im,raw_im_handl.affine),'/data/henry4/jjuwono/new_GM_method/{}/rereg_t_map_edit.nii.gz'.format(soo))
def T_correct_neg(raw,t,soo,cord_im):
	#honestly i added so many qualifications this thing is kind of a mess so as of November  21 before the patch 
	#that only cuts out large lesions use this one. 
	print('helllllo')
	t_map_handl=nb.load(t[0])
	raw_im=raw
	t_map=t_map_handl.get_data()
	im=np.where(raw_im>=0.5,raw_im,0)
	t_map_neg=np.where(t_map<=-1.5,t_map,0)
	t_clust,n_clust=L(t_map_neg)
	print('needs {} corrections'.format(n_clust))
	new_im=copy.deepcopy(raw_im)
	confused_im=np.zeros(raw_im.shape)
	soup=np.zeros(im.shape)
	soups=np.zeros(im.shape)
	for c in range(n_clust):
		edge_coords=[]
		edge_cluster_coords=[]
		working_clust_parent=np.where(t_clust==c+1)

		if len(working_clust_parent[0])>140:
			clusts=valid_for_neg(*outside_coords(im,working_clust_parent,cord_im))
			if not(clusts):
				continue
			for ind in clusts:
				working_clust=working_clust_parent
				tmp=np.zeros(im.shape)
				tmp[working_clust]=1
				tmp=tmp*im
				tmp[ind[1]]=10
				tmp=binary_erosion(tmp).astype(tmp.dtype)
				working_clust=np.where(tmp!=0)
				
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
				print('edge_coords:{}'.format(len(edge_cluster_coords)))
				if edge_cluster_coords and edge_coords:
					edge_mean=np.mean(edge_coords,axis=0)
					edge_cluster_mean=np.mean(edge_cluster_coords,axis=0)
					print(distance(edge_mean,edge_cluster_mean))

				if len(edge_coords)<=2:
					continue
				
				for pix in work_dict.keys():	
					if distance_qualify(pix[0],pix[1],edge_coords):
						#print('qualified{}{}'.format(pix[0],pix[1]))
						new_im[pix[0],pix[1]]=cut(im[pix[0],pix[1]],t_map_neg[pix[0],pix[1]])
		else:
			working_clust=working_clust_parent
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

	nb.save(nb.Nifti1Image(new_im,t_map_handl.affine),'/data/henry4/jjuwono/new_GM_method/{}/rereg_t_map_edit.nii.gz'.format(soo))
# def T_correct_neg(raw,t,soo,cord_im):
# 	#honestly i added so many qualifications this thing is kind of a mess so as of November  21 before the patch 
# 	#that only cuts out large lesions use this one. 
	
# 	t_map_handl=nb.load(t[0])
# 	raw_im=raw
# 	t_map=t_map_handl.get_data()
# 	im=np.where(raw_im>=0.5,raw_im,0)
# 	t_map_neg=np.where(t_map<=-1.5,t_map,0)
# 	t_clust,n_clust=L(t_map_neg)
# 	print('needs {} corrections'.format(n_clust))
# 	new_im=copy.deepcopy(raw_im)


# 	for c in range(n_clust):
# 		edge_coords=[]
# 		edge_cluster_coords=[]

# 		working_clust=np.where(t_clust==c+1)
# 		if len(working_clust[0])>140:
# 			outside_coords(im,working_clust)
# 			work_coords=zip(working_clust[0],working_clust[1])
# 			work_dict={}
# 			for iii in list(work_coords):
# 				work_dict[iii]=1
		
# 			for i in range(len(working_clust[0])):
# 				claas=classify_pixel(im,working_clust[0][i],working_clust[1][i],t_map_neg)
# 				if claas[1]==1:
# 					edge_coords.append((working_clust[0][i],working_clust[1][i]))
# 				if claas[1]==3:
# 					edge_cluster_coords.append((working_clust[0][i],working_clust[1][i]))
# 				if claas[0]:
# 					del work_dict[(working_clust[0][i],working_clust[1][i])]
# 			maxy=0
# 			distancey=0
# 			print('edge_coords:{}'.format(len(edge_cluster_coords)))
# 			if edge_cluster_coords and edge_coords:
# 				edge_mean=np.mean(edge_coords,axis=0)
# 				edge_cluster_mean=np.mean(edge_cluster_coords,axis=0)
# 				print(distance(edge_mean,edge_cluster_mean))
# 				if distance(edge_mean,edge_cluster_mean)<10:
# 					continue
# 				#else:this part is just to mark where the centers are
# 					#print(int(edge_mean[0]),int(edge_mean[1]),int(edge_cluster_mean[0]),int(edge_cluster_mean[1]))
# 					#new_im[int(edge_mean[0]),int(edge_mean[1])]=9000
# 					#new_im[int(edge_cluster_mean[0]),int(edge_cluster_mean[1])]=9000

# 			if len(edge_coords)<=2:
# 				continue

# 			for pix in work_dict.keys():
# 				#print('here')
# 				if distance_qualify(pix[0],pix[1],edge_coords):
# 					#print('qualified{}{}'.format(pix[0],pix[1]))
# 					new_im[pix[0],pix[1]]=cut(im[pix[0],pix[1]],t_map_neg[pix[0],pix[1]])
# 		else:
# 			work_coords=zip(working_clust[0],working_clust[1])
# 			work_dict={}
# 			for iii in list(work_coords):
# 				work_dict[iii]=1
		
# 			for i in range(len(working_clust[0])):
# 				claas=classify_pixel(im,working_clust[0][i],working_clust[1][i],t_map_neg)
# 				if claas[1]==1:
# 					edge_coords.append((working_clust[0][i],working_clust[1][i]))
# 				#del work_dict[(working_clust[0][i],working_clust[1][i])]
# 			if len(edge_coords)<=2:
# 				continue

# 			for pix in work_dict.keys():
# 				#print('here')
# 				if distance_qualify(pix[0],pix[1],edge_coords):
# 					#print('qualified{}{}'.format(pix[0],pix[1]))
# 					new_im[pix[0],pix[1]]=increase(im[pix[0],pix[1]],t_map_neg[pix[0],pix[1]])
# 	return new_im

#	nb.save(nb.Nifti1Image(new_im,t_map_handl.affine),'/data/henry4/jjuwono/new_GM_method/{}/rereg_t_map_edit.nii.gz'.format(soo))
def scanner(x):
	if 'SKYRA' in x:
		return '_SKYRA'
	if 'GE' in x:
		return '_GE'
	if 'PHILIPS' in x:
		return '_PHILIPS'
	else:
		return ''




if __name__ == '__main__':
	subjects=glob('/data/henry4/jjuwono/new_GM_method/ms[0-9][0-9][0-9]*[0-9]')
	#exempt=['ms1954_GE','ms1954_PHILIPS','ms1954retest_PHILIPS','ms1954retest_GE']
	exempt=[]
	study_tag='test_retest'
	for subject in subjects:
		print(subject)
		#if 'ms1954_SKYRA' not in subject:
		#	continue
		if 'PSIR' in subject:
			continue
		if 'retest' in subject:
 			subject=c.get_mse(subject)+'retest'+scanner(subject)
		else:
			subject=c.get_mse(subject)+scanner(subject)
		if subject in exempt or 'mse' in subject:
			continue
		
		t=glob('/data/henry4/jjuwono/new_GM_method/{}/rereg_t_map.nii.gz'.format(subject))
		raw=glob('/data/henry4/jjuwono/new_GM_method/{}/rereg_original_line_fit.nii.gz'.format(subject))
		cord_im=glob('/data/henry4/jjuwono/new_GM_method/{}/raw_im.nii.gz'.format(subject))
		#try:
		if not(raw and t and cord_im):
			continue
		cord_im=nb.load(cord_im[0]).get_data()
		yes,affine=T_correct_pos(raw,t,subject,cord_im)
		nb.save(nb.Nifti1Image(T_correct_neg(yes,t,subject,cord_im),affine),'/data/henry4/jjuwono/new_GM_method/{}/rereg_t_map_edit.nii.gz'.format(subject))

		#except:
		print('error:{}'.format(subject))





			


