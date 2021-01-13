from glob import glob
import nibabel as nb
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict 
def test_different_words(x,y):
	return y in x

d=defaultdict(list)	
raw_data=glob('/data/henry4/jjuwono/new_GM_method/ms*_*/rereg_original_line_fit.nii.gz')
for x in raw_data:
	if not(any(map(test_different_words,[x,x],['PSIR','only_cord','zoomed']))):
		#d[x.split('/')[5]+x.split('/')[6]].append((x,len(x)))
		d[''.join(x.split('/')[5].split('retest'))].append((x,len(x)))
for i in d:
	if len(d[i])>2:
		#print('yes',d[i])
		d[i]=sorted(d[i],key=lambda x:x[1])[:2]
	print(d[i])
	datas=nb.load(d[i][0][0]).get_data()
	try:
		datas1=nb.load(d[i][1][0]).get_data()
	except:
		datas1=np.asarray([])
	datas_shape=datas.shape
	datas1_shape=datas1.shape
	subtraction_datas=int((datas_shape[0]-215)/2)
	subtraction_datas1=int((datas1_shape[0]-215)/2)
	datas=datas[subtraction_datas:-subtraction_datas,subtraction_datas:-subtraction_datas]
	r=''
	if 'retest' in d[i][0][0]:
		r='retest'
	datas_PIL=PIL.Image.fromarray(datas)
	plt.imsave('/data/henry4/jjuwono/ims4figs/{}'.format(i+r+os.path.basename(d[i][0][0]).split('.')[0]),datas,cmap='Reds',vmin=0.5,vmax=0.51)
	if datas1_shape[0]:
		r1=''
		if 'retest' in d[i][1][0]:
			r1='retest'
		datas1=datas1[subtraction_datas1:-subtraction_datas1,subtraction_datas1:-subtraction_datas1]
		datas1_PIL=PIL.Image.fromarray(datas1)
		plt.imsave('/data/henry4/jjuwono/ims4figs/{}'.format(i+r1+os.path.basename(d[i][1][0]).split('.')[0]),datas1,cmap='Reds',vmin=0.5,vmax=0.51)
	


#for t2star slice 9
