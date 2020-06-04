import numpy as np
import nibabel as nb
import os
def avgim(path,im=None):
	try:
		os.remove(os.path.join(path,'synslice_avggmsegs.nii.gz'))
	except:
		pass
	#im is reference for affine
	files=os.listdir(path)
	if im==None:
		im=path+files[0]
	#try:
		#a=nb.load(path+files.pop(files.index(im)))
	#except:
	a=nb.load(im)
	aff=a.affine
	avg=np.zeros(a.get_data().shape)
	for i in files:
		print(i)
		print(avg.shape)
		try:
			handle=nb.load(path+i)
		except:
			continue
		avg=avg+handle.get_data()

	def savenii(data, aff, path):
	    img = nb.Nifti1Image(data, aff)
	    img.get_data_dtype() == np.dtype(np.float32)
	    img.set_qform(aff, 1)
	    img.set_sform(aff, 1)
	    print(path)
	    img.to_filename(path)

	savenii(avg/len(files),aff,os.path.join(path,'synslice_avggmsegs.nii.gz'))
	return avg/len(files)

