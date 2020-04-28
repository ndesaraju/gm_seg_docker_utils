import numpy as np
from scipy.ndimage.interpolation import zoom
import nibabel as nb
from sklearn.cluster import KMeans

def kmeans_im(image_path,save_path,clusters,affi=None):
	if type(image_path)== str: 
		image=nb.load(image_path)
		im=image.get_data()
		aff=image.affine
	else:
		im=image_path
		if np.any(affi)==None:
			raise ValueError('need affine')
		aff=affi
		
	sh=im.shape
	x=sh[0]
	y=sh[1]
	z=sh[2]
	print('{},{},{}'.format(x,y,z))
	data=[]
	for i in range(x):
		for j in range(y):
			for k in range(z):
				#tmp=[i,j,im[i,j]]
				tmp=im[i,j,k]
				data.append(tmp)
	data=np.asarray(data)
	data=data.reshape(-1,1)
	k=KMeans(n_clusters=clusters).fit(data)
	new_im=k.labels_.reshape(im.shape)
	nb.save(nb.Nifti1Image(new_im,aff),save_path)
	return new_im,aff,k.cluster_centers_

def get_mse(string):
    sp=string
    while True:
            try:
                    i=sp.index('mse')
            except:
                    n=get_ms(string)
                    return n
            sp=sp[i+3:]
            n=subsequent_nums(sp)
            if len(n)>0:
                    return 'mse'+n
def get_ms(string):
        sp=string
        while True:
                try:
                        i=sp.index('ms')
                except:
                        raise ValueError(string+' no mse found')
                sp=sp[i+2:]
                n=subsequent_nums(sp)
                if len(n)>0:
                        return 'ms'+n
def subsequent_nums(string):
        try:
                seq=int(string[0])
                seq=str(seq)+subsequent_nums(string[1:])
        except:

                return ''

        return str(seq)
def round(m):
	tmp=m%1
	print('tmp: '+str(tmp))

	if tmp>=0.5:
		return int(m/1)+1
	else:
		return int(m/1)
def test_retest(i):
	if 'retest' in i:
		return 'retest'
	else:
		return 'test'
def Scanner(i):
	if 'GE' in i:
		return 'GE'
	elif 'PHILIPS' in i:
		return 'PHILIPS'
	elif 'SKYRA' in i:
		return 'SKYRA'
	else:
		raise ValueError('no scanner found')
def get_all_mse(msid):
	import pandas as pd
	from subprocess import Popen,PIPE
	cmd = ["ms_get_patient_imaging_exams", "--patient_id",msid]
	proc = Popen(cmd, stdout=PIPE)
	lines = [l.decode("utf-8").split() for l in proc.stdout.readlines()[5:]]
	tmp = pd.DataFrame(lines, columns=["mse", "date"])
	tmp["mse"] = "mse"+tmp.mse
	tmp["msid"] = msid
	#tmp["a0"] = par_returner(x,ch[0][0],ch[0][1], ch[0][2], ch[0][3], ch[0][4], ch[0][5])[1]
	return tmp
