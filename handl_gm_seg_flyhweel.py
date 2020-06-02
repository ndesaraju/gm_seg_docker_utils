import bulk_process_GM_flywheel as bulk
import midnight_flywheel as mn 
import midnight_second_part_flywheel as mn2
import crop_zoom_to_roi_original_flywheel as crz 
import t_map_flywheel as tmf
from glob import glob
import nibabel as nb
import numpy as np
pth='/flywheel/v0/input/pth/'
pth2='/flywheel/v0/input/pth2/'
def gray_matter_seg(psir,roi,outputs_path, prefix=0):
	cord,crop_aff,cordpth=crz.create_zoomed_files(psir, roi,outputs_path)
	if bulk.run_this(cordpth,outputs_path,pth,pth2, prefix):
		if not mn.run_this(cordpth,outputs_path, prefix):
			if mn2.run_this(cordpth,outputs_path, prefix):
				tmf.run_this(cordpth,outputs_path, prefix)
			else:
				raise ValueError('rawr midnight 2 doesnt work')

		# else:
		# 	raise ValueError('rawr midnight doesnt work')
	else:
		raise ValueError('rawr bulk doesnt work')
def gray_matter_seg_no_preproc(psir,outputs_path,prefix=0):
	#input(mn.run_this(psir,outputs_path,prefix))
	if bulk.run_this(psir,outputs_path,pth,pth2,prefix):
		if not mn.run_this(psir,outputs_path,prefix):
			if mn2.run_this(psir,outputs_path,prefix):
				tmf.run_this(psir,outputs_path,prefix)
			else:
				raise ValueError('rawr midnight 2 doesnt work')

		else:
			raise ValueError('rawr midnight doesnt work')
	else:
		raise ValueError('rawr bulk doesnt work')
if __name__== '__main__':
	sups=glob('/data/henry4/jjuwono/new_GM_method/ms*PSIR*/raw_im.nii.gz')
#	for s in sups:
#		output=s.split('/')[5]
#		tmp_handl=nb.load(s)
#		tmp=tmp_handl.get_data()
#		tmp_q=tmp.min()
#		if tmp_q<1400:
#			tmp=tmp+(1400-tmp_q)
#			tmp=np.where(tmp!=(1400-tmp_q),tmp,0)
#			nb.save(nb.Nifti1Image(tmp,tmp_handl.affine),s.split('.nii.gz')[0]+'transform.nii.gz') 
#			s=s.split('.nii.gz')[0]+'transform.nii.gz'
#		gray_matter_seg_no_preproc(s,'/data/henry4/jjuwono/new_GM_method_flywheel/{}'.format(output))
	gray_matter_seg('/data/henry11/PBR/subjects/mse8473/nii/ms500-mse8473-025-C2_3_2Fl_seg_psir_TI_PSIR.nii.gz','/data/henry11/PBR/subjects/mse8473/nii/ms500-mse8473-025-C2_3_2Fl_seg_psir_TI_PSIR_TCA.roi','/data/henry11/PBR/subjects/mse8473/nii/spinal_gm_seg')
