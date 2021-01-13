#!/usr/bin/env python3

import bulk_process_GM_flywheel as bulk
import midnight_flywheel as mn 
import midnight_second_part_flywheel as mn2
import crop_zoom_to_roi_original_flywheel as crz 
import t_map_flywheel as tmf
from glob import glob
import nibabel as nb
import numpy as np
import sys
import logging as log

pth='/flywheel/v0/pth/'
pth2='/flywheel/v0/pth2/'

psir_path = sys.argv[1]
roi_path = sys.argv[2]
output_path = sys.argv[3]

def gray_matter_seg(psir,roi,outputs_path, cycle_size, subj, sess, protocol,prefix=0):
	cord,crop_aff,cordpth=crz.create_nifti_zoomed(psir, roi,outputs_path)
	log.info("created zoomed files")
	if bulk.run_this(cordpth,outputs_path,pth,pth2, cycle_size, subj, sess, protocol, prefix):
		if not mn.run_this(cordpth,outputs_path, cycle_size, subj, sess, protocol, prefix):
			if mn2.run_this(cordpth,outputs_path, prefix):
				tmf.run_this(cordpth,outputs_path, prefix)
			else:
				raise ValueError('rawr midnight 2 doesnt work')
		else:
			raise ValueError('rawr midnight doesnt work')
	else:
		raise ValueError('rawr bulk doesnt work')

gray_matter_seg(psir_path, roi_path, output_path)
