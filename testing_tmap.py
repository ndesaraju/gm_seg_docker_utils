from glob import glob
import t_map_flywheel as tmf
a=glob('final*out*/only*')
tmf.run_this(a[0],'/home/gm_seg_docker_utils')

