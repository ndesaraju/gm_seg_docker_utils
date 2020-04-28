from glob import glob 
import xlrd 
import json
import os

workbook=xlrd.open_workbook('EPIC1_completed.xlsx')
sheet=workbook.sheet_by_index(0)
for i in range(1,sheet.nrows):
	try:
		print('here')
		putter=glob('/data/henry*/PBR/subjects/{}/alignment/status.json'.format(sheet.cell_value(i,0)))[0]
		if putter:
			data_json=json.load(putter)
			image_path=data_json['t1_files'][0]
			data_set.add(image_path)

	except:
		pass
data_list=list(data_set)
for i in data_list:
	os.symlink(i,'/data/henry4/tf-3dgan/data/'+os.path.basename(i))