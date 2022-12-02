import argparse
import os

CUDA_VISIBLE_DEVICES = "0,1"
PYTHON_INTERPRETER = "/miniconda3/envs/satellogic/bin/python"

OUT_PATH = "/iqf/??"
TRAIN_DS = "/iqf/??"

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	
	# These are modified by IQF
	#"python {} --trainds {} --valds {} --outputpath {} {}"
	#"python {} --trainds {} --outputpath {} {}"
	parser.add_argument('--trainds', type=str, default=TRAIN_DS, help='input dataset path')
	parser.add_argument('--outputpath', type=str, default=OUT_PATH, help='input dataset path')
	
	# These are fixed for now
	parser.add_argument('--cu', type=str, default=CUDA_VISIBLE_DEVICES, help='CUDA_VISIBLE_DEVICES')
	parser.add_argument('--py', type=str, default=PYTHON_INTERPRETER, help='Full python path')
	
	opt = parser.parse_args()
	
	cuda_vis_dev = opt.cu
	python_path = opt.py
	trainds = opt.trainds
	outputpath = opt.outputpath
	
	cmd =  f"CUDA_VISIBLE_DEVICES={cuda_vis_dev} && {python_path} train.py "
	cmd += f"--trainds {trainds} --outputpath {outputpath}"
	
	with open('./wrapper.log','w') as f:
		f.write(cmd)
	
	os.system(cmd)
