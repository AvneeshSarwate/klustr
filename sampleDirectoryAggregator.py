import os
import shutil


targetDir = "./target"

for directory in os.walk("./sampDir0"):
	for file in directory[2]:
		print directory[0]+"/"+file, targetDir+"/"+file
		shutil.copyfile(directory[0]+"/"+file, targetDir+"/"+file)

