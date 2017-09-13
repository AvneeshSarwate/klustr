import os
import shutil


targetDir = "./target"

fileNames = []

for directory in os.walk("./sampDir0"):
	for file in directory[2]:
		oldPath = directory[0]+"/"+file
		newPath = targetDir+"/"+file
		print oldPath, newPath
		#shutil.copyfile(oldPath, newPath)
		fileNames.append(oldPath)

allPaths = open("allPaths.txt", 'w')
allPaths.write("\n".join(fileNames))
allPaths.write("\n")


