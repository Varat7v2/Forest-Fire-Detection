import glob
import os, sys
import pandas as pd
from sklearn.utils import shuffle
import argparse

from tqdm import tqdm

def main(argv):
	parser = argparse.ArgumentParser(description="Online Image Downloader")
	parser.add_argument("--max-number", "-n", type=int, default=100,
	                    help="Max number of images download for the keywords.")
	args = parser.parse_args(args=argv)

	### Downloading images from Google / Baidu / Bing
	fire = ['forest-fire', 'aerial-forest-fire', 'brush-fire', 'forest-fire-smoke', 'aerial-forest-fire-smoke',' ground-fires', ' crown-fires', ' surface-fires']
	no_fire = ['forest', 'aerial-forest','Tropical-Deciduous-Forest', 'Equatorial-Moist-Evergreen-or-Rainforest', 'Mediterranean-Forest', 'Temperate-Broad-leaved-Deciduous-and-Mixed-Forest','Warm-Temperate-Broad-leaved-Deciduous-Forest','Coniferous-Forest']
	class_dict = {'fire':fire, 'no_fire':no_fire}
	engine = {"Google", "Bing"}

	print('Downloading Images for class=fire')
	for key,value in class_dict.items():
		for keyword in value:
			for search in engine:
				os.system('python image_downloader.py {} --engine {} --max-number 2 --output "data/forest-fire-detection/{}/{}"'
					.format(keyword, search, key, keyword))


	print("Creating CSV file of imagepaths ... ")
	### For creating csv file for all the imagepaths
	main_path = 'data/forest-fire-detection'

	root, dirs, files = next(os.walk(main_path))
	mylist = list()

	for d in dirs:
		subdir = next(os.walk(os.path.join(root,d)))[1]
		for s in subdir:
			for file in glob.glob(os.path.join(root, d, s) + '/*.jpeg'):
				mylist.append(file)

	# converting to dataframe shuffling
	df = shuffle(pd.DataFrame(mylist, columns=['filename']))
	df.reset_index(inplace=True, drop=True)

	# save to csv file
	df.to_csv('data/forest-fire-images.csv', index=False)

if __name__ == '__main__':
	main(sys.argv[1:])