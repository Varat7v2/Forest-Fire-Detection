import pandas as pd
import shutil
import os, sys
from sklearn.utils import shuffle

from tqdm import tqdm

df = pd.read_csv('data/forest-fire-images.csv')

# for two classes create two empty list
class1, class2 = list(), list()
for index, row in df.iterrows():
    myclass = row['filename'].split('/')[-3]

    if myclass == 'fire':
    	# if not os.path.exists(myclass):
    	#     os.makedirs(myclass)
    	class1.append(row)
    elif myclass == 'no_fire':
    	class2.append(row)

# # convert to dataframe and shuffle the elements
df1 = shuffle(pd.DataFrame(class1))
df1.reset_index(inplace=True, drop=True)

df2 = shuffle(pd.DataFrame(class2))
df2.reset_index(inplace=True, drop=True) 

# # # save to respective csv files
# df1.to_csv('fire.csv', index=False)
# df2.to_csv('no_fire.csv', index=False)
# # print('saved to csv files')


class1_size = df1.shape[0]
class2_size = df2.shape[0]

train_data = pd.concat([df1.iloc[0:int(0.7*class1_size)], 
						df2.iloc[0:int(0.7*class2_size)]], 
						ignore_index=True)
valid_data = pd.concat([df1.iloc[int(0.7*class1_size)+1:int(0.8*class1_size)], 
						df2.iloc[int(0.7*class2_size)+1:int(0.8*class2_size)]], 
						ignore_index=True)
test_data =  pd.concat([df1.iloc[int(0.8*class1_size)+1:class1_size], 
						df2.iloc[int(0.8*class2_size)+1:class2_size]], 
						ignore_index=True)

# df_train = pd.DataFrame(train_data)
# df_val = pd.DataFrame(val_data)
# df_test = pd.DataFrame(test_data)

# # save to csv files
# df_train.to_csv('train.csv', index=False)
# df_val.to_csv('val.csv', index=False)
# df_test.to_csv('test.csv', index=False)

# Copy images from source to destination folders
folders = ['Train', 'Valid', 'Test']
dest_dir = 'data/training'
datasets = {'Train':train_data['filename'],'Valid':valid_data['filename'], 'Test':test_data['filename']}

# creating destination dataset directory
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for folder in folders:
	if not os.path.exists(os.path.join(dest_dir, folder)):
	    os.makedirs(os.path.join(dest_dir, folder))
	for key, value in datasets.items():
		if key == folder:
			count = 0
			for image_path in tqdm(value):
				classes = image_path.split('/')[-3]
				if classes == 'fire':
					if not os.path.exists(os.path.join(dest_dir, folder, classes)):
					    os.makedirs(os.path.join(dest_dir, folder, classes))
					shutil.copy2(image_path, os.path.join(dest_dir, folder,classes)+'/'+str(count)+'_{}.jpg'.format(folder))
				if classes == 'no_fire':
					if not os.path.exists(os.path.join(dest_dir, folder, classes)):
					    os.makedirs(os.path.join(dest_dir, folder, classes))
					shutil.copy2(image_path, os.path.join(dest_dir, folder,classes)+'/'+str(count)+'_{}.jpg'.format(folder))
				count += 1

print("All the images copied to allocaed folders successfully.")