# Loading Dicom files Shoulder Data

import numpy as np
import nibabel as nib
import pydicom as dicom
import matplotlib.pyplot as plt
from skimage import filters
from skimage.transform import resize
import glob, time
import os
import cv2
from sklearn.model_selection import train_test_split

start = time.time()

#stdoutOrigin=sys.stdout 
#sys.stdout = open("new_data_info.txt", "w")

# Define the location of the directory

dir_path =r"D:/PostDoc Project/CLEAN_Torn_with_labels/"


dir_list = os.listdir(dir_path)

image_list = []
label_list = []

MRI_image = []
Seg_mask = [] 


for i in range(0, len(dir_list)):
    image_path = dir_path + str(dir_list[i]) + "/Image/"
    file_list = os.listdir(image_path)
    ds = dicom.dcmread(image_path + file_list[0])
    image = ds.pixel_array
    print(i, dir_list[i], image.shape)
    for j in range(0, image.shape[0]):
        print(i,j)
        mri_image = image[j,:,:]
        mri_image = resize(mri_image, (256, 256))
        MRI_image.append(mri_image)    
        
print("End of Images")       
    
for i in range(0, len(dir_list)):    
    label_path = dir_path + str(dir_list[i]) + "/Label_bursa/"
    label_file_list = os.listdir(label_path)
    nii_img  = nib.load(label_path + label_file_list[0])
    nii_data = nii_img.get_fdata()
    print(i, dir_list[i], nii_data.shape)
    for j in range(0, nii_data.shape[2]):
        mask_image = (nii_data[:,:,j]).T
        t = 0.5
        binary_mask = mask_image > t
        binary_mask = resize(binary_mask, (256, 256))
        print(j)
        Seg_mask.append(binary_mask)
 

MRI_image_copy = []
Seg_mask_copy = []

count = 0

for i in range(0, len(Seg_mask)):
    if len(np.unique(Seg_mask[i])) > 1:
        Seg_mask_copy.append(Seg_mask[i])
        MRI_image_copy.append(MRI_image[i])
    elif len(np.unique(Seg_mask[i])) == 1:
        count = count + 1
        
        

data_train, data_test, data_mask_train, data_mask_test = train_test_split(MRI_image_copy, Seg_mask_copy, test_size=0.05, random_state=42)

#np.save('data_train_torn_b.npy', data_train)
#np.save('data_mask_train_torn_b.npy', data_mask_train)
#np.save('data_test_torn_b.npy', data_test)
#np.save('data_mask_test_torn_b.npy', data_mask_test)




