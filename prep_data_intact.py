# Loading Dicom files Shoulder Data

import numpy as np
import nibabel as nib
import pydicom as dicom
import matplotlib.pyplot as plt
from skimage import filters
from skimage.transform import resize
import glob
import os
import cv2
from sklearn.model_selection import train_test_split


# Define the location of the directory
image_path =r"D:/PostDoc Project/DATA/Shoulder_Data/Images/"
label_path =r"D:/PostDoc Project/DATA/Shoulder_Data/Labels_bursa/"

# Change the directory
os.chdir(image_path)
image_list = []

# Iterate over all the files in the directory
for file in os.listdir():
   if file.endswith('.dcm'):
      # Create the filepath of particular file
      file_path =f"{image_path}/{file}"
      ds = dicom.dcmread(file_path)
      image = ds.pixel_array
      #image = np.double(resize(image, (256, 256)))
      image_list.append(image)
      

os.chdir(label_path)
label_list = []  

for file in os.listdir():
   if file.endswith('.gz'):
      # Create the filepath of particular file
      file_path =f"{label_path}/{file}"
      nii_img  = nib.load(file_path)
      nii_data = nii_img.get_fdata()
      #nii_data = np.double(resize(nii_data, (256, 256)))
      label_list.append(nii_data)  
      

for i in range(0, len(image_list)):
    print(i, image_list[i].shape, label_list[i].shape)
    

MRI_image = []
Seg_mask = []

for i in range(0, len(image_list)):
    for j in range(0, image_list[i].shape[0]):
        print(i,j)
        mri_image = image_list[i][j,:,:]
        mri_image = resize(mri_image, (256, 256))
        MRI_image.append(mri_image)
        mask_image = (label_list[i][:,:,j]).T
        t = 0.5
        binary_mask = mask_image > t
        binary_mask = resize(binary_mask, (256, 256))
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
        

data_train, data_test, data_mask_train, data_mask_test = train_test_split(MRI_image_copy, Seg_mask_copy, test_size=0.20, random_state=42)


#np.save('data_train_b.npy', data_train)
#np.save('data_mask_train_b.npy', data_mask_train)
#np.save('data_test_b.npy', data_test)
#np.save('data_mask_test_b.npy', data_mask_test)







    










