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
image_path =r"D:/PostDoc Project/CODES/10295869MIC/Image/"
label_path =r"D:/PostDoc Project/CODES/10295869MIC/Label/"

image_list = []

# Iterate over all the files in the directory
for file in os.listdir(image_path):
    file_path =f"{image_path}/{file}"
    ds = dicom.dcmread(file_path)
    image = ds.pixel_array
    image_list.append(image)
   
      
label_list = []  

for file in os.listdir(label_path):
    file_path =f"{label_path}/{file}"
    nii_img  = nib.load(file_path)
    nii_data = nii_img.get_fdata()
    label_list.append(nii_data)  
      

for i in range(0, len(image_list)):
    print(i, image_list[i].shape, label_list[i].shape)
    

'''for i in range(0, len(image_list)):
    for j in range(0, image_list[i].shape[0]):
        print(i,j)
        mri_image = image_list[i][j,:,:]
        mri_image = np.double(resize(mri_image, (256, 256)))
        mask_image = (label_list[i][:,:,j]).T
        t = 0.5
        binary_mask = mask_image > t
        binary_mask = np.double(resize(binary_mask, (256, 256)))
        plt.subplot(1, 2, 1)
        plt.imshow(mri_image,cmap='gray')
        plt.title('MRI')
        plt.subplot(1, 2, 2)
        plt.imshow(binary_mask,cmap='gray')
        plt.title('Mask')
        plt.show()'''
    

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
        
        
for i in range(0, len(MRI_image_copy)):
    plt.subplot(1, 2, 1)
    plt.imshow(MRI_image_copy[i],cmap='gray')
    plt.title('MRI')
    plt.subplot(1, 2, 2)
    plt.imshow(Seg_mask_copy[i],cmap='gray')
    plt.title('Mask')
    plt.show()
        

#data_train, data_test, data_mask_train, data_mask_test = train_test_split(MRI_image_copy, Seg_mask_copy, test_size=0.20, random_state=42)

#np.save('10295869MIC_Image.npy', MRI_image_copy)
#np.save('10295869MIC_Label.npy', Seg_mask_copy)



'''np.save('data_train.npy', data_train)
np.save('data_mask_train.npy', data_mask_train)
np.save('data_test.npy', data_test)
np.save('data_mask_test.npy', data_mask_test)'''


'''for i in range(0, len(data_train)):
    plt.subplot(1, 2, 1)
    plt.imshow(data_train[i],cmap='gray')
    plt.title('MRI')
    plt.subplot(1, 2, 2)
    plt.imshow(data_mask_train[i],cmap='gray')
    plt.title('Mask')
    plt.show()'''



'''MRI_image = []
Seg_mask = []

for i in range(0, len(image_list)):
    for j in range(0, image_list[i].shape[0]):
        print(i,j)
        mri_image = image_list[i][j,:,:]
        mri_image = np.double(resize(mri_image, (256, 256)))
        MRI_image.append(mri_image)
        mask_image = (label_list[i][:,:,j]).T
        t = 0.5
        binary_mask = mask_image > t
        binary_mask = np.double(resize(binary_mask, (256, 256)))
        Seg_mask.append(binary_mask)
        plt.subplot(1, 2, 1)
        plt.imshow(mri_image,cmap='gray')
        plt.title('MRI')
        plt.subplot(1, 2, 2)
        plt.imshow(binary_mask,cmap='gray')
        plt.title('Mask')
        plt.show()'''
    




'''A = np.zeros((len(image_list),image_list[0].shape[1],image_list[0].shape[2]))

for item in image_list:
    A = np.concatenate((A, item), axis = 0)

print(A.shape)'''


'''image_path = 'D:/PostDoc Project/Intact tendon/Intact tendon/437543MIC/IM-0001-11164.dcm'

ds = dicom.dcmread(image_path)
print(ds)
image = ds.pixel_array

nii_img  = nib.load('D:/PostDoc Project/Intact tendon/Intact tendon/437543MIC/IM-0001-11164.nii.gz')

#nii_img  = nib.load('D:/PostDoc Project/Shoulder_Data/Shoulder_Data/Jessica_s Labels/Intact/007/007-2.nii.gz')
nii_data = nii_img.get_fdata()


for i in range (0, image.shape[0]): 
    print(i)
    one_image = ds.pixel_array[i,:,:]
    plt.imshow(one_image,cmap='gray')
    plt.pause(0.01)
    plt.show()
    
    
for i in range (0, nii_data.shape[2]): 
    print(i)
    one_image = nii_data[:,:,i]
    t = 0.5
    threshold = filters.threshold_otsu(one_image)
    binary_image = one_image > t
    plt.imshow(binary_image.T,cmap='gray')
    plt.pause(0.01)
    plt.show()'''




