#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./DAGM2007/training/images/"
groundTruth_imgs_train = "./DAGM2007/training/label/"

#original_imgs_train = "./test/MIL/img/"
#groundTruth_imgs_train = "./test/MIL/label/"

# borderMasks_imgs_train = "./DRIVE/training/mask/"
#test
original_imgs_test = "./DAGM2007/test/images/"
groundTruth_imgs_test = "./DAGM2007/test/label/"

# original_imgs_test = "./DAGM2007/test/img/"
# groundTruth_imgs_test = "./DAGM2007/test/lab/"


#original_imgs_test = "./test/MIL/img/"
#groundTruth_imgs_test = "./test/MIL/label/"

# borderMasks_imgs_test = "./DRIVE/test/mask/"
#---------------------------------------------------------------------------------------------

NimgsTrain = 446
NimgsTest = 454
channels = 3
height = 512
width = 512
dataset_path = "./DAGM2007hdf5/"

# def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
def get_datasets(Nimgs,imgs_dir,groundTruth_dir,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    # border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print(("original image: " +files[i]))
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            # groundTruth_name = files[i][0:2] + "_manual1.gif"
            groundTruth_name = files[i]
            print(("ground truth name: " + groundTruth_name))
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            #corresponding border masks
            # border_masks_name = ""
            # if train_test=="train":
            #     border_masks_name = files[i][0:2] + "_training_mask.gif"
            # elif train_test=="test":
            #     border_masks_name = files[i][0:2] + "_test_mask.gif"
            # else:
            #     print("specify if train or test!!")
            #     exit()
            # print(("border masks name: " + border_masks_name))
            # b_mask = Image.open(borderMasks_dir + border_masks_name)
            # border_masks[i] = np.asarray(b_mask)

    print(("imgs max: " +str(np.max(imgs))))
    print(("imgs min: " +str(np.min(imgs))))
    # assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    # assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    # print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    assert(np.max(groundTruth)==255)
    assert(np.min(groundTruth)==0)
    print("ground truth is correctly withih pixel value range 0-255 (black-white)")

    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    # border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    # assert(border_masks.shape == (Nimgs,1,height,width))
    # return imgs, groundTruth, border_masks
    return imgs,groundTruth

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
# imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
imgs_train, groundTruth_train = get_datasets(NimgsTrain,original_imgs_train,groundTruth_imgs_train,"train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DAGM_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DAGM_dataset_groundTruth_train.hdf5")
# write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
# imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
imgs_test, groundTruth_test = get_datasets(NimgsTest,original_imgs_test,groundTruth_imgs_test,"test")
print("saving test datasets")
write_hdf5(imgs_test,dataset_path + "DAGM_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DAGM_dataset_groundTruth_test.hdf5")
# write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")
