from skimage import io
from shutil import copyfile
import os

def convert_tif_to_jpg_cnn_test(sourcepath, sinkpath):
    print("Converting tifs to jpgs")
    print("Source %s" % sourcepath)
    print("Sink %s" % sinkpath)
    folders = os.listdir(sourcepath)

    i = 1
    for folder in folders:
        print("Folder %0.d of %0.d" % (i,len(folders)))
        sourcefolderpath = sourcepath + "/" + folder
        images = os.listdir(sourcefolderpath)
        sinkfolderpath = sinkpath + "/" + folder
        if not os.path.exists(sinkfolderpath):
            os.makedirs(sinkfolderpath)
        for img in images:
            sourceimgpath = sourcefolderpath + "/" + img
            imread = io.imread(sourceimgpath)
            sinkimgpath = sinkfolderpath + "/" + img
            io.imsave(sinkimgpath + ".jpg", imread, quality=100)
        i +=1

def convert_tif_to_jpg_training(sourcepath, sinkpath):
    print("Converting tifs to jpgs")
    print("Source %s" % sourcepath)
    print("Sink %s" % sinkpath)
    folders = os.listdir(sourcepath)

    i = 1
    for folder in folders:
        print("Folder %0.d of %0.d" % (i, len(folders)))
        sourcefolder_patientpath = sourcepath + "/" + folder
        patient_slides = os.listdir(sourcefolder_patientpath)
        for patient_slide in patient_slides:
            sourcefolderpath = sourcefolder_patientpath + "/" + patient_slide
            images = os.listdir(sourcefolderpath)
            sinkfolderpath = sinkpath + "/" + folder + "/" + patient_slide
            if not os.path.exists(sinkfolderpath):
                os.makedirs(sinkfolderpath)
            for img in images:
                if img == "info.txt":
                    copyfile(sourcefolderpath + "/" + img, sinkfolderpath + "/" + img)
                    continue
                sourceimgpath = sourcefolderpath + "/" + img
                imread = io.imread(sourceimgpath)
                sinkimgpath = sinkfolderpath + "/" + img
                io.imsave(sinkimgpath + ".jpg", imread, quality=100)
        i += 1

convert_tif_to_jpg_training("/media/oole/GAMING/testing_Data/Production/data_training/patients_patches", "/home/oole/Data/training/patient_patches_jpg")
print("Done!")