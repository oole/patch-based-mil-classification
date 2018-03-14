from skimage import io
import os

def convert_tif_to_jpg(sourcepath, sinkpath):
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

convert_tif_to_jpg("/home/oole/Documents/UKN/BP/validation/patient", "/home/oole/Documents/UKN/BP/val_jpg_test/patient")