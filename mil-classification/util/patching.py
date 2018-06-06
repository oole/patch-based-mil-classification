# Copy input to output

from skimage import io
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import resize
from skimage import img_as_uint
from skimage.morphology import binary_erosion
import openslide as osl
import numpy as np
import os


# 1 all BG, 0 all FG
def get_bg_ratio(image):
    max_project = image[:,:,1]
    gf = gaussian(max_project, 5)
    th = threshold_otsu(gf)
    th_patch = img_as_uint(gf > th)
    for i in range(3):
        th_patch = binary_erosion(th_patch)
    th_nonzero = np.count_nonzero(th_patch)
    non_th_ratio = th_nonzero / th_patch.size
    # io.imshow(th_patch)
    return non_th_ratio#, th_patch


def patchisgood(patch):
    #filter out background
    if np.std(patch[:,:,1]) < 15:
        return False, "notcalc"
    #check amount of background
    bg_ratio = get_bg_ratio(patch)
    if bg_ratio < 0.65:
        return True, bg_ratio
    else:
        return False, bg_ratio

def normalize_img(img, scale=255, d_type=np.uint8):
    mini = np.min(img)
    maxi = np.max(img)
    return d_type(((img - mini) / (maxi - mini)) * scale)


def getPatches(slide, downsamplefactor, folder, fileEnding, patchsize):
    (x, y) = slide.dimensions
    slidesize = x * y
    thumb = slide.get_thumbnail((x // 100, y // 100))
    bg_ratio = get_bg_ratio(np.asarray(thumb))

    roi_size = bg_ratio * slidesize
    stepsize = patchsize
    # stepsize = int(np.sqrt(roi_size) // np.sqrt(1000) * downsamplefactor)
    # lvl = slide.get_best_level_for_downsample(magfactor);
    lvl = 0

    slide_dimensions = slide.level_dimensions

    scalingFactor = slide_dimensions[0][0] // slide_dimensions[lvl][0]
    # (width, height) = slide.dimensions
    width = slide_dimensions[lvl][0]
    height = slide_dimensions[lvl][1]
    # i = 0

    extractsize = patchsize * downsamplefactor
    stepsize = extractsize

    xcord = 0
    ycord = 0
    xmax = 0
    ymax = 0
    numpatches = 0
    for x in range(0, width, stepsize):
        for y in range(0, height, stepsize):
            # check border condition
            if (x + stepsize > width or y + stepsize > height):
                # print("Slice too small")
                continue

            patch = normalize_img(
                np.asarray(slide.read_region(level=lvl, location=(x, y), size=(extractsize, extractsize)))[
                :, :, 0:3])

            isgood, bg_ratio = patchisgood(patch)

            if (not isgood):
                io.imsave("/home/oole/patchingtest/discarded/" + "/patch_x-" + repr(xcord) + "_y-" + repr(ycord) + "_" + str(bg_ratio) + fileEnding,
                          resize(patch, (patchsize, patchsize)))
                ycord = ycord + 1
                continue

            io.imsave(folder + "/patch_x-" + repr(xcord) + "_y-" + repr(ycord) + "_" + str(bg_ratio) + fileEnding,
                      resize(patch, (patchsize, patchsize)))
            numpatches += 1
            ycord = ycord + 1
            if (ycord > ymax):
                ymax = ycord

        xcord = xcord + 1
        ycord = 0
    infofile = open(folder + "/info.txt", "w")
    infofile.write("x = " + repr(xcord) + "\n")
    infofile.write("y = " + repr(ymax) + "\n")
    infofile.write("size = " + repr(numpatches))
    infofile.close()
    return numpatches


# output_table = input_table.copy()

locations = []
locations.append("/media/oole/GAMING/Data/TCGA-LGG/e9d5e6cc-a09a-4d61-aab3-f2690b3b3949/TCGA-HT-7481-01A-01-TS1.f17cfae1-4694-4da5-9de0-4537cfc2a1c8.svs")
locations.append("/media/oole/GAMING/Data/TCGA-LGG/9321927c-3a41-45e8-b320-69c97ef85dcb/TCGA-HW-7490-01A-01-BS1.a32239ed-e49f-4bad-a134-9a94f1b067ed.svs")
names = []
names.append("TCGA-HT-7481-01A-01-TS1.f17cfae1-4694-4da5-9de0-4537cfc2a1c8.svs")
names.append("TCGA-HW-7490-01A-01-BS1.a32239ed-e49f-4bad-a134-9a94f1b067ed.svs")

classes = []
classes.append("OD")
classes.append("DA")

downsamplefactor = 40 // 20
i = 0
for loc in locations:
    cl = classes[i]
    slide = osl.open_slide(loc)
    sinklocation = "/home/oole/patchingtest" + "/" + names[i][:-4] + "_class-" + cl + "/"
    if not os.path.exists(sinklocation):
        os.makedirs(sinklocation)
    numpatches = getPatches(slide, downsamplefactor, sinklocation, ".jpg", 500)
    i += 1


