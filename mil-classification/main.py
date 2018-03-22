import data_tf
import dataset
import numpy as np
import train

batch_size=64

simple_train_savepath = "/home/oole/tfnetsave/tfnet_full"
train_datapath = "/home/oole/Data/training/patient_patches_jpg"
# train_datapath = '/home/oole/tf_test_data/validation'
val_datapath = "/home/oole/tf_test_data/validation"

model_name="model"

train_slidelist, train_slide_dimensions, old_disc_patches, _ = data_tf.collect_data(train_datapath, batch_size)
val_slidelist, _, _, _ = data_tf.collect_data(val_datapath, batch_size)

train_patches = dataset.slidelist_to_patchlist(train_slidelist)
val_patches = dataset.slidelist_to_patchlist(val_slidelist)

np.random.shuffle(train_patches)
np.random.shuffle(val_patches)

train.train_net(train_patches, val_patches, num_epochs=2, batch_size=batch_size, savepath=simple_train_savepath,
                do_augment=True, model_name=model_name)

# train.train_net(train_patches, val_patches, num_epochs=2, batch_size=batch_size, savepath=simple_train_savepath,
#                 loadpath=simple_train_savepath, do_augment=False, model_name="model")

