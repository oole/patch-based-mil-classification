import data_tf
import dataset
import numpy as np
import train
import train_em
import util


batch_size=64
def train_augment():
    # simple_train_savepath = "/home/oole/tfnetsave/tfnet_full"
    simple_train_savepath = "/home/oole/tfnetsave/tfnet_em_full"
    em_train_savepath = "/home/oole/tfnetsave/tfnet_em_full"

    initial_epoch = 0

    train_datapath = "/home/oole/Data/training/patient_patches_jpg"
    # train_datapath = '/home/oole/tf_test_data/validation'
    val_datapath = "/home/oole/Data/validation/patient_patches_jpg"

    logfile_path = "/home/oole/tfnetsave/tfnet_em_full_log.csv"
    logreg_savepath = "/home/oole/tfnetsave/tfnet_em_full_logreg"


    model_name = "model"

    label_encoder = data_tf.labelencoder()

    train_slidelist, train_slide_dimensions, old_disc_patches, _ = data_tf.collect_data(train_datapath, batch_size)
    val_slidelist, _, _, _ = data_tf.collect_data(val_datapath, batch_size)

    train_patches = dataset.slidelist_to_patchlist(train_slidelist)
    val_patches = dataset.slidelist_to_patchlist(val_slidelist)
    np.random.shuffle(train_patches)
    np.random.shuffle(val_patches)

    # Initial training
    #train_accuracy, val_accuracy = train.train_net(train_patches, val_patches, num_epochs=2, batch_size=batch_size,
    #                                               savepath=simple_train_savepath, do_augment=True, model_name=model_name)

    #util.write_log_file(logfile_path, train_accuracy=train_accuracy, val_accuracy=val_accuracy)

    # Test continue training
    # train.train_net(train_patches, val_patches, num_epochs=2, batch_size=batch_size, savepath=simple_train_savepath,
    #                 loadpath=simple_train_savepath, do_augment=False, model_name="model")

    train_em.emtrain(train_datapath, val_datapath,
                     simple_train_savepath, em_train_savepath,
                     label_encoder, batch_size,
                     initial_epochnum=initial_epoch,
                     model_name=model_name,
                     spatial_smoothing=False,
                     do_augment=True,
                     num_epochs=2, dropout_ratio=0.5, learning_rate=0.0005, sanity_check=False,
                     logfile_path=logfile_path, logreg_savepath=logreg_savepath)


def train_premod():
    # simple_train_savepath = "/home/oole/tfnetsave/tfnet_full_premod"
    simple_train_savepath = "/home/oole/tfnetsave/tfnet_em_full_premod"
    em_train_savepath = "/home/oole/tfnetsave/tfnet_em_full_premod"

    logfile_path = "/home/oole/tfnetsave/tfnet_em_full_premod_log.csv"
    logreg_savepath = "/home/oole/tfnetsave/tfnet_em_full_premod_logreg"

    initial_epoch = 8

    train_datapath = "/home/oole/Data/training/patient_patches_premod_jpg"
    # train_datapath = '/home/oole/tf_test_data/validation'
    val_datapath = "/home/oole/Data/training/patient_patches_premod_jpg"

    model_name = "model"


    label_encoder = data_tf.labelencoder()

    train_slidelist, train_slide_dimensions, old_disc_patches, _ = data_tf.collect_data(train_datapath, batch_size)
    val_slidelist, _, _, _ = data_tf.collect_data(val_datapath, batch_size)

    train_patches = dataset.slidelist_to_patchlist(train_slidelist)
    val_patches = dataset.slidelist_to_patchlist(val_slidelist)

    np.random.shuffle(train_patches)
    np.random.shuffle(val_patches)

    # Initial training
    train_accuracy, val_accuracy = train.train_net(train_patches, val_patches, num_epochs=2, batch_size=batch_size, savepath=simple_train_savepath,
                    do_augment=False, model_name=model_name)

    util.write_log_file(logfile_path, trainAccuracy=train_accuracy, valAccuracy=val_accuracy)

    # Test continue training
    # train.train_net(train_patches, val_patches, num_epochs=2, batch_size=batch_size, savepath=simple_train_savepath,
    #                 loadpath=simple_train_savepath, do_augment=False, model_name="model")

    train_em.emtrain(train_datapath, val_datapath,
                     simple_train_savepath, em_train_savepath,
                     label_encoder, batch_size,
                     initial_epochnum=initial_epoch,
                     model_name=model_name,
                     spatial_smoothing=False,
                     do_augment=False,
                     num_epochs=2, dropout_ratio=0.5, learning_rate=0.0005, sanity_check=False,
                     logfile_path=logfile_path, logreg_savepath=logreg_savepath)

def train_normal():
    simple_train_savepath = "/home/oole/tf_normal_save/tfnet_full"

    initial_epoch = 0

    train_datapath = "/home/oole/Data/training/patient_patches_jpg"
    # train_datapath = '/home/oole/tf_test_data/validation'
    val_datapath = "/home/oole/Data/validation/patient_patches_jpg"

    logfile_path = "/home/oole/tfnetsave/tfnet_em_full_log.csv"
    logreg_savepath = "/home/oole/tfnetsave/tfnet_em_full_logreg"

    model_name = "model"


    train_slidelist, train_slide_dimensions, old_disc_patches, _ = data_tf.collect_data(train_datapath, batch_size)
    val_slidelist, _, _, _ = data_tf.collect_data(val_datapath, batch_size)

    train_patches = dataset.slidelist_to_patchlist(train_slidelist)
    val_patches = dataset.slidelist_to_patchlist(val_slidelist)
    np.random.shuffle(train_patches)
    np.random.shuffle(val_patches)

    train_accuracy, val_accuracy = train.train_net(train_patches, val_patches, num_epochs=20, batch_size=batch_size,
                                                  savepath=simple_train_savepath, do_augment=True, model_name=model_name)


SPATIALSMOOTHING = False
DOAUGMENT = True
EPOCHS=2
DROPOUT=0.5
LR=0.0005
SANITYCHECK=False

def train_augment_csv(train_csv="/home/oole/Data/nice_data/train.csv", test_csv="/home/oole/Data/nice_data/test.csv"):
    val_csv = "/home/oole/Data/nice_data/400x400_data/train.csv"

    simple_train_savepath = "/home/oole/tfnetsave/tfnet_em_full"
    em_train_savepath = "/home/oole/tfnetsave/tfnet_em_full"

    initial_epoch = 0


    logfile_path = "/home/oole/tfnetsave/tfnet_log.csv"
    logreg_savepath = "/home/oole/tfnetsave/tfnet_logreg"


    model_name = "model"
    runName = "testrun2/"

    labelEncoder = data_tf.labelencoder()

    trainSlideData = data_tf.collect_data_csv(train_csv)
    trainSlideData.setLabelEncoder(labelEncoder)

    valSlideData = data_tf.collect_data_csv(val_csv)
    valSlideData.setLabelEncoder(labelEncoder)

    #test purposes
    trainSlideData, valSlideData = data_tf.getTestSizeData(trainSlideData, valSlideData, 20)
    # Initial training
    train_accuracy, val_accuracy, netAcc = train.train_net(trainSlideData, valSlideData, num_epochs=2, batch_size=batch_size,
                                                  savepath=simple_train_savepath, do_augment=True, model_name=model_name,
                                                   getlabel_train=data_tf.getlabel_new, log_savepath=logfile_path, runName=runName)
    print("Done")

    # util.write_log_file(logfile_path, train_accuracy=train_accuracy, val_accuracy=val_accuracy)

    # Test continue training
    # train.train_net(trainSlideData, valSlideData, num_epochs=2, batch_size=batch_size, savepath=simple_train_savepath,
    #                 loadpath=simple_train_savepath, do_augment=False, model_name="model")

    train_em.emtrain(trainSlideData, valSlideData,
                     simple_train_savepath, em_train_savepath, batch_size,
                     initial_epochnum=initial_epoch,
                     model_name=model_name,
                     spatial_smoothing=SPATIALSMOOTHING,
                     do_augment=DOAUGMENT,
                     num_epochs=EPOCHS, dropout_ratio=DROPOUT, learning_rate=LR, sanity_check=SANITYCHECK,
                     logfile_path=logfile_path, logreg_savepath=logreg_savepath, runName=runName, netAcc=netAcc)

train_augment_csv()