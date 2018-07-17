import data_tf
import tensorflow as tf
import dataset


train_csv="/home/oole/Data/nice_data/train.csv"
labelEncoder = data_tf.labelencoder()

trainSlideData = data_tf.collect_data_csv(train_csv, data_tf.getlabel_new)
trainSlideData.setLabelEncoder(labelEncoder)

with tf.Session() as sess:
    batchSize=64

    slideDataset =dataset.img_dataset_augment(trainSlideData.getSlideList()[0], batch_size=batchSize,
                                                   shuffle_buffer_size=None, shuffle=False,
                                                   getlabel=trainSlideData.getLabelFunc())
    iterator = tf.data.Iterator.from_structure(slideDataset.output_types,
                                               slideDataset.output_shapes)

    iterator_ops = []
    for slide in trainSlideData.getSlideList():
        slideDataset = dataset.img_dataset_augment(slide, batch_size=batchSize,
                                                   shuffle_buffer_size=None, shuffle=False,
                                                   getlabel=trainSlideData.getLabelFunc())
        init_op = iterator.make_initializer(slideDataset)
        iterator_ops.append(init_op)

    next_op = iterator.get_next()
    for i in range(trainSlideData.getNumberOfSlides()):
        print("Slide %s of %s" % (str(i), str(trainSlideData.getNumberOfSlides())))
        sess.run(iterator_ops[i])

        while True:
            step = 0
            try:
                print("Step: %s "  %step)
                next = sess.run(next_op)
                step +=1
            except tf.errors.OutOfRangeError:
                print("End of slide dataset.")
                break


