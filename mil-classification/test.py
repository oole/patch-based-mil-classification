import tensorflow as tf
import data_tf
import dataset
import netutil
import validate

def test_existing_net(slide_datapath, net_loadpath, model_name, dropout_ratio, batch_size, do_augment=False, shuffle_buffer_size=2000):

    slide_list, slide_dimensions, num_patches, slide_label = data_tf.collect_data(
        slide_datapath, batch_size)

    patches = dataset.slidelist_to_patchlist(slide_list)

    with tf.Session() as sess:
        if do_augment:
            input_dataset = dataset.img_dataset_augment(patches, batch_size=batch_size,
                                                        shuffle_buffer_size=shuffle_buffer_size, shuffle=True)
        else:
            input_dataset = dataset.img_dataset(patches, batch_size=batch_size,
                                                shuffle_buffer_size=shuffle_buffer_size, shuffle=True)

        input_iterator = input_dataset.make_initializable_iterator()

        iterator_handle = sess.run(input_iterator.string_handle())

        proxy_iterator_handle_ph = tf.placeholder(tf.string, shape=[])
        proxy_iterator = tf.data.Iterator.from_string_handle(proxy_iterator_handle_ph,
                                                             output_types=input_iterator.output_types,
                                                             output_shapes=input_iterator.output_shapes)

        x, y = proxy_iterator.get_next()

        train_op, loss_op, y, accuracy_op, x, keep_prob_ph, learning_rate_ph, is_training_ph, y_pred_op, y_argmax_op = \
            netutil.build_model(model_name, x, y, use_bn_1=True, use_bn_2=True, use_dropout_1=True, use_dropout_2=True)

        # model saver
        saver = tf.train.Saver()
        ########################
        # load model from disc
        saver.restore(sess, net_loadpath)



        validate.validate_existing_net(iterator_handle, input_iterator, num_patches,
                                  dropout_ratio, batch_size,
                                  loss_op, accuracy_op,
                                  keep_prob_ph, is_training_ph, proxy_iterator_handle_ph, sess)



# net_loadpath = "/home/oole/tfnetsave/tfnet_em_full_premod"
# slide_datapath = "/home/oole/Data/training/patient_patches_premod_jpg"
# model_name="model"
# dropout_ratio=0.5
# batch_size=64
# shuffle_buffer_size=2000
#
# test_existing_net(slide_datapath, net_loadpath, model_name, dropout_ratio, batch_size, do_augment=False)