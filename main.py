from __future__ import print_function
from network import GlimpsNetwork, LocationNetwork
from tensorflow import keras
#from tensorflow.python.keras.datasets import cifar
from tensorflow.contrib import distributions
from seq2seq import rnn_decoder
from config import *
import sys, os, time
from datagenerator import ImageDataGenerator
from datetime import datetime
#from tensorflow.contrib.data import Iterator


import tensorlayer as tl
import tensorflow as tf
import numpy as np

# List object to record coordinate
origin_coor_list = []
sample_coor_list = []

# Network object
location_network = None
glimps_network = None

# Path to the textfiles for the trainings and validation set
train_file = '/home/dusa/tools/train.txt'
val_file = '/home/dusa/tools/val.txt'

# params
dropout_rate = 0.5
num_epochs = 200
batch_size = 64

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/tensorboard"
checkpoint_path = "/tmp/checkpoints"


def getNextRetina(output, i):
    global origin_coor_list
    global sample_coor_list
    global location_network
    global glimps_network    
    sample_coor, origin_coor = location_network(output)
    origin_coor_list.append(origin_coor)
    sample_coor_list.append(sample_coor)
    return glimps_network(sample_coor)

def loglikelihood(mean_arr, sampled_arr, sigma):
    mu = tf.stack(mean_arr)                     # mu = [timesteps, batch_sz, loc_dim]
    sampled = tf.stack(sampled_arr)             # same shape as mu
    gaussian = distributions.Normal(mu, sigma)
    logll = gaussian.log_prob(sampled)          # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    logll = tf.transpose(logll)                 # [batch_sz, timesteps]
    return logll

if __name__ == '__main__':
    # Create placeholder
    images_ph = tf.placeholder(tf.float32, [None, 64 , 64 , 3])
    labels_ph = tf.placeholder(tf.int64, [None])

    # Create network
    glimps_network = GlimpsNetwork(images_ph)
    location_network = LocationNetwork()
    
    # Construct Glimps network (part in core network)
    init_location = tf.random_uniform((tf.shape(images_ph)[0], 2), minval=-1.0, maxval=1.0)
    print(init_location)
    init_glimps_tensor = glimps_network(init_location)
    #print(init_glimps_tensor)

    # Construct core network
    lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
    init_lstm_state = lstm_cell.zero_state(tf.shape(images_ph)[0], tf.float32)
    input_glimps_tensor_list = [init_glimps_tensor]
    input_glimps_tensor_list.append([0] * num_glimpses)
    outputs, _ = rnn_decoder(input_glimps_tensor_list, init_lstm_state, lstm_cell, loop_function=getNextRetina)

    # Construct the classification network
    action_net = tl.layers.InputLayer(outputs[-1])
    action_net = tl.layers.DenseLayer(action_net, n_units = num_classes, name='classification_net_fc')
    logits = action_net.outputs
    #print(logits)
    #print(labels_ph)
    softmax = tf.nn.softmax(logits)

    # Cross-entropy
    entropy_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
    entropy_value = tf.reduce_mean(entropy_value)
    predict_label = tf.argmax(logits, 1)

    # Reward
    reward = tf.cast(tf.equal(predict_label, labels_ph), tf.float32)
    rewards = tf.expand_dims(reward, 1)
    rewards = tf.tile(rewards, (1, num_glimpses))
    _log = loglikelihood(origin_coor_list, sample_coor_list, loc_std)
    _log_ratio = tf.reduce_mean(_log)
    reward = tf.reduce_mean(reward)

    # Hybric locc
    loss = -_log_ratio + entropy_value
    var_list = tf.trainable_variables()
    grads = tf.gradients(loss, var_list)

    # Optimizer
    opt = tf.train.AdamOptimizer(0.000001)
    global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
    train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)
    #train_op =opt.minimize(loss, global_step=global_step)

    #print("opt.get_name(): ",opt.get_name(),"opt._lr: ",opt._lr,"opt._lr_t: ",opt._lr_t)

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(train_file,
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     shuffle=True)
        #print(tr_data)
        val_data = ImageDataGenerator(val_file,
                                      mode='inference',
                                      batch_size=batch_size,
                                      num_classes=num_classes,
                                      shuffle=False)

        # create an reinitializable iterator given the dataset structure
        iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)

    # TF placeholder for graph input and output
    #x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    #y = tf.placeholder(tf.float32, [batch_size, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_label, labels_ph), tf.float32))


    print(predict_label)
    print(labels_ph)

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter 
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
    val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

    # Train
    with tf.Session() as sess:
        #(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        #print(x_train.shape)
        sess.run(tf.global_variables_initializer())

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            # Initialize iterator with the training dataset
            sess.run(training_init_op)
            print(sess)

            for step in range(train_batches_per_epoch):

                # get next batch of data
                img_batch, label_batch = sess.run(next_batch)
                #print(img_batch.shape)
                #print(label_batch.shape)

                images = np.tile(img_batch,[M,1,1,1])
                labels = np.tile(label_batch, [M])
            
   
                print(images.shape)
                print(labels)
                # And run the training op
                _loss_value, _reward_value, _ = sess.run([loss, reward, train_op], feed_dict={images_ph: images,
                                              labels_ph: labels,
                                              keep_prob: dropout_rate})
                if step % 10 == 0:
                    print('iter: ', step, '\tloss: ', _loss_value, '\treward: ', _reward_value)



        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            
            images = np.tile(img_batch, [M,1,1,3])
            labels = np.tile(label_batch, [M])
            
            #print(images.shape)
            #print(labels.shape)
            acc = sess.run([accuracy], feed_dict={images_ph: images,
                                                labels_ph: labels,
                                                keep_prob: 1.})
            
            print(predict_label)
            print(labels_ph)
            print(accuracy)
            print(acc)
            test_acc += float(acc[0])
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))

        
     
