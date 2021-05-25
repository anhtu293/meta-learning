import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from src.load_data import DataGenerator
from src.models import loss_function, MANN

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')
flags.DEFINE_string('data_folder', '../data/omniglot_resized',
                    'Path to data folder')
flags.DEFINE_multi_integer('img_size', (28, 28),
                     'Image size')


# @tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        preds = model(inputs, labels)
        loss = loss_function(preds, labels)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


# @tf.function
def test_step(inputs, labels):
    preds = model(inputs, labels)
    loss = loss_function(preds, labels)
    exp = tf.exp(preds[:, -1, :, :])
    sum_exp = tf.expand_dims(tf.math.reduce_sum(exp, axis=-1), axis=-1)
    softmax = exp / sum_exp
    return preds, loss


if __name__ == '__main__':
    config = {
        'data_folder': FLAGS.data_folder,
        'img_size': FLAGS.img_size
    }
    data_generator = DataGenerator(FLAGS.num_classes,
                                   FLAGS.num_samples + 1,
                                   config=config)
    model = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
    input_images, input_labels = data_generator.sample_batch('train', 9)
    out = model(input_images, input_labels)
    optimizer = tf.keras.optimizers.Adam(0.01)
    max_step = 100000
    train_loss = []
    test_loss = []
    accuracy = []
    try:
        for step in range(max_step):
            inputs, labels = data_generator.sample_batch('train', FLAGS.meta_batch_size)
            tr_loss = train_step(inputs, labels)

            if step % 100 == 0:
                print("*" * 5 + "Iter " + str(step) + "*" * 5 + "\n")
                inputs, labels = data_generator.sample_batch('test', 100)
                pred, t_loss = test_step(inputs, labels)
                train_loss.append(tr_loss.numpy())
                test_loss.append(t_loss.numpy())
                print("Train Loss:", tr_loss.numpy(), "Test Loss:", t_loss.numpy())
                pred = tf.reshape(pred, (-1, FLAGS.num_samples + 1,
                                        FLAGS.num_classes, FLAGS.num_classes))
                # calculate softmax
                exp = tf.math.exp(pred[:, -1, :, :])
                sum_exp = tf.expand_dims(tf.math.reduce_sum(exp, axis=-1), axis=-1)
                pred = exp / sum_exp
                pred = tf.math.argmax(pred[:, :, :], axis=2)

                # get labels
                labels = tf.math.argmax(labels[:, -1, :, :], axis=2)
                accuracy.append(tf.math.reduce_mean(tf.cast(pred == labels, dtype=tf.float32)).numpy())
                print("Test Accuracy", (tf.math.reduce_mean(tf.cast(pred == labels, dtype=tf.float32))).numpy())
    except:
        pass

    # plot
    steps = np.arange(0, step, 100)
    plt.plot(steps, train_loss, label='train loss')
    plt.plot(steps, test_loss, label='test loss')
    plt.legend()
    plt.savefig('loss_{}-shot_{}-way.png'.format(FLAGS.num_samples, FLAGS.num_classes))
    plt.close()
    plt.plot(steps, accuracy)
    plt.savefig('accuracy_{}-shot_{}-way.png'.format(FLAGS.num_samples, FLAGS.num_classes))
