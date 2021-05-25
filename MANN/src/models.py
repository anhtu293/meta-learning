import numpy as np
import random
import tensorflow as tf
# from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    # Only calculate loss on the last
    # sample for each class
    exp = tf.exp(preds[:, -1, :, :])
    sum_exp = tf.expand_dims(tf.math.reduce_sum(exp, axis=-1), axis=-1)
    softmax = exp / sum_exp
    log_softmax = tf.math.log(softmax)
    CE = -tf.math.reduce_sum(tf.multiply(labels[:, -1, :, :], log_softmax))
    return CE

class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        # Must convert to numpy array then 
        # assign value and convert back to tf tensor
        # because tf tensor is immutable
        tmp = input_labels.numpy()
        tmp[:, -1, :, :] = 0
        input_labels = tf.convert_to_tensor(tmp)
        concat_input = tf.keras.layers.concatenate([input_images, input_labels], axis=3)

        batch_size = tf.shape(input_images)[0]
        out = []
        # for cls in range(self.num_classes):
        for b in range(batch_size):
            x = self.layer1(concat_input[b, :, :, :])
            out.append(self.layer2(x))
        out = tf.stack(out, axis=0)

        return out


class CNN_MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        # Must convert to numpy array then 
        # assign value and convert back to tf tensor
        # because tf tensor is immutable
        tmp = input_labels.numpy()
        tmp[:, -1, :, :] = 0
        input_labels = tf.convert_to_tensor(tmp)
        concat_input = tf.keras.layers.concatenate([input_images, input_labels], axis=3)

        batch_size = tf.shape(input_images)[0]
        out = []
        # for cls in range(self.num_classes):
        for b in range(batch_size):
            x = self.layer1(concat_input[b, :, :, :])
            out.append(self.layer2(x))
        out = tf.stack(out, axis=0)

        return out


# if __name__ == '__main__':
#     data_generator = DataGenerator(5, 4 + 1)
#     model = MANN(5, 5)
#     input_images, input_labels = data_generator.sample_batch('train', 9)
#     out = model(input_images, input_labels)
#     print(out.shape)
#     print(loss_function(out, input_labels))
