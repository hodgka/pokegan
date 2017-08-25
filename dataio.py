import os

import tensorflow as tf
from scipy.misc import imread


class Dataset:
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

        basepath = os.path.join('/home/alec/datasets', self.dataset)
        fnames = [os.path.join(basepath, fname) for fname in os.listdir(
            basepath) if (('.png' in fname) and ('-' not in fname))]
        fname_queue = tf.train.string_input_producer(fnames, shuffle=True)
        image_reader = tf.WholeFileReader()
        _, ims = image_reader.read(fname_queue)

        ims = tf.image.decode_png(ims, self.channels)
        ims = tf.cast(ims, tf.int32)
        ims = tf.image.resize_images(ims, [self.height, self.width])
        ims = tf.stack([ims, tf.image.flip_left_right(ims)])

        min_after_dequeue = 100
        capacity = min_after_dequeue + 3 * self.batch_size
        self.ims = tf.train.shuffle_batch([ims],
                                          batch_size=self.batch_size,
                                          min_after_dequeue=min_after_dequeue,
                                          capacity=capacity,
                                          enqueue_many=True)


if __name__ == "__main__":
    data = Dataset({"dataset": "pokemon_sprites", "batch_size": 32, "height": 96, "width": 96, "channels": 3})
