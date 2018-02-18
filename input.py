import os

from matplotlib.image import imread
import tensorflow as tf

import dataset


# https://www.youtube.com/watch?v=oxrcZ9uUblI  funny and good tutorial on tensorflow records
def create_tfrecords(name='OxfordFlower', datadir=os.path.expanduser('~/OxfordFlower')):
    # name:  Name of dataset
    # datadir:  Directory to store data in

    # get or retrieve data/paths
    data = dataset.Dataset(name, datadir)

    # targets for efficient tensorflow record files
    train_record_path = os.path.join(datadir, 'train.tfrecord')
    test_record_path = os.path.join(datadir, 'test.tfrecord')

    # run data conversion
    convert(
        *data.get_train_data(), train_record_path
    )
    convert(
        *data.get_test_data(), test_record_path
    )


def wrap_bytes(value):
    # converts raw image data into tf object
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def wrap_int64(value):
    # converts int to tf object
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert(image_paths, labels, record_path):
    # Number of images. Used when printing the progress.

    # Open each image and write it into TFRecord.
    with tf.python_io.TFRecordWriter(record_path) as writer:
        for path, label in zip(image_paths, labels):
            # load image data
            img = imread(path)
            x, y, channels = img.shape  #@TODO double check dimension order

            # Convert the image to raw bytes.
            img_bytes = img.tostring()

            # decide how label must be wrapped:
            if isinstance(label, int):
                wrap_label = wrap_int64
            elif isinstance(label, str):
                # assuming we have segmentations:
                img = imread(label)
                label = img.tostring()
                wrap_label = wrap_bytes

            # TFRecords data storage, helper functions convert raw data to tf objects
            data = \
                {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_label(label),
                    'xdim': wrap_int64(x),
                    'ydim': wrap_int64(y),
                    'channels': wrap_int64(channels)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)
