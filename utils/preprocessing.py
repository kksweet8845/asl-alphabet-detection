import tensorflow as tf
import glob
import os


def _bytes_feature(value):
    """ Return a byte_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """ Return the float_list from a float / double. """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """ Return the int64_list from a bool/ enum / int / uint. """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tfRecord(from_dir, type="jpg", tf_output_file="images.tfrecords"):
    """ Transform a set of image data in a directory into tfRecord type
        @from_dir : (string), the pathname of the directory
        @type     : (string), the type of the images
    """

    def img_example(img_raw, label):
        """
            The img features description
            Each sample(image) has the following properties
            {
                height  : int64,
                width   : int64,
                depth   : int64,
                label   : int64,
                img_raw : bytes
            }
        """
        img_tensor = tf.io.decode_image(img_raw)
        img_shape = img_tensor.shape

        feature = {
            'height'   : _int64_feature(img_shape[0]),
            'width'    : _int64_feature(img_shape[1]),
            'depth'    : _int64_feature(img_shape[2]),
            'label'    : _int64_feature(ord(label) - ord('A')),
            'img_raw'  : _bytes_feature(img_raw)
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    data_size = 0
    writer = tf.io.TFRecordWriter(tf_output_file)
    for root, dirs, files in os.walk(from_dir, topdown=False):
        for dirname in dirs:
            label = dirname
            if len(label) != 1:
                continue
            for imagepath in glob.glob(os.path.join(root, dirname, f'*.{type}')):
                img_string = open(imagepath, 'rb').read()
                tf_example = img_example(img_string, label)
                writer.write(tf_example.SerializeToString())
                data_size += 1
    return data_size



def parse_tfRecords(tfRecords_file="images.tfrecords"):
    """ 
        Parse the tfrecords file
    """

    raw_image_dataset = tf.data.TFRecordDataset(tfRecords_file)

    img_feature_desc = {
        'height' : tf.io.FixedLenFeature([], tf.int64),
        'width'  : tf.io.FixedLenFeature([], tf.int64),
        'depth'  : tf.io.FixedLenFeature([], tf.int64),
        'label'  : tf.io.FixedLenFeature([], tf.int64),
        'img_raw' : tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_example_fn(example_proto):
        return tf.io.parse_single_example(example_proto, img_feature_desc)

    parsed_img_dataset = raw_image_dataset.map(_parse_example_fn)

    return parsed_img_dataset






            





    
    
