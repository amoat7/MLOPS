
import tensorflow as tf 
import tensorflow_transform as tft

# import constants from cells above
import census_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_DICT = census_constants.VOCAB_FEATURE_DICT
#_BUCKET_FEATURE_DICT = census_constants.BUCKET_FEATURE_DICT
_NUM_OOV_BUCKETS = census_constants.NUM_OOV_BUCKETS
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name

# Define the transformations
def preprocessing_fn(inputs):

    # Initialize outputs dictionary
    outputs = {}

    # Scale these features to the range of [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        scaled = tft.scale_to_0_1(inputs[key])
        outputs[_transformed_name(key)] = tf.reshape(scaled, [-1]) # flattens to 1D

    # Convert strings to indices and convert to one-hot vectors 
    for key, vocab_size in _VOCAB_FEATURE_DICT.items():
        indices = tft.compute_and_apply_vocabulary(inputs[key], num_oov_buckets = _NUM_OOV_BUCKETS)
        one_hot = tf.one_hot(indices, vocab_size + _NUM_OOV_BUCKETS)
        outputs[_transformed_name(key)] = tf.reshape(one_hot, [-1, vocab_size + _NUM_OOV_BUCKETS])

    # Bucketize this feature and convert to one-hot vectors
    #for key, num_buckets in _BUCKET_FEATURE_DICT.items():
        #indices = tft.bucketize(inputs[key], num_buckets)
        #one_hot = tf.one_hot(indices, num_buckets)
        #outputs[_transformed_name(key)] = tf.reshape(one_hot, [-1, num_buckets])

    # Cast label to float
    outputs[_transformed_name(_LABEL_KEY)] = tf.cast(inputs[_LABEL_KEY], tf.float32)

    return outputs
