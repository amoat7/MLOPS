
# Define imports
from kerastuner.engine import base_tuner 
import kerastuner as kt 
from tensorflow import keras 

from typing import NamedTuple, Dict, Text, Any, List 
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor 
import tensorflow as tf 
import tensorflow_transform as tft

# Declare namedtuple field names 
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text,Any])])

# Callback for search strategy
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

# import same constants from transform module
import census_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_DICT = census_constants.VOCAB_FEATURE_DICT
#_BUCKET_FEATURE_DICT = census_constants.BUCKET_FEATURE_DICT
_NUM_OOV_BUCKETS = census_constants.NUM_OOV_BUCKETS
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name



def _gzip_reader_fn(filenames):
    '''Load compressed dataset

    Args:
    filenames - filenames of TFRecords to load

    Returns:
    TFRecord Dataset loaded from filenames
    '''

    # load the dataset. Specify the compression type as '.gz
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=128) -> tf.data.Dataset:
    '''Create batches of features and labels from TF Records

    Args:
    file_pattern - List of files or patterns of file paths containing Example records.
    tf_transform_output - transform output graph
    num_epochs - Integer specifying the number of times to read through the dataset. 
            If None, cycles through the dataset forever.
    batch_size - An int representing the number of records to combine in a single batch.

    Returns:
    A dataset of dict elements, (or a tuple of dict elements and label). 
    Each dict maps feature keys to Tensor or SparseTensor objects.
    '''

    # Get feature specification based on transform output
    transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy()
    )

    # create batches of features and labels
    dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn, 
      num_epochs=num_epochs, 
      label_key=_transformed_name(_LABEL_KEY)
    )

    return dataset


def model_builder(hp):
    '''
    Builds the model and sets up the hyperparameters to tune.

    Args:
    hp - Keras tuner object

    Returns:
    model with hyperparameters to tune
    '''
    num_unit = hp.Int('num_unit', min_value=1, max_value=10, step=2)
    
    num_hidden_layers = hp.Int('hidden_cat', min_value=1, max_value=5)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Define the input layers for the numeric keys
    input_numeric = [
      tf.keras.layers.Input(name=_transformed_name(colname), shape=(1,), dtype=tf.float32) 
      for colname in _NUMERIC_FEATURE_KEYS
    ]

    # Define the input layers for vocab keys
    input_categorical = [
      tf.keras.layers.Input(name=_transformed_name(colname), shape=(vocab_size + _NUM_OOV_BUCKETS,), dtype=tf.float32) 
      for colname, vocab_size in _VOCAB_FEATURE_DICT.items()
    ]

    # Define input layers for bucket key
    #input_categorical += [
      #tf.keras.layers.Input(name=_transformed_name(colname), shape=(num_buckets,), dtype=tf.float32) 
      #for colname, num_buckets in _BUCKET_FEATURE_DICT.items()
    #]

    # Concatenate numeric inputs
    deep = tf.keras.layers.concatenate(input_categorical)

    # Create a dense deep network for categorical inputs
    for i in range(num_hidden_layers):
        num_nodes =  hp.Int('unit'+str(i), min_value=16, max_value=80, step=16)
        deep = tf.keras.layers.Dense(num_nodes)(deep)

    # Concatenate numeric inputs
    wide = tf.keras.layers.concatenate(input_numeric)

    # create a shallow dense network for categorical inputs
    wide = tf.keras.layers.Dense(num_unit, activation='relu')(wide)

    # Combine wide and deep networks
    combined = tf.keras.layers.concatenate([deep, wide])


    # Define output of binary classifier
    output = tf.keras.layers.Dense(
      1, activation='sigmoid')(combined)

    # Setup combined input
    input_layers = input_numeric + input_categorical

    # Create the Keras model
    model = tf.keras.Model(input_layers, output)

    # Define training parameters
    model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
      metrics='binary_accuracy')

    # Print model summary
    model.summary()

    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:

    """Build the tuner using the KerasTuner API.
    Args:
    fn_args: Holds args as name/value pairs.

      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.

    Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
    """

    # Define tuner search strategy 
    tuner = kt.Hyperband(
      model_builder,
      objective='val_binary_accuracy',
      max_epochs=20,
      factor=2,
      directory=fn_args.working_dir,
      project_name='kt_hyperband'
    )

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Use _input_fn() to extract input features and labels from the train and val set
    train_set = _input_fn(fn_args.train_files[0], tf_transform_output)
    val_set = _input_fn(fn_args.eval_files[0], tf_transform_output)


    return TunerFnResult(
      tuner=tuner,
      fit_kwargs={ 
          "callbacks":[stop_early],
          'x': train_set,
          'validation_data': val_set,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      }
    )
