
# Features with string data types will be convert to indices
VOCAB_FEATURE_DICT = {
    'education': 16, 'marital_status':7, 'occupation':15, 'race':5,
    'relationship':6, 'workclass':9, 'sex':2, 'native_country':42
}

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week','age']

# Feature that can be grouped into buckets
BUCKET_FEATURE_DICT = {'age': 4}

# Number of out-of-vocab buckets
NUM_OOV_BUCKETS = 1

# Feature that the model will predict #predicting if a person makes above 50K
LABEL_KEY = 'label'

# Utility for renaming the feature 
def transformed_name(key):
    key = key.replace('-','_')
    return key + '_xf'
