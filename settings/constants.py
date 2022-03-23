import os

DATA_FOLDER = 'data'
TRAIN_CSV = os.path.join(DATA_FOLDER, 'train.csv')
VAL_CSV = os.path.join(DATA_FOLDER, 'val.csv')

ESTIMATOR_FOLDER = 'models'
SAVED_ESTIMATOR = os.path.join(ESTIMATOR_FOLDER, 'GradientBoostingClassifier.pickle')