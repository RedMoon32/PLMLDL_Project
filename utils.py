import pandas as pd
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import *
import sklearn
from sktime.classification.interval_based import TimeSeriesForest
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.pipeline import Pipeline as SPipe
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.base import clone
from collections import Counter
from nltk.corpus import stopwords
import nltk
from nltk import *  
from sklearn.feature_extraction.text import TfidfVectorizer

#cross_val_score
FID = 'Field ID'
FAR = 'Field Area'
Y = 'Year'
CULT = 'Culture'
days = [f'Day {i}' for i in range(1, 367)]

# train = pd.read_csv('./train.csv')
# test = pd.read_csv('./test.csv')
