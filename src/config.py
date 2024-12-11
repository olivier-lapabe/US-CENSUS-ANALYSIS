from lightgbm import LGBMClassifier
import os
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# Model Selection
## Logistic Regression
# BASE_MODEL = LogisticRegression()
# PARAMS_GRID = {
#     'C': [0.01, 0.1, 1],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear']
# }
## Random Forests
# BASE_MODEL = RandomForestClassifier()
# PARAMS_GRID = {
#     'max_depth': [5, 7, 10, 50, -1],
#     'n_estimators': [100, 200, 500],
# }
## XGBoost
# BASE_MODEL = XGBClassifier()
# PARAMS_GRID = {
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
#     'n_estimators': [50, 100, 200, 500],
# }
## LightGBM
BASE_MODEL = LGBMClassifier(verbose=-1)
PARAMS_GRID = {
    'num_leaves': [40, 50, 60],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.05, 0.1, 0.3],
    'n_estimators': [100, 200, 500],
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

# Columns
TARGET_COLUMN = 'income bracket'
WEIGHT_COLUMN = 'instance weight'
COLUMNS_MISSING_VALUES = [
    'migration code-change in msa', 
    'migration code-change in reg',
    'migration code-move within reg', 
    'migration prev res in sunbelt',
]
COLUMNS_LOW_CRAMERS_V = [
    'sex',
    'full or part time employment stat',
    'family members under 18',
    'veterans benefits',
    'own business or self employed',
    'member of a labor union',
    'country of birth father',
    'country of birth mother',
    'hispanic origin',
    'enroll in edu inst last wk',
    'country of birth self',
    'race',
    'citizenship',
    'state of previous residence',
    'region of previous residence',
    'live in this house 1 year ago',
    'reason for unemployment',
    'fill inc questionnaire for veteran\'s admin',
]
COLUMNS_LOW_SPEARMAN = ['wage per hour']
COLUMNS_MAJOR_DETAILED  =[
    'major industry code',
    'major occupation code',
    'detailed household summary in household',
    #'detailed industry recode',
    #'detailed occupation recode',
    #'detailed household summary in household',
]
CATEGORICAL_COLUMNS = [
    'class of worker',
    'detailed industry recode',
    'detailed occupation recode',
    'marital stat',
    #'major industry code',
    #'major occupation code',
    'tax filer stat',
    'detailed household and family stat',
    #'detailed household summary in household',
    #'education',
    'education_group',
    #'capital income binary',
    #'weeks worked in year binary',
]
NON_CATEGORICAL_COLUMNS = [
    'age',
    'capital income',
    'capital losses',
    'num persons worked for employer',
    'weeks worked in year',
]
EDUCATION_MAP = {
    'No high school graduate': [
        '12th grade no diploma',
        '11th grade',
        '10th grade',
        '9th grade',
        '7th and 8th grade',
        '5th or 6th grade',
        '1st 2nd 3rd or 4th grade',
        'Less than 1st grade',
        'Children',
    ],
    'High school graduate': [
        'High school graduate',
    ],
    'Some college but no degree': [
        'Some college but no degree',
    ],
    'Associates degree': [
        'Associates degree-occup /vocational',
        'Associates degree-academic program',
    ],
    'Bachelors degree': [
        'Bachelors degree(BA AB BS)',
    ],
    'Masters degree': [
        'Masters degree(MA MS MEng MEd MSW MBA)',
    ],
    'Doctorate or Prof school degree': [
        'Prof school degree (MD DDS DVM LLB JD)',
        'Doctorate degree(PhD EdD)',
    ],
}
