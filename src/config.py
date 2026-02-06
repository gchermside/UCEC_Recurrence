# config.py


# Directories ---------------------------------------------------------------------------
DATA_DIR = '../preprocessed_data/'
SPLIT_DATA_DIR = "../split_data/"
MODEL_DIR = '../models/'

# Data file paths ------------------------------------------------------------------------
CLINICAL_DATA_PATH = "../ucec_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
SAMPLE_CLINICAL_DATA_PATH = "../ucec_tcga_pan_can_atlas_2018/data_clinical_sample.txt"
MRNA_DATA_PATH = "../ucec_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt"
TREATMENT_DATA_PATH = "../ucec_tcga_pan_can_atlas_2018/data_timeline_treatment.txt"
STATUS_DATA_PATH = "../ucec_tcga_pan_can_atlas_2018/data_timeline_status.txt"
MUTATION_DATA_PATH = "../ucec_tcga_pan_can_atlas_2018/data_mutations.txt"

X_TRAIN_PATH = DATA_DIR + 'X_train.pkl'
Y_TRAIN_PATH = DATA_DIR + 'y_train.pkl'
X_TEST_PATH = DATA_DIR + 'X_test.pkl'
Y_TEST_PATH = DATA_DIR + 'y_test.pkl'
FEATURE_NAMES = DATA_DIR + 'feature_names.pkl'
CORRELATED_GENES_PATH = DATA_DIR + "correlated_genes_to_remove.pkl"

# Model paths -----------------------------------------------------------------------------
SVC_NO_LASSO_MODEL_PATH = MODEL_DIR + 'SVC_no_LASSO.pkl'
SVC_WITH_LASSO_MODEL_PATH = MODEL_DIR + 'SVC_with_LASSO.pkl'
RF_MODEL_PATH = MODEL_DIR + 'random_forest_model.pkl'
LASSO_MODEL_PATH = MODEL_DIR + 'lasso_model.pkl'
LR_MODEL_PATH = MODEL_DIR + 'logistic_regression.pkl'
XGB_MODEL_PATH = MODEL_DIR + 'xgboost_model_with_LASSO.pkl'

# Clinical Preprocessing hyperparamters ----------------------------------------------------
CLINICAL_COLS_TO_REMOVE = [
    "OTHER_PATIENT_ID",
    "AJCC_STAGING_EDITION",
    "DAYS_LAST_FOLLOWUP",
    "DAYS_TO_BIRTH",
    "FORM_COMPLETION_DATE",
    "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
    "PERSON_NEOPLASM_CANCER_STATUS",
    "RADIATION_THERAPY",
    "IN_PANCANPATHWAYS_FREEZE",
    "OS_STATUS",
    "OS_MONTHS",
    "DSS_STATUS",
    "DSS_MONTHS",
    "DFS_STATUS",
    "DFS_MONTHS",
    "PFS_STATUS",
    "PFS_MONTHS",
    "CANCER_TYPE",
    "CANCER_TYPE_DETAILED",
    "TUMOR_TYPE",
    "TISSUE_PROSPECTIVE_COLLECTION_INDICATOR",
    "TISSUE_RETROSPECTIVE_COLLECTION_INDICATOR",
    "TUMOR_TISSUE_SITE",
    "SAMPLE_TYPE",
    "SOMATIC_STATUS",
    "TISSUE_SOURCE_SITE",
    "TISSUE_SOURCE_SITE_CODE",
    "SAMPLE_ID"
]


CATEGORICAL_COLS = ['SUBTYPE',
                    'ETHNICITY', 
                    "ICD_10", 
                    "ICD_O_3_HISTOLOGY", 
                    "PRIOR_DX", 
                    "RACE", 
                    "RADIATION_THERAPY", 
                    "GENETIC_ANCESTRY_LABEL",
                    "CLINICAL_STAGE",
                    "ONCOTREE_CODE",
                    "GRADE",
                    "WEIGHT_BIN"
]

CLINICAL_MAX_NULL_FRAC = 0.25
CLINICAL_UNIFORM_THRESH = 0.90
WEIGHT_NUM_BINS = 4

# Mrna preprocessing hyperparameters ------------------------------------------------------

MAX_NULL_FRAC = 0.50 #FIXME SHOULD BE 0.25
UNIFORM_THRESHOLD = 0.99
CORRELATION_THRESHOLD = 0.9
VARIANCE_THRESHOLD = 1e-5
RE_RUN_PRUNING = False
# Genes from https://pmc.ncbi.nlm.nih.gov/articles/PMC7565375/ 
# and https://pmc.ncbi.nlm.nih.gov/articles/PMC9929804/ FIXME: look more into this later
LITERATURE_GENES = set([
    "MLH1", "MSH2", "MSH6", "PMS2", "PTEN", "POLD1", "POLE", "NTHL1", "MUTYH", "BRCA1", "GINS4", "ESR1"
])

USE_STABILITY_SELECTION = False # so far, does not help
# Stability selection parameters
N_BOOTS_FPR = 50
FPR_ALPHA = 0.01
STABILITY_THRESHOLD_FPR = 0.9

# Bootstrapped SelectKBest parameters
K = 500
N_BOOTS_KBEST = 50
THRESHOLD_KBEST = 0.4

# Mutation preprocessing hyperparameters ----------------------------------------------------
MAX_MUTATION_COUNT = 5
MUTATION_UNIFORM_THRESH = 0.98


# Experiment metadata
SEED = 42
CLIN_COLS_TO_STRATIFY_ON = ['SUBTYPE', 'AGE', 'RACE', 'ETHNICITY', 'ICD_10', 'ICD_O_3_HISTOLOGY', 'NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT', 'PATH_T_STAGE', 'PATH_N_STAGE', 'PATH_M_STAGE', 'OS_STATUS', 'DSS_STATUS', 'DFS_STATUS', 'PFS_STATUS', 'CLINICAL_STAGE']
P_VALUE_STRATIFICATION = 0.05
TEST_SIZE = 0.15
VAL_SIZE = 0.15
STRATIFICATION_MAX_ATTEMPTS = 100


# Neural Network Hyperparameters --------------------------------------------------------------
NUM_EPOCHS = 1000
PATIENCE = 100
LEARNING_RATE = 4.066152607618111e-05
BATCH_SIZE = 32
CLIN_HIDDEN = [128]
MRNA_HIDDEN = [64]
MUT_HIDDEN = [64]
CLIN_DROPOUT = [0.0]
MRNA_DROPOUT = [0.3]
MUT_DROPOUT = [0.3]
FUSION_HIDDEN = 64
FUSION_DROPOUT = 0.26992054565083656
ACTIVATION_FUNC = 'leaky_relu'
