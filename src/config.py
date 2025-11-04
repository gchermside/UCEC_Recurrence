# config.py


# Directories ---------------------------------------------------------------------------
DATA_DIR = '../preprocessed_data/'
SPLIT_DATA_DIR = "../split_data_balanced/"
MODEL_DIR = '../models/'

# Data file paths ------------------------------------------------------------------------
CLINICAL_DATA_PATH = "../data/data_clinical_patient.txt"
CLINICAL_DATA_PATH = "../ucec_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
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
    "CANCER_TYPE_ACRONYM",
    "OTHER_PATIENT_ID",
    "SEX",
    "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS",
    "HISTORY_NEOADJUVANT_TRTYN",
    "PATH_M_STAGE",
    "ICD_O_3_SITE", # removed because is the same as ICD_10
    "ICD_O_3_",
    "DAYS_LAST_FOLLOWUP",              # follow-up time after diagnosis (future info)
    "FORM_COMPLETION_DATE",            # administrative metadata, not predictive
    "INFORMED_CONSENT_VERIFIED",       # administrative, no biological meaning
    "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",  # recurrence event → direct leakage
    "PERSON_NEOPLASM_CANCER_STATUS",   # disease status at follow-up → leakage
    "IN_PANCANPATHWAYS_FREEZE",        # technical/analysis flag, not biological
    "OS_STATUS",                       # overall survival outcome → leakage
    "OS_MONTHS",                       # overall survival time → leakage
    "DSS_STATUS",                      # disease-specific survival outcome → leakage
    "DSS_MONTHS",                      # disease-specific survival time → leakage
    "DFS_STATUS",                      # disease-free survival outcome → leakage
    "DFS_MONTHS",                      # disease-free survival time → leakage
    "PFS_STATUS",                      # progression-free survival outcome → leakage
    "PFS_MONTHS"                       # progression-free survival time → leakage
]

CATEGORICAL_COLS = ['SUBTYPE',
                    'ETHNICITY', 
                    "ICD_10", 
                    "ICD_O_3_HISTOLOGY", 
                    "PRIOR_DX", 
                    "RACE", 
                    "RADIATION_THERAPY", 
                    "GENETIC_ANCESTRY_LABEL"
]

CLINICAL_MAX_NULL_FRAC = 0.25
CLINICAL_UNIFORM_THRESH = 0.99

# Mrna preprocessing hyperparameters ------------------------------------------------------

MAX_NULL_FRAC = 0.25
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
MUTATION_COLS_TO_REMOVE = [] # consider removing common passenger genes
MUTATION_MAX_NULL_FRAC = 0.3
MUTATION_UNIFORM_THRESH = 0.99


# Experiment metadata
SEED = 100
CLIN_COLS_TO_STRATIFY_ON = ['SUBTYPE', 'AGE', 'RACE', 'ETHNICITY', 'ICD_10', 'ICD_O_3_HISTOLOGY', 'NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT', 'PATH_T_STAGE', 'PATH_N_STAGE', 'PATH_M_STAGE', 'OS_STATUS', 'DSS_STATUS', 'DFS_STATUS', 'PFS_STATUS']
P_VALUE_STRATIFICATION = 0.05
TEST_SIZE = 0.15
VAL_SIZE = 0.15
STRATIFICATION_MAX_ATTEMPTS = 100


# Neural Network Hyperparameters --------------------------------------------------------------
NUM_EPOCHS = 200
PATIENCE = 40
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
HIDDEN_DIM = 64
DROPOUT = 0.0
