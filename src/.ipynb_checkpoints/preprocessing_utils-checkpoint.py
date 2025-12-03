#### Imports ######################################################

import numpy as np
import pandas as pd
import os
import joblib

from sklearn.utils import resample
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFpr, f_classif, chi2
from sklearn.utils import resample
from sklearn.preprocessing import RobustScaler

from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import config

#### Loading and Splitting Data ######################################################


def load_clinical_data(clinical_patient_file, clinical_sample_file, timeline_file):
    '''Loads in clinical data, from both the patient and sample files. These files share the same patients. 
    Loads the Tumor Stage at inital diagnosis from the timeline status file.'''
    # Load files
    clinical_patient_file = pd.read_csv(clinical_patient_file, sep="\t", comment="#", low_memory=False)
    clinical_sample_file = pd.read_csv(clinical_sample_file, sep="\t", comment="#", low_memory=False)

    clinical_df = clinical_patient_file.merge(clinical_sample_file, on="PATIENT_ID", how="left")
    
    timeline_df = pd.read_csv(timeline_file, sep="\t", comment="#", low_memory=False)

    # Extract initial diagnosis rows
    init_dx = timeline_df[timeline_df["STATUS"] == "Initial Diagnosis"]

    # Check for duplicates
    dupes = init_dx[init_dx.duplicated("PATIENT_ID", keep=False)]
    if len(dupes) > 0:
        print("WARNING: Duplicate INITIAL DIAGNOSIS entries detected!")
        print(dupes.sort_values("PATIENT_ID"))
        raise ValueError("Found duplicate INITIAL DIAGNOSIS entries. Resolve before merging.")

    init_dx = init_dx[["PATIENT_ID", "CLINICAL_STAGE"]]
    merged = clinical_df.merge(init_dx, on="PATIENT_ID", how="left")

    return merged


def load_mrna_data(mrna_file):
    mrna_df = pd.read_csv(mrna_file, sep="\t", comment="#", low_memory=False)
    # The first 2 columns of the mRNA data are labels (Hugo_Symbol then Entrez_Gene_Id). 
    # 13 of the genes do not have Hugo_symbols, so for these I will you the Entrex_Gene_Id as the label.
    missing_symbols = mrna_df['Hugo_Symbol'].isnull()
    mrna_df.loc[missing_symbols, 'Hugo_Symbol'] = mrna_df.loc[missing_symbols, 'Entrez_Gene_Id'].astype(str)
    # There are 7 rows that have both the same Hugo_Symbol and Entrez_Gene_Id but different values for the patients.
    # I will rename these rows to have unique labels by appending -1-of-2 and -2-of-2 to the Hugo_Symbol.
    # Get value counts
    counts = mrna_df['Hugo_Symbol'].value_counts()

    # Generate unique labels for duplicates
    def label_duplicates(value, index):
        if counts[value] == 1:
            return value  # Keep unique values unchanged
        occurrence = mrna_df.groupby('Hugo_Symbol').cumcount() + 1  # Count occurrences per group
        return f"{value}-{occurrence[index]}-of-{counts[value]}"

    # Apply the labeling function
    mrna_df['Hugo_Symbol'] = [label_duplicates(value, idx) for idx, value in mrna_df['Hugo_Symbol'].items()]

    mrna_df = mrna_df.set_index('Hugo_Symbol')
    mrna_df = mrna_df.drop(columns="Entrez_Gene_Id") # removing the label column before I transpose the df
    mrna_df= mrna_df.transpose() # now the patients are the index and the genes are the columns
    assert all(idx.endswith("01") for idx in mrna_df.index), "Not all IDs end with '01'"
    mrna_df.index = [id[:-3] for id in mrna_df.index] # removes extranious -01 so that the patient ids match the clinical data
    return mrna_df

def load_mutation_data(mutation_file):
    """
    Load TCGA mutation data and convert to a patient × gene binary matrix.
    Checks sample suffixes to ensure only expected codes (-01, -10) appear.
    
    Parameters
    ----------
    mutation_file : str
        Path to TCGA mutation file (MAF or similar).

    Returns
    -------
    mut_df : pd.DataFrame
        Patient × gene binary mutation matrix (index = patient ID root).
    """
    # Load mutation file
    df = pd.read_csv(mutation_file, sep="\t", low_memory=False)
    
    # Filter: remove silent/RNA mutations, remove sex chromosomes
    q = "Chromosome not in ['X', 'Y'] and Variant_Classification not in ['Silent', 'RNA']"
    df = df.query(q).dropna(subset=["Hugo_Symbol"])
    
    # Extract suffix (sample type code) from barcodes
    df["Sample_Suffix"] = df["Tumor_Sample_Barcode"].str[13:15]
    allowed_suffixes = {"01", "10"}
    found_suffixes = set(df["Sample_Suffix"].unique())
    
    # Assert only expected suffixes are present
    unexpected = found_suffixes - allowed_suffixes
    assert not unexpected, f"Unexpected sample suffixes found: {unexpected}"
    
    # Keep only patient ID root (first 12 chars)
    df["Patient_ID"] = df["Tumor_Sample_Barcode"].str[:12]
    
    # Crosstab to patient × gene mutation matrix
    mut_df = pd.crosstab(df["Patient_ID"], df["Hugo_Symbol"]).astype(float)
    mut_df.columns = mut_df.columns.astype(str) + "_mut"
    
    return mut_df


def label_patients_with_stats(status_df, treatment_df=None, clinical_df=None, min_followup_days=1095):
    '''
    Rules:
    1. Positive for recurrence (1) if:
       - STATUS in status_df contains any of:
         "Locoregional Recurrence", "Distant Metastasis", "Metastatic"
       - OR treatment_df contains:
         ANATOMIC_TREATMENT_SITE in ["Local Recurrence", "Distant Recurrence"]
         or REGIMEN_INDICATION == "Recurrence"
       - Use START_DATE from the corresponding row as RECURRENCE_DATE.
    2. No recurrence (0) if:
       - STATUS does NOT contain:
         "Locoregional Recurrence", "Distant Metastasis", "Metastatic",
         "New Primary Tumor", "Locoregional Disease", "DECEASED"
       - AND Last Follow Up row exists with START_DATE >= min_followup_days
       - AND (PRIMARY_THERAPY_OUTCOME_SUCCESS == "Complete Remission/Response"
         OR (NaN and TUMOR_STATUS == "tumor_free"))
    2. Verify DFS_STATUS in clinical_df:
       - If patient is labeled 1 but DFS_STATUS == "0:DiseaseFree", label NaN.
       - If patient is labeled 0 but DFS_STATUS == "1:Recurred/Progressed", label NaN.

    4. Otherwise label as NaN.
    
    Returns
    -------
    labels_df : pd.DataFrame
        DataFrame with one row per patient containing:
            - PATIENT_ID
            - LABEL (1 = recurrence, 0 = no recurrence, NaN = inconclusive)
            - FOLLOW_UP_DAYS
            - RECURRENCE_DATE
            - RECURRENCE_TYPE
            - SOURCE (origin of the label or reason for exclusion)

    stats : dict
        Dictionary of counts summarizing labeling:
            - recurrence_from_status
            - recurrence_from_treatment
            - no_recur_CRR
            - no_recur_tumor_free_fallback
            - excluded_progression_events
            - excluded_not_complete_remission
            - excluded_no_last_followup
            - excluded_short_followup
            - excluded_DFS_conflict

    exclusion_reason : dict
        Dictionary mapping patient IDs to the reason they were excluded or labeled NaN.
    """

'''
    recurrence_events = {"Locoregional Recurrence", "Distant Metastasis", "Metastatic"}
    progression_events = recurrence_events.union({"New Primary Tumor", "Locoregional Disease", "DECEASED"})
    treatment_recur_sites = {"Local Recurrence", "Distant Recurrence"}

    results = []
    stats = {
        "recurrence_from_status": 0,
        "recurrence_from_treatment": 0,
        "no_recur_CRR": 0,
        "no_recur_tumor_free_fallback": 0,
        "excluded_progression_events": 0,
        "excluded_not_complete_remission": 0,
        "excluded_no_last_followup": 0,
        "excluded_short_followup": 0,
        "excluded_DFS_conflict": 0
    }
    exclusion_reason = {}

    # Ensure all DFs have PATIENT_ID as a column
    for df_name, df in zip(["status_df", "treatment_df", "clinical_df"], [status_df, treatment_df, clinical_df]):
        if df is not None and "PATIENT_ID" not in df.columns:
            raise ValueError(f"{df_name} must contain column PATIENT_ID")

    for pid, df in status_df.groupby("PATIENT_ID"):
        df = df.sort_values("START_DATE")
        label = np.nan
        src = None
        followup_days = df["START_DATE"].max()
        rec_date = None
        rec_type = None

        # ------------------------------
        # Step 1: Recurrence via status_df
        # ------------------------------
        recur_rows = df[df["STATUS"].isin(recurrence_events)]
        if len(recur_rows) > 0:
            first = recur_rows.iloc[0]
            label = 1
            followup_days = first["START_DATE"]
            rec_date = first["START_DATE"]
            rec_type = first["STATUS"]
            src = "status_df"
            stats["recurrence_from_status"] += 1
            results.append({
                "PATIENT_ID": pid,
                "LABEL": label,
                "FOLLOW_UP_DAYS": followup_days,
                "RECURRENCE_DATE": rec_date,
                "RECURRENCE_TYPE": rec_type,
                "SOURCE": src
            })
            continue

        # ------------------------------
        # Step 1b: Recurrence via treatment_df
        # ------------------------------
        if treatment_df is not None:
            t_df = treatment_df[treatment_df["PATIENT_ID"] == pid]
            if len(t_df) > 0:
                site_recur = t_df[t_df["ANATOMIC_TREATMENT_SITE"].isin(treatment_recur_sites)]
                regimen_recur = t_df[t_df["REGIMEN_INDICATION"] == "Recurrence"]
                combined = pd.concat([site_recur.head(1), regimen_recur.head(1)])
                if len(combined) > 0:
                    first_recur = combined.sort_values("START_DATE").iloc[0]
                    label = 1
                    followup_days = first_recur["START_DATE"]
                    rec_date = first_recur["START_DATE"]
                    rec_type = (
                        f"Treatment {first_recur['ANATOMIC_TREATMENT_SITE']}"
                        if pd.notna(first_recur.get("ANATOMIC_TREATMENT_SITE"))
                        else "treatment_df"
                    )
                    src = "treatment_df"
                    stats["recurrence_from_treatment"] += 1
                    results.append({
                        "PATIENT_ID": pid,
                        "LABEL": label,
                        "FOLLOW_UP_DAYS": followup_days,
                        "RECURRENCE_DATE": rec_date,
                        "RECURRENCE_TYPE": rec_type,
                        "SOURCE": src
                    })
                    continue

        # ------------------------------
        # Step 2: Exclusions
        # ------------------------------
        if df["STATUS"].isin(progression_events).any():
            label = np.nan
            src = "excluded_progression"
            stats["excluded_progression_events"] += 1
            exclusion_reason[pid] = "progression event in STATUS"
            results.append({
                "PATIENT_ID": pid,
                "LABEL": label,
                "FOLLOW_UP_DAYS": followup_days,
                "RECURRENCE_DATE": None,
                "RECURRENCE_TYPE": None,
                "SOURCE": src
            })
            continue

        lf_rows = df[df["STATUS"] == "Last Follow Up"]
        if len(lf_rows) == 0:
            label = np.nan
            src = "excluded_no_lfu"
            stats["excluded_no_last_followup"] += 1
            exclusion_reason[pid] = "no last follow up"
            results.append({
                "PATIENT_ID": pid,
                "LABEL": label,
                "FOLLOW_UP_DAYS": followup_days,
                "RECURRENCE_DATE": None,
                "RECURRENCE_TYPE": None,
                "SOURCE": src
            })
            continue

        last_follow = lf_rows.iloc[-1]
        followup_days = last_follow["START_DATE"]
        if followup_days < min_followup_days:
            label = np.nan
            src = "excluded_short_followup"
            stats["excluded_short_followup"] += 1
            exclusion_reason[pid] = "short followup"
            results.append({
                "PATIENT_ID": pid,
                "LABEL": label,
                "FOLLOW_UP_DAYS": followup_days,
                "RECURRENCE_DATE": None,
                "RECURRENCE_TYPE": None,
                "SOURCE": src
            })
            continue

        # ------------------------------
        # Step 3: Tumor/therapy labeling
        # ------------------------------
        ptos = last_follow.get("PRIMARY_THERAPY_OUTCOME_SUCCESS", np.nan)
        tumor = last_follow.get("TUMOR_STATUS", np.nan)
        if pd.notna(ptos) and ptos == "Complete Remission/Response":
            label = 0
            src = "no_recur_CRR"
            stats["no_recur_CRR"] += 1
        elif pd.notna(ptos):
            label = np.nan
            src = "excluded_not_complete_remission"
            stats["excluded_not_complete_remission"] += 1
            exclusion_reason[pid] = "excluded_not_complete_remission"
        elif tumor == "Tumor Free":
            label = 0
            src = "no_recur_tumor_free_fallback"
            stats[src] = "no_recur_tumor_free_fallback"
        else:
            label = np.nan
            src = "excluded_not_complete_remission"
            stats["excluded_not_complete_remission"] += 1
            exclusion_reason[pid] = "excluded_not_complete_remission"


        results.append({
            "PATIENT_ID": pid,
            "LABEL": label,
            "FOLLOW_UP_DAYS": followup_days,
            "RECURRENCE_DATE": None,
            "RECURRENCE_TYPE": None,
            "SOURCE": src
        })

    # ------------------------------
    # Step 4: DFS conflict handling
    # ------------------------------
    labels_df = pd.DataFrame(results)
    if clinical_df is not None:
        merged = labels_df.merge(
            clinical_df[["PATIENT_ID", "DFS_STATUS"]],
            on="PATIENT_ID",
            how="left"
        )
        
        # Identify DFS conflicts
        conflict_mask = (
            ((merged["LABEL"] == 1) & (merged["DFS_STATUS"] == "0:DiseaseFree")) |
            ((merged["LABEL"] == 0) & (merged["DFS_STATUS"] == "1:Recurred/Progressed"))
        )
        
        # Update stats and reasons
        conflict_pids = merged.loc[conflict_mask, "PATIENT_ID"].tolist()
        for pid in conflict_pids:
            stats["excluded_DFS_conflict"] += 1
            exclusion_reason[pid] = "DFS conflict"
        
        # Instead of removing, set LABEL to NaN and SOURCE to 'DFS_conflict'
        merged.loc[conflict_mask, "LABEL"] = np.nan
        merged.loc[conflict_mask, "SOURCE"] = "DFS_conflict"
    
        labels_df = merged[labels_df.columns]

    return labels_df, stats, exclusion_reason



# def generate_recurrence_labels(treatment_file, status_file, clinical_file):
#     """
#     Generates a pd.Series of recurrence labels for all patients.
    
#     Label rules:
#      1 (recurred): 
#         * ANATOMIC_TREATMENT_SITE = "Local Recurrence" or "Distant Recurrence"
#         * REGIMEN_INDICATION = "Recurrence"
#         * STATUS = "Locoregional Recurrence"
#         * NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT = "Yes"
#      0 (no recurrence): 
#         * NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT = "No"
#         * AND no other columns show recurrence
#      None (unknown/ambiguous): 
#         * All other patients
#         * Patients with conflicting signals (e.g., "No" in clinical but positive elsewhere)
#     """
    
#     # --- Load data ---
#     df_treatment = pd.read_csv(treatment_file, sep="\t", comment="#", low_memory=False)
#     df_status = pd.read_csv(status_file, sep="\t", comment="#", low_memory=False)
#     df_clinical = pd.read_csv(clinical_file, sep="\t", comment="#", low_memory=False)
    
#     # Ensure PATIENT_ID is a column
#     if df_treatment.index.name == "PATIENT_ID":
#         df_treatment = df_treatment.reset_index()
#     if df_clinical.index.name == "PATIENT_ID":
#         df_clinical = df_clinical.reset_index()
    
#     # --- Set of patient IDs labeled as recurrence ---
#     recur_patients = set()
    
#     # From treatment file
#     treatment_mask = df_treatment["ANATOMIC_TREATMENT_SITE"].isin(["Local Recurrence", "Distant Recurrence"])
#     regimen_mask = df_treatment["REGIMEN_INDICATION"] == "Recurrence"
#     recur_patients.update(df_treatment.loc[treatment_mask | regimen_mask, "PATIENT_ID"].unique())
    
#     # From status file
#     status_mask = df_status["STATUS"].astype(str).str.strip() == "Locoregional Recurrence"
#     recur_patients.update(df_status.loc[status_mask, "PATIENT_ID"].unique())
    
#     # From clinical file
#     clinical_yes_mask = df_clinical["NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"].astype(str).str.strip().str.lower() == "yes"
#     recur_patients.update(df_clinical.loc[clinical_yes_mask, "PATIENT_ID"].unique())
    
#     # --- Set of patients labeled as no recurrence ---
#     clinical_no_mask = df_clinical["NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"].astype(str).str.strip().str.lower() == "no"
#     no_recur_patients = set(df_clinical.loc[clinical_no_mask, "PATIENT_ID"].unique())
    
#     # --- Combine all patient IDs ---
#     all_patients = set(df_clinical["PATIENT_ID"]) | set(df_treatment["PATIENT_ID"]) | set(df_status["PATIENT_ID"])
    
#     # --- Assign labels ---
#     labels = {}
#     for pid in all_patients:
#         if pid in recur_patients and pid in no_recur_patients:
#             # conflict: one source says no, another says yes
#             labels[pid] = None
#         elif pid in recur_patients:
#             labels[pid] = 1
#         elif pid in no_recur_patients:
#             labels[pid] = 0
#         else:
#             labels[pid] = None
    
#     # Return as pd.Series
#     label_series = pd.Series(labels, name="Recurrence_Label")
#     label_series.index.name = "PATIENT_ID"
    
#     return label_series

def ensure_patient_id_index(df):
    """
    Ensures the DataFrame uses PATIENT_ID as its index.
    Works whether PATIENT_ID is already the index or a column.
    """
    if "PATIENT_ID" in df.columns:
        df = df.set_index("PATIENT_ID")
        return df
    else:
        return df


def drop_patients_missing_data(clinical_df, mrna_df, mutation_df, labels):
    """
    Drops patients not shared across clinical_df, mrna_df, mutation_df, and labels.
    Also drops patients missing labeling data (None or NaN).

    Returns:
        clinical_df_clean, mrna_df_clean, mutation_df_clean, labels_clean
    """
    # Step 1: Find shared patient IDs (preserve order)
    clinical_df = ensure_patient_id_index(clinical_df)
    mrna_df = ensure_patient_id_index(mrna_df)
    mutation_df = ensure_patient_id_index(mutation_df)
    shared_patients = (
        clinical_df.index
        .intersection(mrna_df.index)
        .intersection(mutation_df.index)
        .intersection(labels.index)
    )

    # Step 2: Subset all to shared patients, in the same order
    clinical_df_clean = clinical_df.loc[shared_patients].copy()
    mrna_df_clean = mrna_df.loc[shared_patients].copy()
    mutation_df_clean = mutation_df.loc[shared_patients].copy()
    labels_clean = labels.loc[shared_patients].copy()

    # Step 3: Drop patients with missing labels (None/NaN)
    valid_patients = labels_clean[labels_clean.notna()].index
    clinical_df_clean = clinical_df_clean.loc[valid_patients]
    mrna_df_clean = mrna_df_clean.loc[valid_patients]
    mutation_df_clean = mutation_df_clean.loc[valid_patients]
    labels_clean = labels_clean.loc[valid_patients]

    # Step 4: Sanity checks
    n_patients = clinical_df_clean.shape[0]
    assert (
        n_patients == mrna_df_clean.shape[0] == mutation_df_clean.shape[0] == labels_clean.shape[0]
    ), "Dataframes have different number of patients after cleaning"

    assert not labels_clean.isna().any(), "Found unlabeled patient after cleaning"
    assert (
        clinical_df_clean.index.equals(mrna_df_clean.index)
        and clinical_df_clean.index.equals(mutation_df_clean.index)
        and clinical_df_clean.index.equals(labels_clean.index)
    ), "Indexes are not aligned"

    return clinical_df_clean, mrna_df_clean, mutation_df_clean, labels_clean


def stratified_split_with_balance_check(
    df, y, clinical_cols, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE,
    max_attempts=config.STRATIFICATION_MAX_ATTEMPTS, p_thresh=config.P_VALUE_STRATIFICATION, random_state=config.SEED
):
    """
    Split the dataset into train/val/test (70/15/15) stratified by y,
    and ensure clinical features are balanced across splits.

    A split is rejected if ANY clinical feature has p < p_thresh.
    """

    for attempt in range(max_attempts):
        print("on attempt", attempt)
        
        # Step 1: split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            df, y, test_size=test_size, stratify=y, random_state=(random_state + attempt)
        )

        # Step 2: split trainval into train and val
        rel_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=rel_val_size,
            stratify=y_trainval, random_state=attempt
        )

        # Step 3: test for imbalance across splits for clinical features
        p_values = []
        reject_split = False

        for feature in clinical_cols:
            print("feature:", feature)

            # Handle numeric features
            if np.issubdtype(df[feature].dtype, np.number):
                vals = [
                    X_train[feature].dropna(),
                    X_val[feature].dropna(),
                    X_test[feature].dropna()
                ]
                pairs = [(0, 1), (0, 2), (1, 2)]
                for (i, j) in pairs:
                    if len(vals[i]) > 0 and len(vals[j]) > 0:
                        if vals[i].nunique() > 1 or vals[j].nunique() > 1:
                            try:
                                _, p = mannwhitneyu(vals[i], vals[j], alternative='two-sided')
                                print(f"  numerical ({i}-{j}) p =", p)
                                p_values.append(p)
                                if p < p_thresh:
                                    reject_split = True
                            except Exception as e:
                                print(f"  numerical test failed ({i}-{j}):", e)

            # Handle categorical features
            else:
                tmp = pd.concat([
                    X_train.assign(split='train'),
                    X_val.assign(split='val'),
                    X_test.assign(split='test')
                ])
                contingency = pd.crosstab(tmp[feature], tmp['split'])
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    try:
                        _, p, _, _ = chi2_contingency(contingency)
                        print(f"  categorical p =", p)
                        p_values.append(p)
                        if p < p_thresh:
                            reject_split = True
                    except Exception as e:
                        print("  categorical test failed:", e)
                else:
                    print("  categorical skipped (not enough variation)")

        # Step 4: if any p < threshold, reject split
        if not reject_split:
            print(f"Balanced split achieved after {attempt+1} attempts.")
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            print(f"Rejected split {attempt+1} due to imbalance (min p = {min(p_values):.4g})")

    print("Could not achieve balance after max attempts.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_split_data(clinical_patient_file=config.CLINICAL_DATA_PATH,
                        clinical_sample_file=config.SAMPLE_CLINICAL_DATA_PATH,
                        mrna_file=config.MRNA_DATA_PATH,
                        mutation_file=config.MUTATION_DATA_PATH,
                        treatment_file=config.TREATMENT_DATA_PATH,
                        status_file=config.STATUS_DATA_PATH,
                        clin_cols_to_stratify_on=config.CLIN_COLS_TO_STRATIFY_ON,
                        test_size=config.TEST_SIZE,
                        val_size=config.VAL_SIZE,
                        max_attempts=config.STRATIFICATION_MAX_ATTEMPTS,
                        p_thresh=config.P_VALUE_STRATIFICATION,
                        random_state=config.SEED):
    """loads data, generates labels, drops patients missing from any dataset, 
    and splits into train/validation/and test sets.
    Returns:"""
    clinical_df = load_clinical_data(clinical_patient_file, clinical_sample_file, status_file)
    mrna_df = load_mrna_data(mrna_file)
    mutation_df = load_mutation_data(mutation_file)

    status_df = pd.read_csv(
        status_file,
        sep="\t",
        comment="#",
        low_memory=False
    )
    
    treatment_df = pd.read_csv(
        treatment_file,
        sep="\t",
        comment="#",   # skip header comments if present
        low_memory=False
    )
    
    labels_df, _, _ = label_patients_with_stats(
        clinical_df=clinical_df,
        status_df=status_df,
        treatment_df=treatment_df,
    )
    labels_df = labels_df.set_index("PATIENT_ID")
    labels = pd.Series(labels_df["LABEL"])
    clinical_df, mrna_df, mutation_df, labels = drop_patients_missing_data(clinical_df, mrna_df, mutation_df, labels)

    clinical_cols = clinical_df.columns.tolist()
    mrna_cols = mrna_df.columns.tolist()
    mutation_cols = mutation_df.columns.tolist()

    full_df = clinical_df.join(mrna_df, how="inner").join(mutation_df, how="inner")
    labels = labels.reindex(full_df.index)


    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_with_balance_check(
        df=full_df,
        y=labels,
        clinical_cols=clin_cols_to_stratify_on,
        test_size=test_size,
        val_size=val_size,
        max_attempts=max_attempts,
        p_thresh=p_thresh,
        random_state=random_state
    )
    return (X_train, y_train, X_val, y_val, X_test, y_test, clinical_cols, mrna_cols, mutation_cols)

#### Preprocessors ######################################################

class BasePreprocessor:
    def __init__(self, max_null_frac, uniform_thresh):
        self.max_null_frac = max_null_frac
        self.uniform_thresh = uniform_thresh
        self.removed_cols_ = []
        self.columns_ = None
    
    def _drop_high_null_columns(self, X):
        """Drop columns with too many nulls (>max_null_frac)."""
        high_null_cols = [c for c in X.columns if X[c].isna().mean() > self.max_null_frac]
        return X.drop(columns=high_null_cols, errors="ignore"), high_null_cols
    
    def _drop_highly_uniform_columns(self, X):
        """Drop columns where a single value dominates."""
        cols_to_drop = []
        for col in X.columns:
            non_na = X[col].dropna()
            if not non_na.empty:
                top_freq = non_na.value_counts(normalize=True).iloc[0]
                if top_freq > self.uniform_thresh:
                    cols_to_drop.append(col)
        return X.drop(columns=cols_to_drop, errors="ignore"), cols_to_drop    
        

class ClinicalPreprocessor(BasePreprocessor):
    def __init__(self, cols_to_remove=config.CLINICAL_COLS_TO_REMOVE, categorical_cols=config.CATEGORICAL_COLS, max_null_frac=config.MAX_NULL_FRAC, uniform_thresh=config.UNIFORM_THRESHOLD):
        super().__init__(max_null_frac=max_null_frac, uniform_thresh=uniform_thresh)
        self.cols_to_remove = cols_to_remove
        self.categorical_cols = categorical_cols
        
        # Saved state after fit
        self.removed_cols_ = []
        self.columns_ = None  # final column order
        self.num_fill_values_ = {}
        self.cat_fill_values_ = {}
        
    def fit(self, X, y=None):
        # --- Step 1. Drop specified columns
        removed = [c for c in self.cols_to_remove if c in X.columns]
        
        # --- Step 2. Drop columns with too many nulls
        thresh = len(X) * (1 - self.max_null_frac)
        high_null_cols = [c for c in X.columns if X[c].isna().sum() > len(X) - thresh]
        removed.extend(high_null_cols)
        
        # --- Step 3. Drop highly uniform columns
        modified_df, cols_to_remove = self._drop_highly_uniform_columns(X)

        removed.extend(cols_to_remove)

        # --- Step 4. Drop all identified columns
        X = X.drop(columns=removed)
        
        # --- Step 5. Fill NaNs
        # Numerical → median
        numeric_cols = X.select_dtypes(include=['number']).columns
        self.num_fill_values_ = X[numeric_cols].median()
        X[numeric_cols] = X[numeric_cols].fillna(self.num_fill_values_)
        
        # Categorical → mode
        cat_cols = [c for c in self.categorical_cols if c in X.columns]
        self.cat_fill_values_ = {c: X[c].mode().iloc[0] for c in cat_cols if not X[c].dropna().empty}
        for c, mode_val in self.cat_fill_values_.items():
            X[c] = X[c].fillna(mode_val)
        
        # --- Step 6. One-hot encode categorical
        X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # Save results
        self.removed_cols_ = removed
        self.columns_ = X_enc.columns.tolist()
        
        return self
    
    def transform(self, X):
        # Drop removed cols
        X = X.drop(columns=[c for c in self.removed_cols_ if c in X.columns])
        
        # --- Fill NaNs using training fill values
        numeric_cols = X.select_dtypes(include=['number']).columns
        for c in numeric_cols:
            if c in self.num_fill_values_:
                X[c] = X[c].fillna(self.num_fill_values_[c])
        
        cat_cols = [c for c in self.categorical_cols if c in X.columns]
        for c in cat_cols:
            if c in self.cat_fill_values_:
                X[c] = X[c].fillna(self.cat_fill_values_[c])
        
        # One-hot encode
        X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # Reindex to training columns (fill missing with 0)
        X_enc = X_enc.reindex(columns=self.columns_, fill_value=0)
        
        return X_enc


class MrnaPreprocessor(BasePreprocessor):
    def __init__(self,
            max_null_frac=config.MAX_NULL_FRAC,
            uniform_thresh=config.UNIFORM_THRESHOLD,
            random_state=config.SEED):
        super().__init__(max_null_frac=max_null_frac, uniform_thresh=uniform_thresh)

        self.random_state = random_state

        # Saved state after fit
        self.removed_cols_ = []
        self.medians_ = {}
        self.columns_ = None
        self.selection_freq_ = None


    def fit(self, X, y=None):
        removed = []

        # Drop columns with too many nulls
        high_null_cols = [c for c in X.columns if X[c].isna().sum() > len(X) * self.max_null_frac]
        removed.extend(high_null_cols)
        X_temp = X.drop(columns=high_null_cols, errors="ignore")
        
        if X_temp.isna().any().any():
            raise ValueError("NaN values detected in X_temp, expected none.")

        # Save final state
        self.removed_cols_ = list(set(removed))
        self.columns_ = X_temp.columns.tolist()

        return self

    def transform(self, X):
        # Drop known removed cols
        X = X.drop(columns=[c for c in self.removed_cols_ if c in X.columns], errors="ignore")

        # Reorder X to match training column order
        X = X[self.columns_]

        return X


class MutationPreprocessor(BasePreprocessor):
    def __init__(self,
                max_mutation_count=config.MAX_MUTATION_COUNT,
                uniform_thresh=config.MUTATION_UNIFORM_THRESH
                 ):
        super().__init__(max_null_frac=0, uniform_thresh=uniform_thresh)

        # Saved state after fit
        self.max_mutation_count = max_mutation_count
        self.removed_cols_ = []
        self.medians_ = {}
        self.columns_ = None
        self.selection_freq_ = None

    def fit(self, X, y=None):
        removed = []

        # Clip mutation counts above 10 because these are often due to passenger genes, not valueable information
        X = X.clip(upper=self.max_mutation_count)

        # Drop highly uniform columns
        X_temp, uniform_cols = self._drop_highly_uniform_columns(X)
        removed.extend(uniform_cols)

        if X.isna().any().any():
            raise ValueError("NaN values detected in mutation_df, expected none.")

        # Save final state
        self.removed_cols_ = list(set(removed))
        self.columns_ = X_temp.columns.tolist()

        return self

    def transform(self, X):
        # Drop known removed cols
        X = X.drop(columns=[c for c in self.removed_cols_ if c in X.columns], errors="ignore")

        # Check column alignment
        missing = set(self.columns_) - set(X.columns)
        extra = set(X.columns) - set(self.columns_)
        if missing or extra:
            raise ValueError(
                f"Column mismatch! Missing: {missing}, Extra: {extra}, "
                f"{len(missing)} missing, {len(extra)} extra"
            )

        # Reorder X to match training column order
        X = X[self.columns_]

        return X

class ClinicalPreprocessorWrapper(ClinicalPreprocessor, BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")  # make sure fit() was called
        return np.array(self.columns_)  # or self.cleaned_columns_ if you store them


class MrnaPreprocessorWrapper(MrnaPreprocessor, BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")  # make sure fit() was called
        return np.array(self.columns_)  # or self.cleaned_columns_ if you store them

class MutationPreprocessorWrapper(MutationPreprocessor, BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")  # make sure fit() was called
        return np.array(self.columns_)  # or self.cleaned_columns_ if you store them

#### Feature Selection ######################################################

class BootstrappedSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=config.K, n_bootstrap=config.N_BOOTS_KBEST, threshold=config.THRESHOLD_KBEST, random_state=None):
        """
        Parameters
        ----------
        k : int
            Number of features to select per bootstrap.
        n_bootstrap : int
            Number of bootstrap resamples.
        threshold : float (0-1)
            Minimum fraction of bootstraps a feature must appear in to be kept.
        random_state : int, optional
            Random seed.
        """
        self.k = k
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        feature_counts = pd.Series(0, index=X.columns, dtype=int)

        # Run bootstraps
        for i in range(self.n_bootstrap):
            X_res, y_res = resample(X, y, replace=True, random_state=rng.randint(1e6))
            selector = SelectKBest(score_func=f_classif, k=self.k)
            selector.fit(X_res, y_res)
            selected = X.columns[selector.get_support()]
            feature_counts[selected] += 1

        # Compute frequencies
        self.feature_freq_ = feature_counts / self.n_bootstrap
        # Keep only stable features
        self.selected_features_ = self.feature_freq_[self.feature_freq_ >= self.threshold].index.tolist()

        return self

    def transform(self, X):
        # If no features survive threshold, fall back to top-k overall
        if len(self.selected_features_) == 0:
            self.selected_features_ = self.feature_freq_.sort_values(ascending=False).head(self.k).index.tolist()
        return X[self.selected_features_]

    def get_support(self):
        """Boolean mask of selected features (like SelectKBest)."""
        return [col in self.selected_features_ for col in self.feature_freq_.index]


class StabilitySelection(BaseEstimator, TransformerMixin):
    def __init__(self, n_boots=100, fpr_alpha=0.05, stability_threshold=0.5, random_state=None):
        """
        Stability-based feature selection with automatic test type detection.

        - chi2 for categorical/binary features (nonnegative)
        - f_classif for continuous numeric features
        """
        self.n_boots = n_boots
        self.fpr_alpha = fpr_alpha
        self.stability_threshold = stability_threshold
        self.random_state = random_state

    def _split_feature_types(self, X):
        """Split columns into categorical (chi2) and numerical (f_classif)."""
        cat_cols, num_cols = [], []
        for col in X.columns:
            vals = X[col].dropna().unique()
            if X[col].dtype == bool or len(vals) <= 2:
                cat_cols.append(col)
            elif np.issubdtype(X[col].dtype, np.number):
                num_cols.append(col)
            else:
                # fallback for object/string cols
                cat_cols.append(col)
        return cat_cols, num_cols

    def _bootstrap_select(self, X, y, cols, score_func):
        """Perform stability selection on a subset of features using given score_func."""
        feature_counts = pd.Series(0, index=cols, dtype=int)

        for i in range(self.n_boots):
            X_boot, y_boot = resample(
                X[cols], y,
                stratify=y,
                n_samples=len(y),
                replace=True,
                random_state=(self.random_state + i) if self.random_state is not None else None
            )

            # chi2 requires nonnegative values
            if score_func == chi2:
                X_boot = X_boot.clip(lower=0)

            selector = SelectFpr(score_func=score_func, alpha=self.fpr_alpha)
            try:
                selector.fit(X_boot, y_boot)
                selected = X_boot.columns[selector.get_support()]
                feature_counts[selected] += 1
            except Exception as e:
                print(f"Skipping bootstrap {i} for {score_func.__name__}: {e}")
                continue

        return feature_counts

    def fit(self, X, y):
        np.random.seed(self.random_state)

        # Split features by type
        cat_cols, num_cols = self._split_feature_types(X)

        # Run separate stability selection for categorical and numerical features
        feature_counts = pd.Series(0, index=X.columns, dtype=int)
        if cat_cols:
            feature_counts[cat_cols] += self._bootstrap_select(X, y, cat_cols, chi2)
        if num_cols:
            feature_counts[num_cols] += self._bootstrap_select(X, y, num_cols, f_classif)

        # Compute frequency of selection
        self.selection_freq_ = feature_counts / self.n_boots
        self.selected_features_ = self.selection_freq_[self.selection_freq_ >= self.stability_threshold].index.tolist()
        return self

    def transform(self, X):
        return X.loc[:, X.columns.intersection(self.selected_features_)]

    def get_support(self):
        """Boolean mask of selected features aligned with input order."""
        return [col in self.selected_features_ for col in self.selection_freq_.index]
