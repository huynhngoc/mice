import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score

# load the predictions
val_preds = pd.read_csv('analysis_results/val_preds.csv')
test_preds = pd.read_csv('analysis_results/test_preds.csv')

# ensemble the predictions for test set
ensemble_preds = test_preds.groupby(['group', 'model', 'learning_rate', 'patient_idx', 'slice_idx']).agg({'y': 'mean', 'predicted': 'mean'}).reset_index()
# ensemble_preds.to_csv('analysis_results/ensemble_preds.csv', index=False)
# load the foreground info
info = pd.read_csv('data_info/foreground_info.csv')[['pid', 'slice', 'huang_foreground_ratio', 'huang_foreground_normalized']]
info.columns = ['patient_idx', 'slice_idx', 'huang_foreground_ratio', 'huang_foreground_normalized']

# add the foreground info to the val, test, ensemble predictions
val_preds = val_preds.merge(info, on=['patient_idx', 'slice_idx'], how='left')
test_preds = test_preds.merge(info, on=['patient_idx', 'slice_idx'], how='left')
ensemble_preds = ensemble_preds.merge(info, on=['patient_idx', 'slice_idx'], how='left')

def calculate_metrics(group):
    y_true = group['y']
    y_pred = group['predicted']
    y_pred_binary = (y_pred > 0.5).astype(float)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'auc': roc_auc_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred_binary),
        'f1_0': f1_score(y_true, y_pred_binary, pos_label=0),
        'scaled_mcc': matthews_corrcoef(y_true, y_pred_binary)/2 + 0.5,
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary),
        'specificity': recall_score(y_true, y_pred_binary, pos_label=0),
    }
    return pd.Series(metrics)

def calculate_patient_prediction(group):
    y_pred = group['predicted']
    huang_ratio = group['huang_foreground_ratio']
    huang_ratio_sum = huang_ratio.sum()
    pred = {
        'y': group['y'].mean(),
        'mean_pred': y_pred.mean(),
        'weighted_pred': (y_pred * huang_ratio).sum() / huang_ratio_sum,
    }
    return pd.Series(pred)

def get_metric_calculator(y_name='y', pred_name='predicted'):
    def calculate_metrics(group):
        y_true = group[y_name]
        y_pred = group[pred_name]
        y_pred_binary = (y_pred > 0.5).astype(float)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'auc': roc_auc_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred_binary),
            'f1_0': f1_score(y_true, y_pred_binary, pos_label=0),
            'scaled_mcc': matthews_corrcoef(y_true, y_pred_binary)/2 + 0.5,
            'precision': precision_score(y_true, y_pred_binary),
            'recall': recall_score(y_true, y_pred_binary),
            'specificity': recall_score(y_true, y_pred_binary, pos_label=0),
        }
        return pd.Series(metrics)
    return calculate_metrics

##########################################
# Calculate results for validation set
##########################################

# calculate result per slice
val_experiment_results = val_preds.groupby(['group', 'model_name', 'model', 'learning_rate']
    ).apply(calculate_metrics).reset_index()
val_experiment_results['weighted_score'] = (val_experiment_results['accuracy'] + val_experiment_results['auc'] + val_experiment_results['f1'] + val_experiment_results['f1_0'] + val_experiment_results['scaled_mcc']) / 5

# save to file
val_experiment_results.to_csv('analysis_results/val_experiment_results_slice.csv', index=False)
val_experiment_results.groupby(['group', 'model', 'learning_rate']).mean().reset_index().to_csv('analysis_results/val_results_slice.csv', index=False)

# calculate results per patient
val_patient_preds = val_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'patient_idx']
    ).apply(calculate_patient_prediction).reset_index()

# get patient prediction by mean
val_experiment_results_patients = val_patient_preds.groupby(
    ['group', 'model_name', 'model', 'learning_rate']).apply(
        get_metric_calculator('y', 'mean_pred')).reset_index()
val_experiment_results_patients['weighted_score'] = (val_experiment_results_patients['accuracy'] + val_experiment_results_patients['auc'] + val_experiment_results_patients['f1'] + val_experiment_results_patients['f1_0'] + val_experiment_results_patients['scaled_mcc']) / 5
# save to file
val_experiment_results_patients.to_csv('analysis_results/val_experiment_results_patients_mean.csv', index=False)
val_experiment_results_patients.groupby(['group', 'model', 'learning_rate']).mean().reset_index().to_csv('analysis_results/val_results_patients_mean.csv', index=False)

# get patient prediction by weighted score
val_experiment_results_patients = val_patient_preds.groupby(
    ['group', 'model_name', 'model', 'learning_rate']).apply(
        get_metric_calculator('y', 'weighted_pred')).reset_index()
val_experiment_results_patients['weighted_score'] = (val_experiment_results_patients['accuracy'] + val_experiment_results_patients['auc'] + val_experiment_results_patients['f1'] + val_experiment_results_patients['f1_0'] + val_experiment_results_patients['scaled_mcc']) / 5
# save to file
val_experiment_results_patients.to_csv('analysis_results/val_experiment_results_patients_weighted.csv', index=False)
val_experiment_results_patients.groupby(['group', 'model', 'learning_rate']).mean().reset_index().to_csv('analysis_results/val_results_patients_weighted.csv', index=False)


##########################################
# Calculate results for test set
##########################################

# calculate result per slice
test_experiment_results = test_preds.groupby(['group', 'model_name', 'model', 'learning_rate']
    ).apply(calculate_metrics).reset_index()
test_experiment_results['weighted_score'] = (test_experiment_results['accuracy'] + test_experiment_results['auc'] + test_experiment_results['f1'] + test_experiment_results['f1_0'] + test_experiment_results['scaled_mcc']) / 5

# save to file
test_experiment_results.to_csv('analysis_results/test_experiment_results_slice.csv', index=False)
test_experiment_results.groupby(['group', 'model', 'learning_rate']).mean().reset_index().to_csv('analysis_results/test_results_slice.csv', index=False)

# calculate results per patient
test_patient_preds = test_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'patient_idx']
    ).apply(calculate_patient_prediction).reset_index()

# get patient prediction by mean
test_experiment_results_patients = test_patient_preds.groupby(
    ['group', 'model_name', 'model', 'learning_rate']).apply(
        get_metric_calculator('y', 'mean_pred')).reset_index()
test_experiment_results_patients['weighted_score'] = (test_experiment_results_patients['accuracy'] + test_experiment_results_patients['auc'] + test_experiment_results_patients['f1'] + test_experiment_results_patients['f1_0'] + test_experiment_results_patients['scaled_mcc']) / 5
# save to file
test_experiment_results_patients.to_csv('analysis_results/test_experiment_results_patients_mean.csv', index=False)
test_experiment_results_patients.groupby(['group', 'model', 'learning_rate']).mean().reset_index().to_csv('analysis_results/test_results_patients_mean.csv', index=False)

# get patient prediction by weighted score
test_experiment_results_patients = test_patient_preds.groupby(
    ['group', 'model_name', 'model', 'learning_rate']).apply(
        get_metric_calculator('y', 'weighted_pred')).reset_index()
test_experiment_results_patients['weighted_score'] = (test_experiment_results_patients['accuracy'] + test_experiment_results_patients['auc'] + test_experiment_results_patients['f1'] + test_experiment_results_patients['f1_0'] + test_experiment_results_patients['scaled_mcc']) / 5
# save to file
test_experiment_results_patients.to_csv('analysis_results/test_experiment_results_patients_weighted.csv', index=False)
test_experiment_results_patients.groupby(['group', 'model', 'learning_rate']).mean().reset_index().to_csv('analysis_results/test_results_patients_weighted.csv', index=False)

##########################################
# Calculate results for ensemble set
##########################################

# calculate result per slice
ensemble_results = ensemble_preds.groupby(['group', 'model', 'learning_rate']
    ).apply(calculate_metrics).reset_index()
ensemble_results['weighted_score'] = (ensemble_results['accuracy'] + ensemble_results['auc'] + ensemble_results['f1'] + ensemble_results['f1_0'] + ensemble_results['scaled_mcc']) / 5

# save to file
ensemble_results.to_csv('analysis_results/ensemble_results_slice.csv', index=False)

# calculate results per patient
ensemble_patient_preds = ensemble_preds.groupby(['group', 'model', 'learning_rate', 'patient_idx']
    ).apply(calculate_patient_prediction).reset_index()

# get patient prediction by mean
ensemble_results_patients = ensemble_patient_preds.groupby(
    ['group', 'model', 'learning_rate']).apply(
        get_metric_calculator('y', 'mean_pred')).reset_index()
ensemble_results_patients['weighted_score'] = (ensemble_results_patients['accuracy'] + ensemble_results_patients['auc'] + ensemble_results_patients['f1'] + ensemble_results_patients['f1_0'] + ensemble_results_patients['scaled_mcc']) / 5
# save to file
ensemble_results_patients.to_csv('analysis_results/ensemble_results_patients_mean.csv', index=False)

# get patient prediction by weighted score
ensemble_results_patients = ensemble_patient_preds.groupby(
    ['group', 'model', 'learning_rate']).apply(
        get_metric_calculator('y', 'weighted_pred')).reset_index()
ensemble_results_patients['weighted_score'] = (ensemble_results_patients['accuracy'] + ensemble_results_patients['auc'] + ensemble_results_patients['f1'] + ensemble_results_patients['f1_0'] + ensemble_results_patients['scaled_mcc']) / 5
# save to file
ensemble_results_patients.to_csv('analysis_results/ensemble_results_patients_weighted.csv', index=False)
