import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load the data
val_results = pd.read_csv('analysis_results/val_results_slice.csv')
test_results = pd.read_csv('analysis_results/test_results_slice.csv')
ensemble_results = pd.read_csv('analysis_results/ensemble_results_slice.csv')

val_results['model_name'] = val_results['model'] + '-' + val_results['learning_rate'].astype(str)
test_results['model_name'] = test_results['model'] + '-' + test_results['learning_rate'].astype(str)
ensemble_results['model_name'] = ensemble_results['model'] + '-' + ensemble_results['learning_rate'].astype(str)

# Check if the model index and model name match in all three files
if not (val_results['model_name'].equals(test_results['model_name']) and val_results['model_name'].equals(ensemble_results['model_name'])):
    raise ValueError("Model indices or names do not match across the files")

# # Extract the 'avg', 'model', and 'learning_rate' columns
# models = val_results['model_name']
# learning_rates = val_results['learning_rate']

# # Create labels for the x-axis
# labels = [f'{model}-{lr}' for model, lr in zip(models, learning_rates)]

# # Plot the results
# plt.figure(figsize=(10, 6))

# for i, model in enumerate(models):
#     val_avg = val_results.loc[val_results['model_name'] == model, 'weighted_score'].values[0]
#     test_avg = test_results.loc[test_results['model_name'] == model, 'weighted_score'].values[0]
#     ensemble_avg = ensemble_results.loc[ensemble_results['model_name'] == model, 'weighted_score'].values[0]
#     plt.plot([0, 1, 2], [val_avg, test_avg, ensemble_avg], label=model)

# # Add titles and labels
# plt.title('Model Improvement from Validation to Test to Ensemble')
# plt.xlabel('Phase')
# plt.ylabel('Average Score')
# plt.xticks(ticks=[0, 1, 2], labels=['Validation', 'Test', 'Ensemble'])
# plt.legend()

# # Show the plot
# plt.tight_layout()
plt.show()


# Sort and select the top 10 models from each DataFrame
top_val_results = val_results.nlargest(5, 'weighted_score')
top_test_results = test_results.nlargest(5, 'weighted_score')
top_ensemble_results = ensemble_results.nlargest(5, 'weighted_score')

# Get the unique top models
top_models = set(top_val_results['model_name']).union(set(top_test_results['model_name'])).union(set(top_ensemble_results['model_name']))

# Create labels for the x-axis
model_names = val_results['model_name']

# Plot the results
plt.figure(figsize=(10, 6))

for i, model in enumerate(model_names):
    if model in top_models:
        val_avg = val_results.loc[val_results['model_name'] == model, 'weighted_score'].values[0]
        test_avg = test_results.loc[test_results['model_name'] == model, 'weighted_score'].values[0]
        ensemble_avg = ensemble_results.loc[ensemble_results['model_name'] == model, 'weighted_score'].values[0]
        plt.plot([0, 1, 2], [val_avg, test_avg, ensemble_avg], label=f'{model}')

# Add titles and labels
plt.title('Model Improvement from Validation to Test to Ensemble')
plt.xlabel('Phase')
plt.ylabel('Average Score')
plt.xticks(ticks=[0, 1, 2], labels=['Validation', 'Test', 'Ensemble'])
plt.legend()

# Show the plot
# plt.tight_layout()
plt.show()



# Plot the results
plt.figure(figsize=(10, 6))

for i, model in enumerate(val_results.nlargest(10, 'weighted_score')['model_name']):
    if model in top_models:
        val_avg = val_results.loc[val_results['model_name'] == model, 'weighted_score'].values[0]
        test_avg = test_results.loc[test_results['model_name'] == model, 'weighted_score'].values[0]
        ensemble_avg = ensemble_results.loc[ensemble_results['model_name'] == model, 'weighted_score'].values[0]
        plt.plot([0, 1, 2], [val_avg, test_avg, ensemble_avg], label=f'{model}')

# Add titles and labels
plt.title('Model Improvement from Validation to Test to Ensemble')
plt.xlabel('Phase')
plt.ylabel('Average Score')
plt.xticks(ticks=[0, 1, 2], labels=['Validation', 'Test', 'Ensemble'])
plt.legend()

# Show the plot
# plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for plot_num, metrics in enumerate(['weighted_score', 'accuracy', 'scaled_mcc', 'f1', 'f1_0', 'auc']):
    plt.subplot(2,3,plot_num+1)
    for i, model in enumerate(val_results.nlargest(10, 'weighted_score')['model_name']):
        if model in top_models:
            val_avg = val_results.loc[val_results['model_name'] == model, metrics].values[0]
            test_avg = test_results.loc[test_results['model_name'] == model, metrics].values[0]
            ensemble_avg = ensemble_results.loc[ensemble_results['model_name'] == model, metrics].values[0]
            plt.plot([0, 1, 2], [val_avg, test_avg, ensemble_avg], label=f'{model}')

    # Add titles and labels
    plt.title(metrics)
    plt.xlabel('Phase')
    plt.ylabel(f'{metrics}')
    plt.xticks(ticks=[0, 1, 2], labels=['Validation', 'Test', 'Ensemble'])
plt.legend()
# Show the plot
plt.tight_layout()
plt.show()

#################################
# per patient results (mean)
val_patient_results = pd.read_csv('analysis_results/val_results_patients_mean.csv')
test_patient_results = pd.read_csv('analysis_results/test_results_patients_mean.csv')
ensemble_patient_results = pd.read_csv('analysis_results/ensemble_results_patients_mean.csv')

val_patient_results['model_name'] = val_results['model'] + '-' + val_results['learning_rate'].astype(str)
test_patient_results['model_name'] = test_results['model'] + '-' + test_results['learning_rate'].astype(str)
ensemble_patient_results['model_name'] = ensemble_results['model'] + '-' + ensemble_results['learning_rate'].astype(str)


plt.figure(figsize=(12, 6))
for plot_num, metrics in enumerate(['weighted_score', 'accuracy', 'scaled_mcc', 'f1', 'f1_0', 'auc']):
    plt.subplot(2,3,plot_num+1)
    for i, model in enumerate(val_results.nlargest(10, 'weighted_score')['model_name']):
        if model in top_models:
            val_avg = val_patient_results.loc[val_results['model_name'] == model, metrics].values[0]
            test_avg = test_patient_results.loc[test_results['model_name'] == model, metrics].values[0]
            ensemble_avg = ensemble_patient_results.loc[ensemble_results['model_name'] == model, metrics].values[0]
            plt.plot([0, 1, 2], [val_avg, test_avg, ensemble_avg], label=f'{model}')

    # Add titles and labels
    plt.title(metrics)
    plt.xlabel('Phase')
    plt.ylabel(f'{metrics}')
    plt.xticks(ticks=[0, 1, 2], labels=['Validation', 'Test', 'Ensemble'])
plt.legend()
# Show the plot
plt.tight_layout()
plt.show()

#################################
# per patient results (weighted)
val_patient_results = pd.read_csv('analysis_results/val_results_patients_weighted.csv')
test_patient_results = pd.read_csv('analysis_results/test_results_patients_weighted.csv')
ensemble_patient_results = pd.read_csv('analysis_results/ensemble_results_patients_weighted.csv')

val_patient_results['model_name'] = val_results['model'] + '-' + val_results['learning_rate'].astype(str)
test_patient_results['model_name'] = test_results['model'] + '-' + test_results['learning_rate'].astype(str)
ensemble_patient_results['model_name'] = ensemble_results['model'] + '-' + ensemble_results['learning_rate'].astype(str)


plt.figure(figsize=(12, 6))
for plot_num, metrics in enumerate(['weighted_score', 'accuracy', 'scaled_mcc', 'f1', 'f1_0', 'auc']):
    plt.subplot(2,3,plot_num+1)
    for i, model in enumerate(val_results.nlargest(5, 'weighted_score')['model_name']):
        if model in top_models:
            val_avg = val_patient_results.loc[val_results['model_name'] == model, metrics].values[0]
            test_avg = test_patient_results.loc[test_results['model_name'] == model, metrics].values[0]
            ensemble_avg = ensemble_patient_results.loc[ensemble_results['model_name'] == model, metrics].values[0]
            plt.plot([0, 1, 2], [val_avg, test_avg, ensemble_avg], label=f'{model}')

    # Add titles and labels
    plt.title(metrics)
    plt.xlabel('Phase')
    plt.ylabel(f'{metrics}')
    plt.xticks(ticks=[0, 1, 2], labels=['Validation', 'Test', 'Ensemble'])
plt.legend()
# Show the plot
plt.tight_layout()
plt.show()


#################################
# plot how different slice prediction on patients
val_results.nlargest(5, 'weighted_score')['model_name']
# 3     B1-0.0005
# 13    B6-0.0005
# 1     B0-0.0005
# 18     S-0.0001
# 6     B3-0.0001

# foreground info
foregound_info = pd.read_csv('data_info/foreground_info.csv')[['pid', 'slice', 'huang_foreground_normalized']]
foregound_info.columns = ['patient_idx', 'slice_idx', 'foreground']
# ensemble preds
ensemble_preds = pd.read_csv('analysis_results/ensemble_preds.csv')
ensemble_preds['model_name'] = ensemble_preds['model'] + '-' + ensemble_preds['learning_rate'].astype(str)
# merge
ensemble_preds = ensemble_preds.merge(foregound_info, on=['patient_idx', 'slice_idx'])
# if slice correctly predicted
ensemble_preds['correct'] = (ensemble_preds.predicted > 0.5) == ensemble_preds.y
# correlation
correct_by_slice = ensemble_preds[ensemble_preds.group == 'efficientnet'].groupby(['patient_idx', 'slice_idx', 'foreground']).agg({'correct': 'mean'}).reset_index()

spearmanr(correct_by_slice['foreground'], correct_by_slice['correct'])

spearmanr(ensemble_preds['foreground'], ensemble_preds['predicted'] * (-1)**(ensemble_preds['y'] + 1) + (1-ensemble_preds['y']))



selected_model = 'B3-0.0001'

selected_preds = ensemble_preds[ensemble_preds['model_name'] == selected_model]
plt.figure(figsize=(6, 8))
plt.subplot(2,1,1)
control_preds = selected_preds[selected_preds['y'] == 0]
all_y = []
for pid in np.unique(control_preds['patient_idx']):
    x = np.arange(1, 31)
    y = np.array([np.nan] * 30, dtype=float)
    selected_slices = control_preds[control_preds['patient_idx'] == pid]['slice_idx'].values
    y[selected_slices] = control_preds[control_preds['patient_idx'] == pid]['predicted'].values
    all_y.append(y)
    plt.plot(x, y, label=f'M{pid:02d}')
all_y = np.array(all_y)
mean_y = np.nanmean(all_y, axis=0)
std_y = np.nanstd(all_y, axis=0)
plt.plot(x, mean_y, label='Mean', color='blue', linewidth=2)
plt.fill_between(x, mean_y - std_y, mean_y + std_y, color='blue', alpha=0.2)
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.title('Control Mice')
plt.xlabel('Slice Index')
plt.ylabel('Predicted Probability')
plt.legend()

plt.subplot(2,1,2)
case_preds = selected_preds[selected_preds['y'] == 1]
all_y = []
for pid in np.unique(case_preds['patient_idx']):
    x = np.arange(1, 31)
    y = np.array([np.nan] * 30, dtype=float)
    selected_slices = case_preds[case_preds['patient_idx'] == pid]['slice_idx'].values
    y[selected_slices] = case_preds[case_preds['patient_idx'] == pid]['predicted'].values
    all_y.append(y)
    plt.plot(x, y, label=f'M{pid:02d}')
all_y = np.array(all_y)
mean_y = np.nanmean(all_y, axis=0)
std_y = np.nanstd(all_y, axis=0)
plt.plot(x, mean_y, label='Mean', color='blue', linewidth=2)
plt.fill_between(x, mean_y - std_y, mean_y + std_y, color='blue', alpha=0.2)
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.title('Irradiated Mice')
plt.xlabel('Slice Index')
plt.ylabel('Predicted Probability')
plt.legend()
plt.tight_layout()
plt.show()


########
# relationship between foreground and prediction
selected_model = 'B3-0.0001'

selected_preds = ensemble_preds[ensemble_preds['model_name'] == selected_model]
plt.figure(figsize=(6, 8))
plt.subplot(2,1,1)
control_preds = selected_preds[selected_preds['y'] == 0]
# all_y = []
for pid in np.unique(control_preds['patient_idx']):
    # x = np.arange(1, 31)
    # y = np.array([np.nan] * 30, dtype=float)
    x = control_preds[control_preds['patient_idx'] == pid]['foreground'].values
    max_index = np.argmax(x)
    x[max_index + 1:] = 2 - x[max_index + 1:]
    y = control_preds[control_preds['patient_idx'] == pid]['predicted'].values
    # all_y.append(y)
    plt.plot(x, y, label=f'M{pid:02d}')
# all_y = np.array(all_y)
# mean_y = np.nanmean(all_y, axis=0)
# std_y = np.nanstd(all_y, axis=0)
# plt.plot(x, mean_y, label='Mean', color='blue', linewidth=2)
# plt.fill_between(x, mean_y - std_y, mean_y + std_y, color='blue', alpha=0.2)
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.title('Control Mice')
plt.xlabel('Slice Index')
plt.ylabel('Predicted Probability')
plt.legend()

plt.subplot(2,1,2)
case_preds = selected_preds[selected_preds['y'] == 1]
# all_y = []
for pid in np.unique(case_preds['patient_idx']):
    # x = np.arange(1, 31)
    # y = np.array([np.nan] * 30, dtype=float)
    x = case_preds[case_preds['patient_idx'] == pid]['foreground'].values
    max_index = np.argmax(x)
    x[max_index + 1:] = 2 - x[max_index + 1:]
    y = case_preds[case_preds['patient_idx'] == pid]['predicted'].values
    # all_y.append(y)
    plt.plot(x, y, label=f'M{pid:02d}')
# all_y = np.array(all_y)
# mean_y = np.nanmean(all_y, axis=0)
# std_y = np.nanstd(all_y, axis=0)
# plt.plot(x, mean_y, label='Mean', color='blue', linewidth=2)
# plt.fill_between(x, mean_y - std_y, mean_y + std_y, color='blue', alpha=0.2)
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.title('Irradiated Mice')
plt.xlabel('Slice Index')
plt.ylabel('Predicted Probability')
plt.legend()
plt.tight_layout()
plt.show()

# scatter plot
selected_model = 'B3-0.0001'

selected_preds = ensemble_preds[ensemble_preds['model_name'] == selected_model]
plt.figure(figsize=(6, 8))
plt.subplot(2,1,1)
control_preds = selected_preds[selected_preds['y'] == 0]

for pid in np.unique(control_preds['patient_idx']):
    x = control_preds[control_preds['patient_idx'] == pid]['foreground'].values
    y = control_preds[control_preds['patient_idx'] == pid]['predicted'].values
    plt.scatter(x, y, label=f'M{pid:02d}')
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.title('Control Mice')
plt.xlabel('Foreground')
plt.ylabel('Predicted Probability')
plt.legend()

plt.subplot(2,1,2)
case_preds = selected_preds[selected_preds['y'] == 1]

for pid in np.unique(case_preds['patient_idx']):
    x = case_preds[case_preds['patient_idx'] == pid]['foreground'].values
    y = case_preds[case_preds['patient_idx'] == pid]['predicted'].values
    plt.scatter(x, y, label=f'M{pid:02d}')
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.title('Irradiated Mice')
plt.xlabel('Foreground')
plt.ylabel('Predicted Probability')
plt.legend()
plt.tight_layout()
plt.show()


# std as uncertainty
selected_patient_preds = selected_preds.groupby(['patient_idx']).agg({
    'predicted': ['mean', 'std'],
    'y': 'mean',
    'correct': 'sum'
}).reset_index()
selected_patient_preds.columns = ['_'.join(col).strip('_') for col in selected_patient_preds.columns[:-2].values] + [col[0] for col in selected_patient_preds.columns[-2:]]
plt.scatter(selected_patient_preds['patient_idx'], selected_patient_preds['predicted_mean'], s=selected_patient_preds['predicted_std']*100, c=selected_patient_preds['y'])
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.title('Predicted Probability with Uncertainty')
plt.xlabel('Patient Index')
plt.ylabel('Predicted Probability')
plt.legend()
plt.tight_layout()
plt.show()

selected_preds['entropy'] = -selected_preds['predicted'] * np.log(selected_preds['predicted'])
selected_preds['cross_entropy'] = -selected_preds['predicted'] * np.log2(selected_preds['predicted']) - (1 - selected_preds['predicted']) * np.log2(1 - selected_preds['predicted'])
selected_patient_preds = selected_preds.groupby(['patient_idx']).agg({
    'predicted': 'mean',
    'entropy': 'mean',
    'cross_entropy': 'mean',
    'y': 'mean',
    'correct': 'sum'
}).reset_index()
plt.scatter(selected_patient_preds['patient_idx'], selected_patient_preds['predicted'], s=selected_patient_preds['entropy']*100, c=selected_patient_preds['y'] + 0.5 * (selected_patient_preds['entropy'] > 0.3))
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.title('Predicted Probability with Uncertainty')
plt.xlabel('Patient Index')
plt.ylabel('Predicted Probability')
plt.legend()
plt.tight_layout()
plt.show()

plt.scatter(selected_patient_preds['patient_idx'], selected_patient_preds['predicted'], s=selected_patient_preds['entropy']*100, c=selected_patient_preds['y'] + 0.5 * (selected_patient_preds['cross_entropy'] > 0.8))
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.title('Predicted Probability with Uncertainty')
plt.xlabel('Patient Index')
plt.ylabel('Predicted Probability')
plt.legend()
plt.tight_layout()
plt.show()


selected_patient_preds


for model_name in val_results.nlargest(5, 'weighted_score')['model_name']:
    selected_columns = ['group','model','learning_rate','accuracy', 'scaled_mcc', 'f1', 'f1_0', 'auc','weighted_score']
    row_values = val_results[val_results['model_name'] == model_name][selected_columns].values.astype(str).tolist()
    joined_values = ','.join([','.join(row) for row in row_values])
    print(joined_values)


for model_name in val_results.nlargest(5, 'weighted_score')['model_name']:
    selected_columns = ['group','model','learning_rate','accuracy', 'scaled_mcc', 'f1', 'f1_0', 'auc','weighted_score']
    row_values = test_results[test_results['model_name'] == model_name][selected_columns].values.astype(str).tolist()
    joined_values = ','.join([','.join(row) for row in row_values])
    print(joined_values)


for model_name in val_results.nlargest(5, 'weighted_score')['model_name']:
    selected_columns = ['group','model','learning_rate','accuracy', 'scaled_mcc', 'f1', 'f1_0', 'auc','weighted_score']
    row_values = ensemble_results[ensemble_results['model_name'] == model_name][selected_columns].values.astype(str).tolist()
    joined_values = ','.join([','.join(row) for row in row_values])
    print(joined_values)


for model_name in val_results.nlargest(5, 'weighted_score')['model_name']:
    selected_columns = ['group','model','learning_rate','accuracy', 'scaled_mcc', 'f1', 'f1_0', 'auc','weighted_score']
    row_values = val_patient_results[val_patient_results['model_name'] == model_name][selected_columns].values.astype(str).tolist()
    joined_values = ','.join([','.join(row) for row in row_values])
    print(joined_values)


for model_name in val_results.nlargest(5, 'weighted_score')['model_name']:
    selected_columns = ['group','model','learning_rate','accuracy', 'scaled_mcc', 'f1', 'f1_0', 'auc','weighted_score']
    row_values = test_patient_results[test_patient_results['model_name'] == model_name][selected_columns].values.astype(str).tolist()
    joined_values = ','.join([','.join(row) for row in row_values])
    print(joined_values)


for model_name in val_results.nlargest(5, 'weighted_score')['model_name']:
    selected_columns = ['group','model','learning_rate','accuracy', 'scaled_mcc', 'f1', 'f1_0', 'auc','weighted_score']
    row_values = ensemble_patient_results[ensemble_patient_results['model_name'] == model_name][selected_columns].values.astype(str).tolist()
    joined_values = ','.join([','.join(row) for row in row_values])
    print(joined_values)
