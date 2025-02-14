import h5py
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
import seaborn as sns

dataset_filename = 'P:/REALTEK-HeadNeck-Project/OUS Mice/datasets/mice.h5'

result_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/orion/'

preprocess_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/preprocess_data/'

sns.set_style('whitegrid')

model = 'b4_lr0005_concat'

def print_metrics(model):
    df = pd.read_csv(result_path + model + '/test/result.csv')
    print('ACC, AUC, F1, F1_0, MCC')
    print(f'{df.iloc[0,4]},{df.iloc[0,2]},{df.iloc[0,6]},{df.iloc[0,7]},{df.iloc[0,5]}')

print_metrics(model)

for i in range(5):
    print_metrics(f'MH_3c_b4_lr0005_test_fold{i}')

slice_results = []
for i in range(1, 6):
    for lr in [0.001, 0.0005, 0.0001]:
        model = f'b{i}_lr{str(lr)[2:]}_concat'
        df = pd.read_csv(result_path + model + '/test/result.csv')
        slice_results.append(
            {
                'model': f'b{i}',
                'lr': lr,
                'ACC': df.iloc[0,4],
                'AUC': df.iloc[0,2],
                'F1_class_1': df.iloc[0,6],
                'F1_class_0': df.iloc[0,7],
                'MCC': df.iloc[0,5],
                'MCC_scaled': df.iloc[0,5] / 2 + 0.5,
            }
        )

slice_df = pd.DataFrame(slice_results)
slice_df.mean().sort_values()
orders = ['F1_class_0', 'ACC', 'MCC_scaled', 'F1_class_1', 'AUC']

slice_data = slice_df.melt(['lr', 'model'], var_name='Metrics', value_name='Values')
sns.catplot(data=slice_data, x='Metrics', y='Values', hue='model', col='lr', kind='point', linestyles='--', order=orders)
plt.show()



slice_results_2 = []
for input_name in ['b4', 'AH_b4', 'unsharp_b4', 'all_b4', 'MH_b4', 'MH_AH_b4', 'MH_unsharp_b4', 'MH_3c_b4']:
        model = f'{input_name}_lr0005_concat'
        df = pd.read_csv(result_path + model + '/test/result.csv')
        slice_results_2.append(
            {
                'input': {'b4': 'A', 'AH_b4':'B', 'unsharp_b4':'C', 'all_b4':'D', 'MH_b4':'E', 'MH_AH_b4':'F', 'MH_unsharp_b4':'G', 'MH_3c_b4':'H'}[input_name],
                'ACC': df.iloc[0,4],
                'AUC': df.iloc[0,2],
                'F1_class_1': df.iloc[0,6],
                'F1_class_0': df.iloc[0,7],
                'MCC': df.iloc[0,5],
                'MCC_scaled': df.iloc[0,5] / 2 + 0.5,
            }
        )



slice_df_2 = pd.DataFrame(slice_results_2)
slice_data_2 = slice_df_2.melt(['input'], var_name='Metrics', value_name='Values')
sns.pointplot(data=slice_data_2, x='Metrics', y='Values', hue='input', linestyles='--', order=orders)
plt.show()



def check_per_mouse(model):
    with h5py.File(result_path + model + '/test/prediction_test.h5', 'r') as f:
        data = {
            'pid': f['patient_idx'][:],
            'y': f['y'][:, 0],
            'predicted': f['predicted'][:, 0],
        }

    result_df = pd.DataFrame(data)

    with h5py.File(dataset_filename, 'r') as f:
        pids = []
        sids = []
        for i in range(5):
            pids.append(f[f'fold_{i}']['patient_idx'][:])
            sids.append(f[f'fold_{i}']['slice_idx'][:])
        pids = np.concatenate(pids)
        sids = np.concatenate(sids)

    assert np.all(result_df.pid == pids)

    result_df['slice_idx'] = sids
    result_df['predicted_class'] = (result_df['predicted'] > 0.5).astype(float)
    result_df['correct'] = (result_df['y'] == result_df['predicted_class'])
    result_df[result_df.y == 1].groupby('pid').agg({'correct': 'sum', 'slice_idx': 'count'})
    result_df[result_df.y == 0].groupby('pid').agg({'correct': 'sum', 'slice_idx': 'count'})

    summary_df = result_df.groupby('pid').agg(y=('y', 'mean'),
                                            avg_pred = ('predicted', 'mean'),
                                            correct_slices=('correct', 'sum'),
                                            all_slices=('slice_idx', 'count')).reset_index()
    summary_df['correct_ratio'] = summary_df['correct_slices'] / summary_df['all_slices']
    summary_df['major_vote_correct'] = (summary_df['correct_ratio'] > 0.5)
    summary_df['avg_correct'] = summary_df['y'] == (summary_df['avg_pred'] > 0.5).astype(float)
    return summary_df

check_per_mouse('b1_lr0005_concat')



sample_results = []
for i in range(1, 6):
    for lr in [0.001, 0.0005, 0.0001]:
        model = f'b{i}_lr{str(lr)[2:]}_concat'
        df = check_per_mouse(model)
        total_samples = df.shape[0]
        avg_predicted_class = (df['avg_pred'] >= 0.5).astype(float)
        major_vote_pred = (1-df['y']) + df['correct_ratio'] * ((-1) ** (1 + df['y']))
        major_vote_class = (major_vote_pred >= 0.5).astype(float)
        # avg res
        sample_results.append(
            {
                'model': f'b{i}',
                'lr': lr,
                'criteria': 'average',
                'ACC': accuracy_score(df['y'], avg_predicted_class),
                'AUC': roc_auc_score(df['y'], df['avg_pred']),
                'F1_class_1': f1_score(df['y'], avg_predicted_class),
                'F1_class_0': f1_score(1 - df['y'], 1 - avg_predicted_class),
                'MCC': matthews_corrcoef(df['y'], avg_predicted_class),
                'MCC_scaled': matthews_corrcoef(df['y'], avg_predicted_class) / 2 + 0.5,
            }
        )
        # majority res
        sample_results.append(
            {
                'model': f'b{i}',
                'lr': lr,
                'criteria': 'majority',
                'ACC': accuracy_score(df['y'], major_vote_class),
                'AUC': roc_auc_score(df['y'], major_vote_pred),
                'F1_class_1': f1_score(df['y'], major_vote_class),
                'F1_class_0': f1_score(1 - df['y'], 1 - major_vote_class),
                'MCC': matthews_corrcoef(df['y'], major_vote_class),
                'MCC_scaled': matthews_corrcoef(df['y'], major_vote_class) / 2 + 0.5,
            }
        )

sample_df = pd.DataFrame(sample_results)
sample_df.mean().sort_values()
# orders = ['F1_class_0', 'ACC', 'MCC_scaled', 'F1_class_1', 'AUC']

sample_data = sample_df.melt(['lr', 'model', 'criteria'], var_name='Metrics', value_name='Values')
sns.catplot(data=sample_data, x='Metrics', y='Values', hue='model', col='lr', row='criteria', kind='point', linestyles='--', order=orders)
plt.show()



sample_results_2 = []
for input_name in ['b4', 'AH_b4', 'unsharp_b4', 'all_b4', 'MH_b4', 'MH_AH_b4', 'MH_unsharp_b4','MH_3c_b4']:
        model = f'{input_name}_lr0005_concat'
        df = check_per_mouse(model)
        total_samples = df.shape[0]
        avg_predicted_class = (df['avg_pred'] >= 0.5).astype(float)
        major_vote_pred = (1-df['y']) + df['correct_ratio'] * ((-1) ** (1 + df['y']))
        major_vote_class = (major_vote_pred >= 0.5).astype(float)
        # avg res
        sample_results_2.append(
            {
                'input': {'b4': 'A', 'b1': 'A1', 'AH_b4':'B', 'unsharp_b4':'C', 'all_b4':'D', 'all_b1':'D1', 'MH_b4':'E', 'MH_AH_b4':'F', 'MH_unsharp_b4':'G', 'MH_3c_b4':'H'}[input_name],
                'criteria': 'average',
                'ACC': accuracy_score(df['y'], avg_predicted_class),
                'AUC': roc_auc_score(df['y'], df['avg_pred']),
                'F1_class_1': f1_score(df['y'], avg_predicted_class),
                'F1_class_0': f1_score(1 - df['y'], 1 - avg_predicted_class),
                'MCC': matthews_corrcoef(df['y'], avg_predicted_class),
                'MCC_scaled': matthews_corrcoef(df['y'], avg_predicted_class) / 2 + 0.5,
            }
        )
        # majority res
        sample_results_2.append(
            {
                'input': {'b4': 'A', 'b1': 'A1', 'AH_b4':'B', 'unsharp_b4':'C', 'all_b4':'D', 'all_b1':'D1', 'MH_b4':'E', 'MH_AH_b4':'F', 'MH_unsharp_b4':'G', 'MH_3c_b4':'H'}[input_name],
                # 'lr': lr,
                'criteria': 'majority',
                'ACC': accuracy_score(df['y'], major_vote_class),
                'AUC': roc_auc_score(df['y'], major_vote_pred),
                'F1_class_1': f1_score(df['y'], major_vote_class),
                'F1_class_0': f1_score(1 - df['y'], 1 - major_vote_class),
                'MCC': matthews_corrcoef(df['y'], major_vote_class),
                'MCC_scaled': matthews_corrcoef(df['y'], major_vote_class) / 2 + 0.5,
            }
        )

sample_df_2 = pd.DataFrame(sample_results_2)
sample_df_2.mean().sort_values()
# orders = ['F1_class_0', 'ACC', 'MCC_scaled', 'F1_class_1', 'AUC']

sample_data_2 = sample_df_2.melt(['input', 'criteria'], var_name='Metrics', value_name='Values')
sns.catplot(data=sample_data_2, x='Metrics', y='Values', hue='input', col='criteria', kind='point', linestyles='--', order=orders)
plt.show()
