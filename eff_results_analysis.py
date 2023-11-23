import h5py
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

dataset_filename = 'P:/REALTEK-HeadNeck-Project/OUS Mice/datasets/mice.h5'

result_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/orion/'

preprocess_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/preprocess_data/'



model = 'b4_lr0005_concat'

def print_metrics(model):
    df = pd.read_csv(result_path + model + '/test/result.csv')
    print('ACC, AUC, F1, F1_0, MCC')
    print(f'{df.iloc[0,4]},{df.iloc[0,2]},{df.iloc[0,6]},{df.iloc[0,7]},{df.iloc[0,5]}')

print_metrics(model)

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

pid = 0
i = 0
for _, item in result_df[result_df.y == 1].iterrows():
    if pid != item['pid']:
        pid = item['pid']
        plt.show()
        with open(preprocess_path + f'P{pid:02d}.npy', 'rb') as f:
            img = np.load(f)
        i = 0
    i += 1
    sid = item['slice_idx']
    is_correct = 'Correct' if item['correct'] else 'Incorrect'
    pred = item['predicted']
    plt.subplot(4, 6, i)
    plt.axis('off')
    plt.imshow(img[sid - 1])
    plt.title(f'Predicted: {pred: .3f} - {is_correct}')
plt.show()

item[1]['slice_idx']


summary_df = result_df.groupby('pid').agg(y=('y', 'mean'),
                                          avg_pred = ('predicted', 'mean'),
                                          correct_slices=('correct', 'sum'),
                                          all_slices=('slice_idx', 'count')).reset_index()
summary_df['correct_ratio'] = summary_df['correct_slices'] / summary_df['all_slices']
summary_df['major_vote_correct'] = (summary_df['correct_ratio'] > 0.5)
summary_df['avg_correct'] = summary_df['y'] == (summary_df['avg_pred'] > 0.5).astype(float)
summary_df

roc_auc_score(summary_df['y'], summary_df['avg_pred'])



def check_model(model):
    with h5py.File(path + model + '/test/prediction_test.h5', 'r') as f:
        data = {
            'pid': f['patient_idx'][:],
            'y': f['y'][:, 0],
            'predicted': f['predicted'][:, 0],
        }

    result_df = pd.DataFrame(data)

    result_df['predicted_class'] = (result_df['predicted'] > 0.5).astype(float)
    result_df['correct'] = (result_df['y'] == result_df['predicted_class'])

    summary_df = result_df.groupby('pid').agg(y=('y', 'mean'),
                                            avg_pred = ('predicted', 'mean'),
                                            correct_slices=('correct', 'sum'),
                                            all_slices=('predicted', 'count')).reset_index()
    summary_df['correct_ratio'] = summary_df['correct_slices'] / summary_df['all_slices']
    summary_df['major_vote_correct'] = (summary_df['correct_ratio'] > 0.5)
    summary_df['avg_correct'] = summary_df['y'] == (summary_df['avg_pred'] > 0.5).astype(float)

    print(roc_auc_score(summary_df['y'], summary_df['avg_pred']))
    print(summary_df.head())
