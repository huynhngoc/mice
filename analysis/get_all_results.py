import os
import h5py
import pandas as pd

# perf_path = '/mnt/ScratchProjects/KBT/mice/perf/'
perf_path = '../perf/efficientnet/'
for name in os.listdir(perf_path):
    if not name.endswith('_concat') and not 'test_fold' in name:
        print(name)
        log_folder = perf_path + name
        with open(log_folder + '/info.txt', 'r') as f:
            info = f.read()
        print('best_model', info[-25:-22])
        for val_pred in os.listdir(log_folder + '/prediction'):
            if info[-25:-22] in val_pred:
                print(val_pred)
                with h5py.File(log_folder + '/prediction/' + val_pred, 'r') as f:
                    print(f['predicted'].shape)
                    print(f['y'].shape)
                    print(f['patient_idx'].shape)
                    print(f['slice_idx'].shape)
        break


def print_metrics(path, group, model):
    df = pd.read_csv(path + group + '/' + model + '/test/result.csv')
    # print('model,ACC, AUC, F1, F1_0, MCC')
    model, lr, _ = model.split('_')
    print(f'{group},{model},{lr},{df.iloc[0,4]},{df.iloc[0,2]},{df.iloc[0,6]},{df.iloc[0,7]},{df.iloc[0,5]}')


perf_path = '../perf/'
for group in os.listdir(perf_path):
    for name in os.listdir(perf_path + group):
        if name.endswith('_concat'):
            # log_folder = perf_path + name
            print_metrics(perf_path, group, name)

perf_path = '/mnt/ScratchProjects/KBT/mice/perf/'
for group in os.listdir(perf_path):
    for name in os.listdir(perf_path + group):
        if name.endswith('_concat'):
            # log_folder = perf_path + name
            print_metrics(perf_path, group, name)


def print_val_metrics(path, group, model, best_model):
    df = pd.read_csv(path + group + '/' + model + '/log_new.csv')
    # print('model,ACC, AUC, F1, F1_0, MCC')
    best_index = df[df.epochs == best_model].index[0]
    model, lr, foldgroup = model.split('_')
    print(f'{group},{model},{lr},{foldgroup[-2]},{foldgroup[-1]},{df.iloc[best_index,4]},{df.iloc[best_index,2]},{df.iloc[best_index,6]},{df.iloc[best_index,7]},{df.iloc[best_index,5]}')

perf_path = '../perf/'
for group in os.listdir(perf_path):
    for name in os.listdir(perf_path + group):
        if not name.endswith('_concat') and not 'test_fold' in name:
            with open(perf_path + group + '/' + name + '/info.txt', 'r') as f:
                info = f.read()
            # print('best_model', info[-25:-22])
            # log_folder = perf_path + name
            print_val_metrics(perf_path, group, name, int(info[-25:-22]))

perf_path = '/mnt/ScratchProjects/KBT/mice/perf/'
for group in os.listdir(perf_path):
    for name in os.listdir(perf_path + group):
        if not name.endswith('_concat') and not 'test_fold' in name:
            with open(perf_path + group + '/' + name + '/info.txt', 'r') as f:
                info = f.read()
            # print('best_model', info[-25:-22])
            # log_folder = perf_path + name
            print_val_metrics(perf_path, group, name, int(info[-25:-22]))



def print_test_metrics(path, group, model):
    df = pd.read_csv(path + group + '/' + model + '/test/result.csv')
    # print('model,ACC, AUC, F1, F1_0, MCC')
    model, lr, foldgroup = model.split('_')
    print(f'{group},{model},{lr},{foldgroup[-2]},{foldgroup[-1]},{df.iloc[0,4]},{df.iloc[0,2]},{df.iloc[0,6]},{df.iloc[0,7]},{df.iloc[0,5]}')

perf_path = '../perf/'
for group in os.listdir(perf_path):
    for name in os.listdir(perf_path + group):
        if not name.endswith('_concat') and not 'test_fold' in name:
            # print('best_model', info[-25:-22])
            # log_folder = perf_path + name
            print_test_metrics(perf_path, group, name)

perf_path = '/mnt/ScratchProjects/KBT/mice/perf/'
for group in os.listdir(perf_path):
    for name in os.listdir(perf_path + group):
        if not name.endswith('_concat') and not 'test_fold' in name:
            # print('best_model', info[-25:-22])
            # log_folder = perf_path + name
            print_test_metrics(perf_path, group, name)

val_results = []
test_results = []
perf_path = '../perf/'
for group in os.listdir(perf_path):
    for name in os.listdir(perf_path + group):
        if not name.endswith('_concat') and not 'test_fold' in name:
            print(name)
            log_folder = perf_path + group + '/' + name
            with open(log_folder + '/info.txt', 'r') as f:
                info = f.read()
            print('best_model', info[-25:-22])
            for val_pred in os.listdir(log_folder + '/prediction'):
                if info[-25:-22] in val_pred:
                    print(val_pred)
                    with h5py.File(log_folder + '/prediction/' + val_pred, 'r') as f:
                        predicted = f['predicted'][:, 0]
                        y = f['y'][:, 0]
                        patient_idx = f['patient_idx'][:]
                        slice_idx = f['slice_idx'][:]
                    val_results.extend([{
                        'group': group,
                        'model_name': name,
                        'model': name.split('_')[0],
                        'learninge_rate': float('0.' + name.split('_')[1][2:]),
                        'val': name.split('_')[2][-2],
                        'test': name.split('_')[2][-1],
                        'patient_idx': patient_idx,
                        'slice_idx': slice_idx,
                        'y': y,
                        'predicted': predicted
                    } for predicted, y, patient_idx, slice_idx in zip(predicted, y, patient_idx, slice_idx)])
            with h5py.File(log_folder + '/test/prediction_test.h5', 'r') as f:
                predicted_test = f['predicted'][:, 0]
                y_test = f['y'][:, 0]
                patient_idx_test = f['patient_idx'][:]
                slice_idx_test = f['slice_idx'][:]
            test_results.extend([{
                'group': group,
                'model_name': name,
                'model': name.split('_')[0],
                'learninge_rate': float('0.' + name.split('_')[1][2:]),
                'val': name.split('_')[2][-2],
                'test': name.split('_')[2][-1],
                'patient_idx': patient_idx,
                'slice_idx': slice_idx,
                'y': y,
                'predicted': predicted
            } for predicted, y, patient_idx, slice_idx in zip(predicted_test, y_test, patient_idx_test, slice_idx_test)])

perf_path = '/mnt/ScratchProjects/KBT/mice/perf/'
for group in os.listdir(perf_path):
    for name in os.listdir(perf_path + group):
        if not name.endswith('_concat') and not 'test_fold' in name:
            print(name)
            log_folder = perf_path + group + '/' + name
            with open(log_folder + '/info.txt', 'r') as f:
                info = f.read()
            print('best_model', info[-25:-22])
            for val_pred in os.listdir(log_folder + '/prediction'):
                if info[-25:-22] in val_pred:
                    print(val_pred)
                    with h5py.File(log_folder + '/prediction/' + val_pred, 'r') as f:
                        predicted = f['predicted'][:, 0]
                        y = f['y'][:, 0]
                        patient_idx = f['patient_idx'][:]
                        slice_idx = f['slice_idx'][:]
                    val_results.extend([{
                        'group': group,
                        'model_name': name,
                        'model': name.split('_')[0],
                        'learninge_rate': float('0.' + name.split('_')[1][2:]),
                        'val': name.split('_')[2][-2],
                        'test': name.split('_')[2][-1],
                        'patient_idx': patient_idx,
                        'slice_idx': slice_idx,
                        'y': y,
                        'predicted': predicted
                    } for predicted, y, patient_idx, slice_idx in zip(predicted, y, patient_idx, slice_idx)])
            with h5py.File(log_folder + '/test/prediction_test.h5', 'r') as f:
                predicted_test = f['predicted'][:, 0]
                y_test = f['y'][:, 0]
                patient_idx_test = f['patient_idx'][:]
                slice_idx_test = f['slice_idx'][:]
            test_results.extend([{
                'group': group,
                'model_name': name,
                'model': name.split('_')[0],
                'learninge_rate': float('0.' + name.split('_')[1][2:]),
                'val': name.split('_')[2][-2],
                'test': name.split('_')[2][-1],
                'patient_idx': patient_idx,
                'slice_idx': slice_idx,
                'y': y,
                'predicted': predicted
            } for predicted, y, patient_idx, slice_idx in zip(predicted_test, y_test, patient_idx_test, slice_idx_test)])

val_results_df = pd.DataFrame(val_results)
test_results_df = pd.DataFrame(test_results)

val_results_df.to_csv('../analysis/val_preds.csv', index=False)
test_results_df.to_csv('../analysis/test_preds.csv', index=False)
