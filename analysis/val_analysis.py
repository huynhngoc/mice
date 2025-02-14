import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score

# Load data
val_preds = pd.read_csv('analysis_results/val_preds.csv')


def calculate_metrics(group):
    y_true = group['y']
    y_pred = group['predicted']
    y_pred_binary = (y_pred > 0.5).astype(float)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'auc': roc_auc_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred_binary),
        'f1_0': f1_score(y_true, y_pred_binary),
        'scaled_mcc': matthews_corrcoef(y_true, y_pred_binary)/2 + 0.5,
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary),
        'specificity': recall_score(y_true, y_pred_binary, pos_label=0),
    }
    return pd.Series(metrics)

###########################
# Per slice analysis
###########################
# Group by the specified columns and apply the function
grouped_metrics = val_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test']
    ).apply(calculate_metrics).reset_index()

# calculate the weighted score
grouped_metrics['weighted_score'] = (grouped_metrics['accuracy'] + grouped_metrics['auc'] + grouped_metrics['f1'] + grouped_metrics['f1_0'] + grouped_metrics['scaled_mcc']) / 5


grouped_metrics.groupby(['group', 'model', 'learning_rate']).mean().reset_index().sort_values('weighted_score', ascending=False).head(20)[['model', 'learning_rate', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']]
#         model  learning_rate  accuracy       auc        f1      f1_0  scaled_mcc  weighted_score
# 3          B1         0.0005  0.823149  0.891901  0.807700  0.834097    0.826524        0.836674
# 13         B6         0.0005  0.817763  0.897700  0.802452  0.826953    0.818709        0.832716
# 1          B0         0.0005  0.816370  0.893751  0.804396  0.820142    0.818008        0.830533
# 18          S         0.0001  0.816120  0.890432  0.798986  0.827896    0.818087        0.830304
# 6          B3         0.0001  0.816744  0.883365  0.806800  0.820306    0.819124        0.829268
# 15         B7         0.0005  0.812994  0.888274  0.811961  0.807420    0.818735        0.827877
# 19          S         0.0005  0.814465  0.885824  0.803167  0.820470    0.815432        0.827871
# 16          M         0.0001  0.815129  0.875386  0.801968  0.823591    0.817193        0.826654
# 9          B4         0.0005  0.809292  0.882958  0.798191  0.815219    0.816143        0.824361
# 7          B3         0.0005  0.809137  0.883399  0.791518  0.821189    0.812377        0.823524
# 17          M         0.0005  0.810137  0.880773  0.791106  0.817859    0.816529        0.823281
# 5          B2         0.0005  0.806948  0.882397  0.799845  0.810192    0.811705        0.822218
# 4          B2         0.0001  0.809593  0.876040  0.797250  0.816119    0.810451        0.821890
# 24  ResNet101         0.0001  0.807833  0.878877  0.790712  0.818693    0.811278        0.821478
# 14         B7         0.0001  0.805910  0.883110  0.803193  0.804856    0.809289        0.821272
# 11         B5         0.0005  0.806698  0.882011  0.798197  0.807144    0.811688        0.821148
# 2          B1         0.0001  0.806056  0.879301  0.780929  0.822171    0.808357        0.819363
# 12         B6         0.0001  0.798445  0.866652  0.780306  0.808294    0.800148        0.810769
# 21  Inception         0.0005  0.795687  0.872275  0.780149  0.803341    0.798481        0.809987
# 27  ResNet152         0.0001  0.795245  0.870823  0.777672  0.804122    0.797946        0.809161

###########################
# Per patient analysis
###########################
def count_correct_prediction(group):
    y_true = group['y']
    y_pred = group['predicted']
    y_pred_binary = (y_pred > 0.5).astype(float)
    return pd.Series({
        'correct': (y_true == y_pred_binary).sum(),
        'total': len(y_true),
        'correct_rate': (y_true == y_pred_binary).sum() / len(y_true)
    })

# Group by the specified columns and apply the function
patient_mean_results = val_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test', 'patient_idx']
    ).agg({
        'slice_idx': 'count',
        'y': 'mean',
        'predicted': 'mean'
    }).reset_index()
grouped_patients_metrics = patient_mean_results.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test']
    ).apply(calculate_metrics).reset_index()

grouped_patients_metrics[['model', 'learning_rate', 'val','test', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']].head(20)

grouped_patients_metrics['weighted_score'] = (grouped_patients_metrics['accuracy'] + grouped_patients_metrics['auc'] + grouped_patients_metrics['f1'] + grouped_patients_metrics['f1_0'] + grouped_patients_metrics['scaled_mcc']) / 5
grouped_patients_metrics.groupby(['group', 'model', 'learning_rate']).mean().reset_index().sort_values('weighted_score', ascending=False).head(20)[['model', 'learning_rate', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']]
#         model  learning_rate  accuracy       auc        f1      f1_0  scaled_mcc  weighted_score
# 13         B6         0.0005  0.956667  0.994444  0.946190  0.961429    0.961020        0.963950
# 1          B0         0.0005  0.948333  0.983333  0.944762  0.945714    0.953698        0.955168
# 18          S         0.0001  0.941667  0.983333  0.936190  0.944762    0.946722        0.950535
# 14         B7         0.0001  0.940000  0.983333  0.944286  0.932857    0.947733        0.949642
# 3          B1         0.0005  0.940000  0.972222  0.939048  0.939048    0.945711        0.947206
# 6          B3         0.0001  0.933333  0.988889  0.929762  0.930833    0.940224        0.944608
# 16          M         0.0001  0.931667  0.988889  0.928571  0.931429    0.940410        0.944193
# 9          B4         0.0005  0.931667  0.988889  0.930357  0.926429    0.941235        0.943715
# 4          B2         0.0001  0.931667  0.969444  0.924762  0.931429    0.939053        0.939271
# 19          S         0.0005  0.930000  0.972222  0.924762  0.928571    0.938042        0.938720
# 7          B3         0.0005  0.923333  0.972222  0.919048  0.924762    0.931066        0.934086
# 12         B6         0.0001  0.923333  0.972222  0.906905  0.928929    0.932556        0.932789
# 21  Inception         0.0005  0.915000  0.994444  0.907619  0.914286    0.924408        0.931152
# 15         B7         0.0005  0.915000  0.986111  0.923571  0.895714    0.927416        0.929563
# 0          B0         0.0001  0.915000  0.977778  0.921786  0.900714    0.926591        0.928374
# 27  ResNet152         0.0001  0.913333  0.988889  0.898095  0.917143    0.922040        0.927900
# 5          B2         0.0005  0.911667  0.986111  0.909286  0.909048    0.919577        0.927138
# 24  ResNet101         0.0001  0.913333  0.977778  0.899048  0.920000    0.923397        0.926711
# 2          B1         0.0001  0.905000  0.975000  0.881190  0.917500    0.916900        0.919118
# 11         B5         0.0005  0.895000  1.000000  0.878810  0.896071    0.907209        0.915418

patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(count_correct_prediction).reset_index()
patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(count_correct_prediction).reset_index().groupby(['group', 'model', 'learning_rate']).min().reset_index().sort_values('correct_rate', ascending=False)[['model', 'learning_rate', 'correct_rate']]

patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(
    count_correct_prediction).reset_index().groupby(
        ['group', 'model', 'learning_rate', 'correct_rate']
    ).count().reset_index()[['group', 'model', 'learning_rate', 'correct_rate', 'patient_idx']].pivot(
        index=['group', 'model', 'learning_rate'], columns='correct_rate', values='patient_idx').reset_index().fillna(0)
# correct_rate         group      model  learning_rate  0.0  0.25  0.5  0.75   1.0
# 0             efficientnet         B0         0.0001  0.0   1.0  1.0   5.0  22.0
# 1             efficientnet         B0         0.0005  0.0   0.0  1.0   4.0  24.0
# 2             efficientnet         B1         0.0001  0.0   1.0  1.0   6.0  21.0
# 3             efficientnet         B1         0.0005  0.0   0.0  1.0   5.0  23.0
# 4             efficientnet         B2         0.0001  0.0   0.0  0.0   8.0  21.0
# 5             efficientnet         B2         0.0005  0.0   0.0  1.0   8.0  20.0
# 6             efficientnet         B3         0.0001  0.0   0.0  3.0   2.0  24.0
# 7             efficientnet         B3         0.0005  0.0   0.0  0.0   9.0  20.0
# 8             efficientnet         B4         0.0001  0.0   1.0  2.0   9.0  17.0
# 9             efficientnet         B4         0.0005  0.0   0.0  1.0   6.0  22.0
# 10            efficientnet         B5         0.0001  0.0   2.0  1.0   8.0  18.0
# 11            efficientnet         B5         0.0005  0.0   0.0  1.0  10.0  18.0
# 12            efficientnet         B6         0.0001  0.0   1.0  1.0   4.0  23.0
# 13            efficientnet         B6         0.0005  0.0   0.0  0.0   5.0  24.0
# 14            efficientnet         B7         0.0001  0.0   0.0  1.0   5.0  23.0
# 15            efficientnet         B7         0.0005  0.0   0.0  2.0   6.0  21.0
# 16            efficientnet          M         0.0001  0.0   0.0  0.0   8.0  21.0
# 17            efficientnet          M         0.0005  0.0   0.0  3.0   8.0  18.0
# 18            efficientnet          S         0.0001  0.0   0.0  0.0   7.0  22.0
# 19            efficientnet          S         0.0005  0.0   0.0  1.0   6.0  22.0
# 20               inception  Inception         0.0001  0.0   0.0  1.0  11.0  17.0
# 21               inception  Inception         0.0005  0.0   1.0  0.0   7.0  21.0
# 22               mobilenet  MobileNet         0.0001  1.0   2.0  6.0   8.0  12.0
# 23               mobilenet  MobileNet         0.0005  0.0   1.0  2.0  13.0  13.0
# 24                  resnet  ResNet101         0.0001  0.0   1.0  2.0   3.0  23.0
# 25                  resnet  ResNet101         0.0005  0.0   2.0  4.0   6.0  17.0
# 26                  resnet  ResNet101         0.0010  2.0   2.0  5.0   6.0  14.0
# 27                  resnet  ResNet152         0.0001  0.0   0.0  2.0   6.0  21.0
# 28                  resnet  ResNet152         0.0005  0.0   1.0  4.0   6.0  18.0
# 29                  resnet  ResNet152         0.0010  2.0   2.0  9.0  10.0   6.0
# 30                  resnet   ResNet50         0.0001  0.0   1.0  2.0   6.0  20.0
# 31                  resnet   ResNet50         0.0005  0.0   2.0  2.0   3.0  22.0
# 32                  resnet   ResNet50         0.0010  1.0   0.0  4.0  18.0   6.0
# 33                     vgg      VGG16         0.0001  2.0   8.0  5.0  12.0   2.0
# 34                     vgg      VGG19         0.0001  5.0   5.0  6.0   7.0   6.0


###########################
# Per patient by median analysis
###########################

patient_median_results = val_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test', 'patient_idx']
    ).agg({
        'slice_idx': 'count',
        'y': 'median',
        'predicted': 'median'
    }).reset_index()

grouped_patients_metrics_median = patient_median_results.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test']
    ).apply(calculate_metrics).reset_index()
grouped_patients_metrics_median['weighted_score'] = (grouped_patients_metrics_median['accuracy'] + grouped_patients_metrics_median['auc'] + grouped_patients_metrics_median['f1'] + grouped_patients_metrics_median['f1_0'] + grouped_patients_metrics_median['scaled_mcc']) / 5
grouped_patients_metrics_median.groupby(['group', 'model', 'learning_rate']).mean().reset_index().sort_values('weighted_score', ascending=False).head(20)[['model', 'learning_rate', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']]
#         model  learning_rate  accuracy       auc        f1      f1_0  scaled_mcc  weighted_score
# 1          B0         0.0005  0.958333  0.977778  0.961429  0.952857    0.963388        0.962757
# 14         B7         0.0001  0.948333  0.988889  0.954286  0.940000    0.955055        0.957313
# 13         B6         0.0005  0.946667  0.994444  0.936190  0.951429    0.952687        0.956283
# 19          S         0.0005  0.948333  0.983333  0.941905  0.948571    0.953698        0.955168
# 6          B3         0.0001  0.941667  0.988889  0.939286  0.937500    0.949569        0.951382
# 18          S         0.0001  0.941667  0.983333  0.936190  0.944762    0.946722        0.950535
# 4          B2         0.0001  0.940000  0.969444  0.931905  0.941429    0.946375        0.945831
# 21  Inception         0.0005  0.933333  0.988889  0.934286  0.928571    0.941421        0.945300
# 3          B1         0.0005  0.940000  0.963889  0.935714  0.941429    0.944700        0.945146
# 7          B3         0.0005  0.931667  0.972222  0.921190  0.936548    0.939213        0.940168
# 5          B2         0.0005  0.921667  0.991667  0.924286  0.915714    0.932077        0.937082
# 9          B4         0.0005  0.923333  0.977778  0.920357  0.919286    0.933913        0.934933
# 0          B0         0.0001  0.923333  0.972222  0.933571  0.902857    0.934738        0.933344
# 16          M         0.0001  0.921667  0.969444  0.911905  0.924286    0.930720        0.931604
# 15         B7         0.0005  0.915000  0.980556  0.923571  0.895714    0.927416        0.928451
# 24  ResNet101         0.0001  0.913333  0.975000  0.899048  0.920000    0.923397        0.926156
# 27  ResNet152         0.0001  0.913333  0.975000  0.898095  0.917143    0.922040        0.925122
# 2          B1         0.0001  0.905000  0.983333  0.874524  0.920357    0.915543        0.919751
# 30   ResNet50         0.0001  0.903333  0.977778  0.892738  0.899286    0.914532        0.917533
# 12         B6         0.0001  0.905000  0.952778  0.889762  0.908929    0.916900        0.914674

patient_median_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(
    count_correct_prediction).reset_index().groupby(
        ['group', 'model', 'learning_rate']).min().reset_index().sort_values(
            'correct_rate', ascending=False)[['model', 'learning_rate', 'correct_rate']]

patient_median_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(
    count_correct_prediction).reset_index().groupby(
        ['group', 'model', 'learning_rate', 'correct_rate']
    ).count().reset_index()[['group', 'model', 'learning_rate', 'correct_rate', 'patient_idx']].pivot(
        index=['group', 'model', 'learning_rate'], columns='correct_rate', values='patient_idx').reset_index().fillna(0)
# correct_rate         group      model  learning_rate  0.0  0.25  0.5  0.75   1.0
# 0             efficientnet         B0         0.0001  0.0   0.0  3.0   3.0  23.0
# 1             efficientnet         B0         0.0005  0.0   0.0  1.0   3.0  25.0
# 2             efficientnet         B1         0.0001  0.0   1.0  2.0   4.0  22.0
# 3             efficientnet         B1         0.0005  0.0   0.0  1.0   5.0  23.0
# 4             efficientnet         B2         0.0001  0.0   0.0  0.0   7.0  22.0
# 5             efficientnet         B2         0.0005  0.0   0.0  0.0   9.0  20.0
# 6             efficientnet         B3         0.0001  0.0   0.0  2.0   3.0  24.0
# 7             efficientnet         B3         0.0005  0.0   0.0  0.0   8.0  21.0
# 8             efficientnet         B4         0.0001  0.0   1.0  2.0   8.0  18.0
# 9             efficientnet         B4         0.0005  0.0   0.0  1.0   7.0  21.0
# 10            efficientnet         B5         0.0001  0.0   2.0  1.0   7.0  19.0
# 11            efficientnet         B5         0.0005  0.0   0.0  1.0  10.0  18.0
# 12            efficientnet         B6         0.0001  0.0   1.0  1.0   6.0  21.0
# 13            efficientnet         B6         0.0005  0.0   0.0  0.0   6.0  23.0
# 14            efficientnet         B7         0.0001  0.0   0.0  1.0   4.0  24.0
# 15            efficientnet         B7         0.0005  0.0   0.0  2.0   6.0  21.0
# 16            efficientnet          M         0.0001  0.0   0.0  0.0   9.0  20.0
# 17            efficientnet          M         0.0005  0.0   0.0  3.0   8.0  18.0
# 18            efficientnet          S         0.0001  0.0   0.0  0.0   7.0  22.0
# 19            efficientnet          S         0.0005  0.0   0.0  1.0   4.0  24.0
# 20               inception  Inception         0.0001  0.0   0.0  2.0   9.0  18.0
# 21               inception  Inception         0.0005  0.0   0.0  1.0   6.0  22.0
# 22               mobilenet  MobileNet         0.0001  0.0   3.0  6.0   7.0  13.0
# 23               mobilenet  MobileNet         0.0005  0.0   1.0  2.0  13.0  13.0
# 24                  resnet  ResNet101         0.0001  0.0   1.0  2.0   3.0  23.0
# 25                  resnet  ResNet101         0.0005  0.0   0.0  6.0   4.0  19.0
# 26                  resnet  ResNet101         0.0010  1.0   2.0  5.0   9.0  12.0
# 27                  resnet  ResNet152         0.0001  0.0   0.0  2.0   6.0  21.0
# 28                  resnet  ResNet152         0.0005  0.0   1.0  3.0   5.0  20.0
# 29                  resnet  ResNet152         0.0010  1.0   3.0  9.0   7.0   9.0
# 30                  resnet   ResNet50         0.0001  0.0   0.0  2.0   7.0  20.0
# 31                  resnet   ResNet50         0.0005  0.0   2.0  2.0   3.0  22.0
# 32                  resnet   ResNet50         0.0010  1.0   1.0  4.0  17.0   6.0
# 33                     vgg      VGG16         0.0001  2.0   6.0  9.0   8.0   4.0
# 34                     vgg      VGG19         0.0001  5.0   6.0  4.0   8.0   6.0


###########################
# Per patient by majority voting analysis
###########################
patient_vote_results = val_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test', 'patient_idx']
    ).agg({
        'slice_idx': 'count',
        'y': 'mean',
        'predicted': lambda predicted: (predicted > 0.5).sum() / len(predicted)
    }).reset_index()

grouped_patients_metrics_vote = patient_vote_results.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test']
    ).apply(calculate_metrics).reset_index()

grouped_patients_metrics_vote['weighted_score'] = (grouped_patients_metrics_vote['accuracy'] + grouped_patients_metrics_vote['auc'] + grouped_patients_metrics_vote['f1'] + grouped_patients_metrics_vote['f1_0'] + grouped_patients_metrics_vote['scaled_mcc']) / 5
grouped_patients_metrics_vote.groupby(['group', 'model', 'learning_rate']).mean().reset_index().sort_values('weighted_score', ascending=False).head(20)[['model', 'learning_rate', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']]
#         model  learning_rate  accuracy       auc        f1      f1_0  scaled_mcc  weighted_score
# 13         B6         0.0005  0.956667  0.994444  0.946190  0.961429    0.961020        0.963950
# 6          B3         0.0001  0.950000  0.994444  0.946429  0.947500    0.956891        0.959053
# 14         B7         0.0001  0.948333  0.983333  0.954286  0.940000    0.955055        0.956201
# 1          B0         0.0005  0.948333  0.983333  0.944762  0.945714    0.953698        0.955168
# 18          S         0.0001  0.941667  0.983333  0.936190  0.944762    0.946722        0.950535
# 7          B3         0.0005  0.940000  0.977778  0.927857  0.946071    0.948558        0.948053
# 3          B1         0.0005  0.940000  0.977778  0.935714  0.941429    0.944700        0.947924
# 16          M         0.0001  0.931667  0.994444  0.921905  0.934286    0.939053        0.944271
# 4          B2         0.0001  0.931667  0.975000  0.921905  0.934286    0.939053        0.940382
# 21  Inception         0.0005  0.923333  0.994444  0.914762  0.924286    0.931731        0.937711
# 0          B0         0.0001  0.923333  0.983333  0.933571  0.902857    0.934738        0.935567
# 24  ResNet101         0.0001  0.921667  0.983333  0.906190  0.930000    0.930720        0.934382
# 15         B7         0.0005  0.915000  1.000000  0.923571  0.895714    0.927416        0.932340
# 9          B4         0.0005  0.915000  0.988889  0.905357  0.913929    0.927416        0.930118
# 5          B2         0.0005  0.913333  0.986111  0.914286  0.908571    0.924755        0.929411
# 19          S         0.0005  0.921667  0.962500  0.900238  0.928929    0.930187        0.928704
# 27  ResNet152         0.0001  0.913333  0.988889  0.898095  0.917143    0.922040        0.927900
# 30   ResNet50         0.0001  0.911667  0.988889  0.899881  0.909286    0.921854        0.926315
# 20  Inception         0.0001  0.905000  0.988889  0.900476  0.901429    0.916075        0.922374
# 11         B5         0.0005  0.905000  0.991667  0.888810  0.906071    0.915543        0.921418


patient_vote_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(count_correct_prediction).reset_index()
patient_vote_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(
    count_correct_prediction).reset_index().groupby(
        ['group', 'model', 'learning_rate']).min().reset_index().sort_values(
            'correct_rate', ascending=False)[['model', 'learning_rate', 'correct_rate']]

patient_vote_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(
    count_correct_prediction).reset_index().groupby(
        ['group', 'model', 'learning_rate', 'correct_rate']
    ).count().reset_index()[['group', 'model', 'learning_rate', 'correct_rate', 'patient_idx']].pivot(
        index=['group', 'model', 'learning_rate'], columns='correct_rate', values='patient_idx').reset_index().fillna(0)
# correct_rate         group      model  learning_rate  0.0  0.25  0.5  0.75   1.0
# 0             efficientnet         B0         0.0001  0.0   0.0  3.0   3.0  23.0
# 1             efficientnet         B0         0.0005  0.0   0.0  1.0   4.0  24.0
# 2             efficientnet         B1         0.0001  0.0   1.0  2.0   4.0  22.0
# 3             efficientnet         B1         0.0005  0.0   0.0  1.0   5.0  23.0
# 4             efficientnet         B2         0.0001  0.0   0.0  0.0   8.0  21.0
# 5             efficientnet         B2         0.0005  0.0   0.0  0.0  10.0  19.0
# 6             efficientnet         B3         0.0001  0.0   0.0  1.0   4.0  24.0
# 7             efficientnet         B3         0.0005  0.0   0.0  0.0   7.0  22.0
# 8             efficientnet         B4         0.0001  0.0   1.0  2.0   8.0  18.0
# 9             efficientnet         B4         0.0005  0.0   0.0  1.0   8.0  20.0
# 10            efficientnet         B5         0.0001  0.0   2.0  1.0   6.0  20.0
# 11            efficientnet         B5         0.0005  0.0   0.0  1.0   9.0  19.0
# 12            efficientnet         B6         0.0001  0.0   1.0  1.0   6.0  21.0
# 13            efficientnet         B6         0.0005  0.0   0.0  0.0   5.0  24.0
# 14            efficientnet         B7         0.0001  0.0   0.0  1.0   4.0  24.0
# 15            efficientnet         B7         0.0005  0.0   0.0  2.0   6.0  21.0
# 16            efficientnet          M         0.0001  0.0   0.0  0.0   8.0  21.0
# 17            efficientnet          M         0.0005  0.0   0.0  3.0   7.0  19.0
# 18            efficientnet          S         0.0001  0.0   0.0  0.0   7.0  22.0
# 19            efficientnet          S         0.0005  0.0   0.0  1.0   7.0  21.0
# 20               inception  Inception         0.0001  0.0   0.0  1.0   9.0  19.0
# 21               inception  Inception         0.0005  0.0   0.0  1.0   7.0  21.0
# 22               mobilenet  MobileNet         0.0001  0.0   3.0  6.0   7.0  13.0
# 23               mobilenet  MobileNet         0.0005  0.0   1.0  2.0  13.0  13.0
# 24                  resnet  ResNet101         0.0001  0.0   1.0  1.0   4.0  23.0
# 25                  resnet  ResNet101         0.0005  0.0   0.0  4.0   5.0  20.0
# 26                  resnet  ResNet101         0.0010  2.0   1.0  5.0   8.0  13.0
# 27                  resnet  ResNet152         0.0001  0.0   0.0  2.0   6.0  21.0
# 28                  resnet  ResNet152         0.0005  0.0   1.0  2.0   6.0  20.0
# 29                  resnet  ResNet152         0.0010  1.0   4.0  8.0   7.0   9.0
# 30                  resnet   ResNet50         0.0001  0.0   0.0  1.0   8.0  20.0
# 31                  resnet   ResNet50         0.0005  0.0   2.0  2.0   3.0  22.0
# 32                  resnet   ResNet50         0.0010  1.0   1.0  3.0  18.0   6.0
# 33                     vgg      VGG16         0.0001  2.0   6.0  9.0   8.0   4.0
# 34                     vgg      VGG19         0.0001  5.0   6.0  4.0   8.0   6.0
