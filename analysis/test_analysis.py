import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score

# Load data
test_preds = pd.read_csv('analysis_results/test_preds.csv')


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

###########################
# Per slice analysis
###########################
# Group by the specified columns and apply the function
grouped_metrics = test_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test']
    ).apply(calculate_metrics).reset_index()

# calculate the weighted score
grouped_metrics['weighted_score'] = (grouped_metrics['accuracy'] + grouped_metrics['auc'] + grouped_metrics['f1'] + grouped_metrics['f1_0'] + grouped_metrics['scaled_mcc']) / 5


grouped_metrics.groupby(['group', 'model', 'learning_rate']).mean().reset_index().sort_values('weighted_score', ascending=False).head(20)[['model', 'learning_rate', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']]
#         model  learning_rate  accuracy       auc        f1      f1_0  scaled_mcc  weighted_score
# 6          B3         0.0001  0.766200  0.864456  0.752423  0.770078    0.769445        0.784521
# 14         B7         0.0001  0.763520  0.851627  0.763703  0.758898    0.769452        0.781440
# 18          S         0.0001  0.756868  0.841746  0.743798  0.764911    0.761757        0.773816
# 11         B5         0.0005  0.753428  0.833094  0.743351  0.752400    0.758312        0.768117
# 7          B3         0.0005  0.751595  0.840882  0.729599  0.763723    0.753272        0.767814
# 15         B7         0.0005  0.746964  0.846448  0.735903  0.743772    0.752610        0.765140
# 5          B2         0.0005  0.745324  0.842122  0.736926  0.746877    0.752275        0.764705
# 1          B0         0.0005  0.743877  0.843973  0.738029  0.740294    0.753112        0.763857
# 19          S         0.0005  0.745555  0.834902  0.728821  0.754473    0.749500        0.762650
# 30   ResNet50         0.0001  0.747909  0.826854  0.731466  0.751275    0.749985        0.761498
# 24  ResNet101         0.0001  0.747788  0.822052  0.728696  0.757713    0.748133        0.760877
# 4          B2         0.0001  0.743395  0.828890  0.731031  0.749544    0.744982        0.759568
# 12         B6         0.0001  0.742640  0.830979  0.717563  0.757501    0.744951        0.758727
# 9          B4         0.0005  0.741095  0.824362  0.721324  0.750703    0.746137        0.756724
# 2          B1         0.0001  0.740725  0.823890  0.709769  0.760126    0.744530        0.755808
# 13         B6         0.0005  0.731109  0.830508  0.714904  0.737577    0.737837        0.750387
# 0          B0         0.0001  0.731557  0.820748  0.725753  0.729180    0.737093        0.748866
# 8          B4         0.0001  0.733986  0.812487  0.730372  0.728362    0.739035        0.748848
# 16          M         0.0001  0.731114  0.822388  0.717191  0.737639    0.735840        0.748834
# 3          B1         0.0005  0.731464  0.819469  0.709856  0.743959    0.735661        0.748082


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
patient_mean_results = test_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test', 'patient_idx']
    ).agg({
        'slice_idx': 'count',
        'y': 'mean',
        'predicted': 'mean'
    }).reset_index()
grouped_patients_metrics = patient_mean_results.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test']
    ).apply(calculate_metrics).reset_index()

# grouped_patients_metrics[['model', 'learning_rate', 'val','test', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']].head(20)

grouped_patients_metrics['weighted_score'] = (grouped_patients_metrics['accuracy'] + grouped_patients_metrics['auc'] + grouped_patients_metrics['f1'] + grouped_patients_metrics['f1_0'] + grouped_patients_metrics['scaled_mcc']) / 5
grouped_patients_metrics.groupby(['group', 'model', 'learning_rate']).mean().reset_index().sort_values('weighted_score', ascending=False).head(20)[['model', 'learning_rate', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']]
#         model  learning_rate  accuracy       auc        f1      f1_0  scaled_mcc  weighted_score
# 6          B3         0.0001  0.900000  0.972222  0.895952  0.897024    0.908913        0.914822
# 15         B7         0.0005  0.886667  0.966667  0.881667  0.878214    0.903117        0.903266
# 14         B7         0.0001  0.878333  0.969444  0.885595  0.861905    0.895130        0.898082
# 2          B1         0.0001  0.881667  0.955556  0.856667  0.895952    0.894082        0.896785
# 8          B4         0.0001  0.878333  0.958333  0.882619  0.858571    0.894747        0.894521
# 10         B5         0.0001  0.881667  0.933333  0.862262  0.877381    0.892246        0.889378
# 9          B4         0.0005  0.875000  0.944444  0.864881  0.873452    0.887771        0.889110
# 4          B2         0.0001  0.865000  0.938889  0.854286  0.866667    0.873397        0.879648
# 19          S         0.0005  0.863333  0.941667  0.851548  0.864524    0.872424        0.878699
# 12         B6         0.0001  0.856667  0.961111  0.842619  0.860833    0.870279        0.878302
# 5          B2         0.0005  0.855000  0.966667  0.858929  0.840952    0.869119        0.878133
# 18          S         0.0001  0.860000  0.941667  0.860476  0.850238    0.872237        0.876924
# 24  ResNet101         0.0001  0.861667  0.908333  0.842619  0.865595    0.873876        0.870418
# 7          B3         0.0005  0.848333  0.911111  0.831548  0.849643    0.861414        0.860410
# 16          M         0.0001  0.838333  0.936111  0.822619  0.843690    0.854624        0.859076
# 13         B6         0.0005  0.836667  0.938889  0.827381  0.824762    0.860994        0.857738
# 1          B0         0.0005  0.828333  0.936111  0.827024  0.811190    0.852846        0.851101
# 0          B0         0.0001  0.830000  0.933333  0.834167  0.814048    0.839878        0.850285
# 30   ResNet50         0.0001  0.831667  0.944444  0.809286  0.821429    0.842725        0.849910
# 3          B1         0.0005  0.833333  0.916667  0.800952  0.845238    0.849776        0.849193


patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(count_correct_prediction).reset_index()
patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(count_correct_prediction).reset_index().groupby(['group', 'model', 'learning_rate']).min().reset_index().sort_values('correct_rate', ascending=False)[['model', 'learning_rate', 'correct_rate']]

patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(
    count_correct_prediction).reset_index().groupby(
        ['group', 'model', 'learning_rate', 'correct_rate']
    ).count().reset_index()[['group', 'model', 'learning_rate', 'correct_rate', 'patient_idx']].pivot(
        index=['group', 'model', 'learning_rate'], columns='correct_rate', values='patient_idx').reset_index().fillna(0)
# correct_rate         group      model  learning_rate  0.0  0.25   0.5  0.75   1.0
# 0             efficientnet         B0         0.0001  1.0   3.0   1.0   5.0  19.0
# 1             efficientnet         B0         0.0005  1.0   2.0   3.0   4.0  19.0
# 2             efficientnet         B1         0.0001  1.0   0.0   2.0   6.0  20.0
# 3             efficientnet         B1         0.0005  0.0   3.0   3.0   5.0  18.0
# 4             efficientnet         B2         0.0001  0.0   2.0   1.0   8.0  18.0
# 5             efficientnet         B2         0.0005  2.0   0.0   1.0   7.0  19.0
# 6             efficientnet         B3         0.0001  0.0   0.0   3.0   6.0  20.0
# 7             efficientnet         B3         0.0005  0.0   2.0   3.0   6.0  18.0
# 8             efficientnet         B4         0.0001  0.0   1.0   1.0   9.0  18.0
# 9             efficientnet         B4         0.0005  0.0   0.0   4.0   7.0  18.0
# 10            efficientnet         B5         0.0001  0.0   0.0   4.0   6.0  19.0
# 11            efficientnet         B5         0.0005  0.0   1.0   5.0   7.0  16.0
# 12            efficientnet         B6         0.0001  0.0   1.0   4.0   6.0  18.0
# 13            efficientnet         B6         0.0005  0.0   1.0   5.0   6.0  17.0
# 14            efficientnet         B7         0.0001  0.0   2.0   2.0   4.0  21.0
# 15            efficientnet         B7         0.0005  0.0   1.0   1.0   8.0  19.0
# 16            efficientnet          M         0.0001  0.0   1.0   4.0   8.0  16.0
# 17            efficientnet          M         0.0005  0.0   1.0   8.0   4.0  16.0
# 18            efficientnet          S         0.0001  0.0   1.0   2.0   9.0  17.0
# 19            efficientnet          S         0.0005  1.0   0.0   3.0   6.0  19.0
# 20               inception  Inception         0.0001  0.0   1.0   5.0   9.0  14.0
# 21               inception  Inception         0.0005  0.0   1.0   4.0   9.0  15.0
# 22               mobilenet  MobileNet         0.0001  2.0   4.0   3.0   9.0  11.0
# 23               mobilenet  MobileNet         0.0005  0.0   1.0   8.0  12.0   8.0
# 24                  resnet  ResNet101         0.0001  0.0   0.0   4.0   8.0  17.0
# 25                  resnet  ResNet101         0.0005  0.0   5.0   2.0   8.0  14.0
# 26                  resnet  ResNet101         0.0010  2.0   4.0  10.0   7.0   6.0
# 27                  resnet  ResNet152         0.0001  0.0   2.0   2.0  10.0  15.0
# 28                  resnet  ResNet152         0.0005  3.0   1.0   6.0   7.0  12.0
# 29                  resnet  ResNet152         0.0010  1.0   7.0   6.0  13.0   2.0
# 30                  resnet   ResNet50         0.0001  0.0   1.0   4.0   9.0  15.0
# 31                  resnet   ResNet50         0.0005  1.0   2.0   6.0   5.0  15.0
# 32                  resnet   ResNet50         0.0010  1.0   5.0   5.0  15.0   3.0
# 33                     vgg      VGG16         0.0001  4.0  10.0   2.0  12.0   1.0
# 34                     vgg      VGG19         0.0001  7.0   5.0   5.0   7.0   5.0


###########################
# Per patient by median analysis
###########################

patient_median_results = test_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test', 'patient_idx']
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
# 6          B3         0.0001  0.881667  0.972222  0.878333  0.876548    0.895279        0.900810
# 2          B1         0.0001  0.881667  0.961111  0.861667  0.894167    0.893257        0.898374
# 14         B7         0.0001  0.870000  0.977778  0.876071  0.855238    0.885786        0.892975
# 15         B7         0.0005  0.870000  0.955556  0.857143  0.866190    0.887276        0.887233
# 10         B5         0.0001  0.865000  0.961111  0.861905  0.852262    0.882285        0.884512
# 8          B4         0.0001  0.861667  0.952778  0.865476  0.841429    0.880102        0.880290
# 18          S         0.0001  0.861667  0.936111  0.855238  0.861905    0.870364        0.877057
# 5          B2         0.0005  0.855000  0.950000  0.858929  0.840952    0.869119        0.874800
# 9          B4         0.0005  0.850000  0.955556  0.836071  0.848571    0.864607        0.870961
# 4          B2         0.0001  0.846667  0.950000  0.837619  0.847143    0.855720        0.867430
# 24  ResNet101         0.0001  0.853333  0.916667  0.829762  0.861310    0.863521        0.864918
# 7          B3         0.0005  0.848333  0.927778  0.831548  0.849643    0.861414        0.863743
# 13         B6         0.0005  0.836667  0.927778  0.827381  0.824762    0.860994        0.855516
# 12         B6         0.0001  0.831667  0.950000  0.805952  0.842262    0.846955        0.855367
# 11         B5         0.0005  0.838333  0.933333  0.806786  0.838214    0.852069        0.853747
# 19          S         0.0005  0.838333  0.908333  0.819881  0.842500    0.849260        0.851662
# 0          B0         0.0001  0.830000  0.933333  0.833690  0.813571    0.841900        0.850499
# 21  Inception         0.0005  0.826667  0.950000  0.782619  0.835714    0.840841        0.847168
# 16          M         0.0001  0.818333  0.950000  0.812619  0.813929    0.838819        0.846740
# 3          B1         0.0005  0.823333  0.933333  0.789286  0.836310    0.839261        0.844305

patient_median_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(
    count_correct_prediction).reset_index().groupby(
        ['group', 'model', 'learning_rate']).min().reset_index().sort_values(
            'correct_rate', ascending=False)[['model', 'learning_rate', 'correct_rate']]

patient_median_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(
    count_correct_prediction).reset_index().groupby(
        ['group', 'model', 'learning_rate', 'correct_rate']
    ).count().reset_index()[['group', 'model', 'learning_rate', 'correct_rate', 'patient_idx']].pivot(
        index=['group', 'model', 'learning_rate'], columns='correct_rate', values='patient_idx').reset_index().fillna(0)
# correct_rate         group      model  learning_rate  0.0  0.25   0.5  0.75   1.0
# 0             efficientnet         B0         0.0001  1.0   3.0   2.0   3.0  20.0
# 1             efficientnet         B0         0.0005  1.0   1.0   4.0   6.0  17.0
# 2             efficientnet         B1         0.0001  1.0   0.0   2.0   6.0  20.0
# 3             efficientnet         B1         0.0005  0.0   2.0   4.0   7.0  16.0
# 4             efficientnet         B2         0.0001  0.0   2.0   1.0  10.0  16.0
# 5             efficientnet         B2         0.0005  2.0   0.0   1.0   7.0  19.0
# 6             efficientnet         B3         0.0001  0.0   0.0   3.0   8.0  18.0
# 7             efficientnet         B3         0.0005  0.0   2.0   3.0   6.0  18.0
# 8             efficientnet         B4         0.0001  0.0   2.0   0.0  10.0  17.0
# 9             efficientnet         B4         0.0005  0.0   0.0   6.0   6.0  17.0
# 10            efficientnet         B5         0.0001  0.0   0.0   3.0  10.0  16.0
# 11            efficientnet         B5         0.0005  0.0   1.0   3.0  10.0  15.0
# 12            efficientnet         B6         0.0001  0.0   3.0   2.0   7.0  17.0
# 13            efficientnet         B6         0.0005  0.0   1.0   5.0   6.0  17.0
# 14            efficientnet         B7         0.0001  0.0   1.0   3.0   6.0  19.0
# 15            efficientnet         B7         0.0005  0.0   1.0   2.0   8.0  18.0
# 16            efficientnet          M         0.0001  0.0   0.0   7.0   7.0  15.0
# 17            efficientnet          M         0.0005  0.0   1.0   9.0   3.0  16.0
# 18            efficientnet          S         0.0001  0.0   1.0   2.0   9.0  17.0
# 19            efficientnet          S         0.0005  1.0   1.0   3.0   6.0  18.0
# 20               inception  Inception         0.0001  0.0   1.0   5.0   9.0  14.0
# 21               inception  Inception         0.0005  0.0   0.0   5.0  10.0  14.0
# 22               mobilenet  MobileNet         0.0001  2.0   4.0   4.0   8.0  11.0
# 23               mobilenet  MobileNet         0.0005  0.0   1.0   8.0  12.0   8.0
# 24                  resnet  ResNet101         0.0001  0.0   0.0   3.0  11.0  15.0
# 25                  resnet  ResNet101         0.0005  0.0   5.0   3.0   5.0  16.0
# 26                  resnet  ResNet101         0.0010  0.0   3.0  11.0  10.0   5.0
# 27                  resnet  ResNet152         0.0001  0.0   2.0   4.0   9.0  14.0
# 28                  resnet  ResNet152         0.0005  3.0   1.0   5.0   8.0  12.0
# 29                  resnet  ResNet152         0.0010  1.0   1.0  13.0  11.0   3.0
# 30                  resnet   ResNet50         0.0001  0.0   1.0   6.0   7.0  15.0
# 31                  resnet   ResNet50         0.0005  1.0   2.0   6.0   4.0  16.0
# 32                  resnet   ResNet50         0.0010  2.0   3.0   6.0  10.0   8.0
# 33                     vgg      VGG16         0.0001  4.0  10.0   2.0  10.0   3.0
# 34                     vgg      VGG19         0.0001  7.0   5.0   6.0   5.0   6.0


###########################
# Per patient by majority voting analysis
###########################
patient_vote_results = test_preds.groupby(['group', 'model_name', 'model', 'learning_rate', 'val', 'test', 'patient_idx']
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
# 6          B3         0.0001  0.891667  0.961111  0.888333  0.886548    0.903613        0.906254
# 14         B7         0.0001  0.888333  0.958333  0.888095  0.885238    0.898744        0.903749
# 2          B1         0.0001  0.881667  0.955556  0.861667  0.894167    0.893257        0.897263
# 10         B5         0.0001  0.881667  0.916667  0.873929  0.876786    0.898126        0.889435
# 15         B7         0.0005  0.868333  0.961111  0.847619  0.869048    0.884907        0.886204
# 9          B4         0.0005  0.866667  0.938889  0.849881  0.868095    0.881274        0.880961
# 8          B4         0.0001  0.861667  0.950000  0.862619  0.844286    0.880102        0.879735
# 18          S         0.0001  0.861667  0.948611  0.855238  0.861905    0.870364        0.879557
# 24  ResNet101         0.0001  0.861667  0.911111  0.836429  0.870833    0.872865        0.870581
# 13         B6         0.0005  0.853333  0.919444  0.839881  0.849762    0.874813        0.867447
# 0          B0         0.0001  0.846667  0.947222  0.847500  0.833095    0.858567        0.866610
# 4          B2         0.0001  0.846667  0.938889  0.837619  0.847143    0.855720        0.865207
# 5          B2         0.0005  0.843333  0.948611  0.830952  0.841667    0.856235        0.864160
# 7          B3         0.0005  0.848333  0.911111  0.831548  0.849643    0.861414        0.860410
# 12         B6         0.0001  0.831667  0.955556  0.805952  0.842262    0.846955        0.856478
# 19          S         0.0005  0.838333  0.930556  0.819881  0.842500    0.849260        0.856106
# 3          B1         0.0005  0.840000  0.919444  0.801190  0.859643    0.854916        0.855039
# 11         B5         0.0005  0.828333  0.936111  0.791786  0.831548    0.839569        0.845469
# 21  Inception         0.0005  0.826667  0.938889  0.774762  0.840357    0.841666        0.844468
# 30   ResNet50         0.0001  0.823333  0.944444  0.797381  0.818095    0.834392        0.843529


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
# correct_rate         group      model  learning_rate  0.0  0.25   0.5  0.75   1.0
# 0             efficientnet         B0         0.0001  1.0   1.0   4.0   3.0  20.0
# 1             efficientnet         B0         0.0005  1.0   1.0   4.0   6.0  17.0
# 2             efficientnet         B1         0.0001  1.0   0.0   2.0   6.0  20.0
# 3             efficientnet         B1         0.0005  0.0   2.0   3.0   7.0  17.0
# 4             efficientnet         B2         0.0001  0.0   2.0   1.0  10.0  16.0
# 5             efficientnet         B2         0.0005  1.0   1.0   1.0   9.0  17.0
# 6             efficientnet         B3         0.0001  0.0   0.0   3.0   7.0  19.0
# 7             efficientnet         B3         0.0005  0.0   2.0   3.0   6.0  18.0
# 8             efficientnet         B4         0.0001  0.0   1.0   1.0  11.0  16.0
# 9             efficientnet         B4         0.0005  0.0   0.0   5.0   6.0  18.0
# 10            efficientnet         B5         0.0001  0.0   0.0   3.0   8.0  18.0
# 11            efficientnet         B5         0.0005  0.0   1.0   4.0   9.0  15.0
# 12            efficientnet         B6         0.0001  0.0   3.0   2.0   7.0  17.0
# 13            efficientnet         B6         0.0005  0.0   1.0   3.0   8.0  17.0
# 14            efficientnet         B7         0.0001  0.0   1.0   3.0   4.0  21.0
# 15            efficientnet         B7         0.0005  0.0   1.0   1.0  10.0  17.0
# 16            efficientnet          M         0.0001  0.0   0.0   7.0   9.0  13.0
# 17            efficientnet          M         0.0005  0.0   0.0  11.0   3.0  15.0
# 18            efficientnet          S         0.0001  0.0   1.0   2.0   9.0  17.0
# 19            efficientnet          S         0.0005  1.0   1.0   3.0   6.0  18.0
# 20               inception  Inception         0.0001  0.0   1.0   4.0  10.0  14.0
# 21               inception  Inception         0.0005  0.0   0.0   6.0   8.0  15.0
# 22               mobilenet  MobileNet         0.0001  2.0   4.0   5.0   7.0  11.0
# 23               mobilenet  MobileNet         0.0005  0.0   1.0   8.0  12.0   8.0
# 24                  resnet  ResNet101         0.0001  0.0   0.0   3.0  10.0  16.0
# 25                  resnet  ResNet101         0.0005  0.0   4.0   4.0   6.0  15.0
# 26                  resnet  ResNet101         0.0010  0.0   2.0  12.0  10.0   5.0
# 27                  resnet  ResNet152         0.0001  0.0   2.0   3.0  10.0  14.0
# 28                  resnet  ResNet152         0.0005  3.0   2.0   4.0   8.0  12.0
# 29                  resnet  ResNet152         0.0010  1.0   2.0  12.0  11.0   3.0
# 30                  resnet   ResNet50         0.0001  0.0   1.0   6.0   6.0  16.0
# 31                  resnet   ResNet50         0.0005  1.0   2.0   6.0   4.0  16.0
# 32                  resnet   ResNet50         0.0010  2.0   3.0   5.0  10.0   9.0
# 33                     vgg      VGG16         0.0001  4.0   9.0   3.0  10.0   3.0
# 34                     vgg      VGG19         0.0001  7.0   5.0   5.0   6.0   6.0
