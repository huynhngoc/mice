import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score

# Load data
test_preds = pd.read_csv('analysis_results/test_preds.csv')
ensemble_preds = test_preds.groupby(['group', 'model', 'learning_rate', 'patient_idx', 'slice_idx']).agg({'y': 'mean', 'predicted': 'mean'}).reset_index()

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
grouped_metrics = ensemble_preds.groupby(['group', 'model', 'learning_rate']
    ).apply(calculate_metrics).reset_index()

# calculate the weighted score
grouped_metrics['weighted_score'] = (grouped_metrics['accuracy'] + grouped_metrics['auc'] + grouped_metrics['f1'] + grouped_metrics['f1_0'] + grouped_metrics['scaled_mcc']) / 5


grouped_metrics.sort_values('weighted_score', ascending=False).head(20)[['model', 'learning_rate', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']]
#         model  learning_rate  accuracy       auc        f1      f1_0  scaled_mcc  weighted_score
# 6          B3         0.0001  0.832765  0.914159  0.831615  0.833898    0.833543        0.849196
# 15         B7         0.0005  0.813993  0.898621  0.814310  0.813675    0.815249        0.831170
# 11         B5         0.0005  0.812287  0.905632  0.810345  0.814189    0.812873        0.831065
# 9          B4         0.0005  0.813993  0.894130  0.808436  0.819237    0.813896        0.829938
# 7          B3         0.0005  0.805461  0.896918  0.789668  0.819048    0.805504        0.823320
# 1          B0         0.0005  0.805461  0.888485  0.806780  0.804124    0.807041        0.822378
# 14         B7         0.0001  0.800341  0.900254  0.804020  0.796522    0.802818        0.820791
# 5          B2         0.0005  0.800341  0.884355  0.802698  0.797927    0.802278        0.817520
# 16          M         0.0001  0.802048  0.878744  0.797909  0.806020    0.802199        0.817384
# 24  ResNet101         0.0001  0.800341  0.885428  0.795096  0.805324    0.800325        0.817303
# 19          S         0.0005  0.798635  0.884145  0.786232  0.809677    0.798291        0.815396
# 13         B6         0.0005  0.793515  0.886140  0.782765  0.803252    0.793122        0.811759
# 8          B4         0.0001  0.793515  0.874813  0.796639  0.790295    0.795694        0.810191
# 4          B2         0.0001  0.786689  0.880296  0.776386  0.796085    0.786293        0.805150
# 27  ResNet152         0.0001  0.788396  0.871664  0.785467  0.791246    0.788807        0.805116
# 2          B1         0.0001  0.788396  0.874195  0.767790  0.805643    0.788812        0.804967
# 18          S         0.0001  0.783276  0.888998  0.771171  0.794165    0.782854        0.804093
# 10         B5         0.0001  0.786689  0.866391  0.789916  0.783362    0.788848        0.803041
# 30   ResNet50         0.0001  0.779863  0.884448  0.770870  0.788177    0.779526        0.800577
# 12         B6         0.0001  0.781570  0.875408  0.759398  0.800000    0.782066        0.799689


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

patient_mean_results = ensemble_preds.groupby(['group', 'model', 'learning_rate', 'patient_idx']
    ).agg({
        'slice_idx': 'count',
        'y': 'mean',
        'predicted': 'mean'
    }).reset_index()
grouped_patients_metrics = patient_mean_results.groupby(['group', 'model', 'learning_rate']
    ).apply(calculate_metrics).reset_index()
grouped_patients_metrics['weighted_score'] = (grouped_patients_metrics['accuracy'] + grouped_patients_metrics['auc'] + grouped_patients_metrics['f1'] + grouped_patients_metrics['f1_0'] + grouped_patients_metrics['scaled_mcc']) / 5
grouped_patients_metrics.sort_values('weighted_score', ascending=False).head(20)[['model', 'learning_rate', 'accuracy', 'auc', 'f1', 'f1_0', 'scaled_mcc', 'weighted_score']]
#         model  learning_rate  accuracy       auc        f1      f1_0  scaled_mcc  weighted_score
# 15         B7         0.0005  0.965517  1.000000  0.965517  0.965517    0.966667        0.972644
# 9          B4         0.0005  0.965517  1.000000  0.962963  0.967742    0.966513        0.972547
# 13         B6         0.0005  0.965517  1.000000  0.962963  0.967742    0.966513        0.972547
# 6          B3         0.0001  0.965517  0.995238  0.965517  0.965517    0.966667        0.971691
# 21  Inception         0.0005  0.965517  0.995238  0.962963  0.967742    0.966513        0.971595
# 24  ResNet101         0.0001  0.965517  0.990476  0.965517  0.965517    0.966667        0.970739
# 18          S         0.0001  0.965517  0.985714  0.965517  0.965517    0.966667        0.969787
# 2          B1         0.0001  0.965517  0.976190  0.962963  0.967742    0.966513        0.967785
# 11         B5         0.0005  0.931034  0.995238  0.933333  0.928571    0.935412        0.944718
# 16          M         0.0001  0.931034  0.990476  0.933333  0.928571    0.935412        0.943765
# 8          B4         0.0001  0.931034  0.990476  0.933333  0.928571    0.935412        0.943765
# 10         B5         0.0001  0.931034  0.990476  0.933333  0.928571    0.935412        0.943765
# 14         B7         0.0001  0.931034  0.990476  0.933333  0.928571    0.935412        0.943765
# 4          B2         0.0001  0.931034  0.980952  0.933333  0.928571    0.935412        0.941861
# 20  Inception         0.0001  0.931034  0.980952  0.933333  0.928571    0.935412        0.941861
# 1          B0         0.0005  0.931034  0.971429  0.933333  0.928571    0.935412        0.939956
# 0          B0         0.0001  0.896552  0.985714  0.903226  0.888889    0.905840        0.916044
# 30   ResNet50         0.0001  0.896552  0.985714  0.896552  0.896552    0.897619        0.914598
# 12         B6         0.0001  0.896552  0.980952  0.888889  0.903226    0.897134        0.913351
# 3          B1         0.0005  0.896552  0.980952  0.888889  0.903226    0.897134        0.913351


patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(count_correct_prediction).reset_index()
patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(count_correct_prediction).reset_index().groupby(['group', 'model', 'learning_rate']).min().reset_index().sort_values('correct_rate', ascending=False)[['model', 'learning_rate', 'correct_rate']]

# find patient idx that always predicted wrong
patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(count_correct_prediction).reset_index().groupby('patient_idx').agg({'correct': 'sum', 'total': 'sum'}).reset_index().sort_values('correct', ascending=True)
#     patient_idx  correct  total
# 2             3     15.0   35.0
# 17           18     16.0   35.0
# 3             4     24.0   35.0
# 25           26     24.0   35.0
# 18           19     24.0   35.0
# 11           12     27.0   35.0
# 21           22     27.0   35.0
patient_mean_results[patient_mean_results.patient_idx.isin([3])].head(50)
#             group      model  learning_rate  patient_idx  slice_idx    y  predicted
# 2    efficientnet         B0         0.0001            3         20  0.0   0.609569
# 31   efficientnet         B0         0.0005            3         20  0.0   0.691136
# 60   efficientnet         B1         0.0001            3         20  0.0   0.420614
# 89   efficientnet         B1         0.0005            3         20  0.0   0.502456
# 118  efficientnet         B2         0.0001            3         20  0.0   0.633197
# 147  efficientnet         B2         0.0005            3         20  0.0   0.734939
# 176  efficientnet         B3         0.0001            3         20  0.0   0.570770
# 205  efficientnet         B3         0.0005            3         20  0.0   0.504946


patient_mean_results.groupby(['group', 'model', 'learning_rate', 'patient_idx']).apply(
    count_correct_prediction).reset_index().groupby(
        ['group', 'model', 'learning_rate', 'correct_rate']
    ).count().reset_index()[['group', 'model', 'learning_rate', 'correct_rate', 'patient_idx']].pivot(
        index=['group', 'model', 'learning_rate'], columns='correct_rate', values='patient_idx').reset_index().fillna(0)
# correct_rate         group      model  learning_rate  0.0  1.0
# 0             efficientnet         B0         0.0001    3   26
# 1             efficientnet         B0         0.0005    2   27
# 2             efficientnet         B1         0.0001    1   28
# 3             efficientnet         B1         0.0005    3   26
# 4             efficientnet         B2         0.0001    2   27
# 5             efficientnet         B2         0.0005    3   26
# 6             efficientnet         B3         0.0001    1   28
# 7             efficientnet         B3         0.0005    5   24
# 8             efficientnet         B4         0.0001    2   27
# 9             efficientnet         B4         0.0005    1   28
# 10            efficientnet         B5         0.0001    2   27
# 11            efficientnet         B5         0.0005    2   27
# 12            efficientnet         B6         0.0001    3   26
# 13            efficientnet         B6         0.0005    1   28
# 14            efficientnet         B7         0.0001    2   27
# 15            efficientnet         B7         0.0005    1   28
# 16            efficientnet          M         0.0001    2   27
# 17            efficientnet          M         0.0005    4   25
# 18            efficientnet          S         0.0001    1   28
# 19            efficientnet          S         0.0005    4   25
# 20               inception  Inception         0.0001    2   27
# 21               inception  Inception         0.0005    1   28
# 22               mobilenet  MobileNet         0.0001    8   21
# 23               mobilenet  MobileNet         0.0005    3   26
# 24                  resnet  ResNet101         0.0001    1   28
# 25                  resnet  ResNet101         0.0005    5   24
# 26                  resnet  ResNet101         0.0010    9   20
# 27                  resnet  ResNet152         0.0001    3   26
# 28                  resnet  ResNet152         0.0005    7   22
# 29                  resnet  ResNet152         0.0010    9   20
# 30                  resnet   ResNet50         0.0001    3   26
# 31                  resnet   ResNet50         0.0005    6   23
# 32                  resnet   ResNet50         0.0010    7   22
# 33                     vgg      VGG16         0.0001   15   14
# 34                     vgg      VGG19         0.0001   16   13
