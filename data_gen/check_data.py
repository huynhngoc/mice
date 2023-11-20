import os
import numpy as np
import pandas as pd

path = 'D:/OUS_mice/Classes_from_experiments_DICOM/'
control_path = path + 'Control'
irrad_path = path + 'Irradiated'


os.listdir(control_path)

control_df = pd.DataFrame(np.array([fn[8: -4].split('_') for fn in os.listdir(control_path)]).astype(int),
                          columns=['group', 'name', 'slice']).sort_values(['group', 'name', 'slice']).reset_index(drop=True)


irrad_df = pd.DataFrame(np.array([fn[4: -4].split('_') for fn in os.listdir(irrad_path)]).astype(int),
                        columns=['group', 'name', 'slice']).sort_values(['group', 'name', 'slice']).reset_index(drop=True)

irrad_df


control_df['class'] = 'control'
irrad_df['class'] = 'irradiated'

pd.concat([control_df, irrad_df]).reset_index(drop=True).to_csv(
    'data_info/info_tmp.csv', index_label='pid')


control_df = pd.DataFrame(np.array([fn[8: -4].split('_') for fn in os.listdir(control_path)]).astype(int),
                          columns=['group', 'name', 'slice']).groupby(['group', 'name']).count().reset_index()


irrad_df = pd.DataFrame(np.array([fn[4: -4].split('_') for fn in os.listdir(irrad_path)]).astype(int),
                        columns=['group', 'name', 'slice']).groupby(['group', 'name']).count().reset_index()

control_df
irrad_df

control_df['class'] = 'control'
irrad_df['class'] = 'irradiated'


df = pd.concat([control_df[['group', 'name', 'class']], irrad_df[['group', 'name', 'class']]]
          ).reset_index(drop=True)
df.index = df.index + 1

df.to_csv('data_info/info_official.csv', index_label='pid')


#===========================================================================================================================
path = 'D:/OUS_mice/Visually_Filtered_300_per_class_Jpeg/'
control_path = path + 'control'
irrad_path = path + 'irradiated'


os.listdir(control_path)

control_df = pd.DataFrame(np.array([fn[8: -4].split('_') for fn in os.listdir(control_path)]).astype(int),
                          columns=['group', 'name', 'slice']).sort_values(['group', 'name', 'slice']).reset_index(drop=True)


irrad_df = pd.DataFrame(np.array([fn[4: -4].split('_') for fn in os.listdir(irrad_path)]).astype(int),
                        columns=['group', 'name', 'slice']).sort_values(['group', 'name', 'slice']).reset_index(drop=True)


control_df['class'] = 'control'
irrad_df['class'] = 'irradiated'
pd.concat([control_df, irrad_df])

pd.read_csv('data_info/info_official.csv').merge(
    pd.concat([control_df, irrad_df]), how='left', on=['group', 'name', 'class']
    ).to_csv('data_info/info_selected.csv', index=False)
