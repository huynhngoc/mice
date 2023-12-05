import os
import json


filenames = [fn for fn in os.listdir(
    'config/eff') if fn.startswith('b1_lr0001')]

assert len(filenames) == 20

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    print(config['model_params']['optimizer']['config']['learning_rate'])

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    config['model_params']['optimizer']['config']['learning_rate'] = 0.0005
    new_fn = fn.replace('0001', '0005')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    config['model_params']['optimizer']['config']['learning_rate'] = 0.001
    new_fn = fn.replace('0001', '001')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    config['model_params']['optimizer']['config']['learning_rate'] = 0.005
    new_fn = fn.replace('0001', '005')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


# ===========================================================================
filenames = [fn for fn in os.listdir('config/eff') if fn.startswith('b1_')]
assert len(filenames) == 20 * 4

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    print(config['architecture']['class_name'])


for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    config['architecture']['class_name'] = 'B2'
    new_fn = fn.replace('b1', 'b2')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    config['architecture']['class_name'] = 'B3'
    new_fn = fn.replace('b1', 'b3')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    config['architecture']['class_name'] = 'B4'
    new_fn = fn.replace('b1', 'b4')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)

for fn in filenames:
    if 'lr005_' in fn:
        continue
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    config['architecture']['class_name'] = 'B4'
    new_fn = fn.replace('b1', 'b5')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)
# ===========================================================================
filenames = [fn for fn in os.listdir(
    'config/eff') if fn.startswith('b4_lr0005')]
assert len(filenames) == 20

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset_params']['config']['filename']
    config['dataset_params']['config']['filename'] = dataset_name.replace(
        'mice.h5', 'mice_AH.h5')
    new_fn = fn.replace('.json', '_AH.json')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


filenames = [fn for fn in os.listdir('config/eff') if fn.endswith('_AH.json')]
assert len(filenames) == 20

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset_params']['config']['filename']
    config['dataset_params']['config']['filename'] = dataset_name.replace(
        '_AH', '_unsharp')
    new_fn = fn.replace('_AH', '_unsharp')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


# filenames = [fn for fn in os.listdir('config/eff') if fn.endswith('_AH.json')]
# for fn in filenames:
#     new_fn = fn.replace('b4', 'AH_b4')
#     new_fn = new_fn.replace('_AH.json', '.json')
#     print(new_fn)
#     os.rename(f'config/eff/{fn}', f'config/eff/{new_fn}')


# filenames = [fn for fn in os.listdir(
#     'config/eff') if fn.endswith('_unsharp.json')]
# for fn in filenames:
#     new_fn = fn.replace('b4', 'unsharp_b4')
#     new_fn = new_fn.replace('_unsharp.json', '.json')
#     print(new_fn)
#     os.rename(f'config/eff/{fn}', f'config/eff/{new_fn}')



filenames = [fn for fn in os.listdir('config/eff') if fn.startswith('AH_')]
assert len(filenames) == 20

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset_params']['config']['filename']
    config['dataset_params']['config']['filename'] = dataset_name.replace(
        'mice_AH.h5', 'mice_3c.h5')
    new_fn = fn.replace('AH_', 'all_')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)
