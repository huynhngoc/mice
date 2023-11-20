import os
import json


filenames = [fn for fn in os.listdir('config/eff') if fn.startswith('b1_lr0001')]

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
