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
