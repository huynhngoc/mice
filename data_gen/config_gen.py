import os
import json
import itertools

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


filenames = [fn for fn in os.listdir('config/eff') if fn.startswith('AH_')]
assert len(filenames) == 20

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset_params']['config']['filename']
    config['dataset_params']['config']['filename'] = dataset_name.replace(
        'mice_AH.h5', 'mice_MH.h5')
    new_fn = fn.replace('AH_', 'MH_')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


filenames = [fn for fn in os.listdir('config/eff') if fn.startswith('AH_')]
assert len(filenames) == 20

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset_params']['config']['filename']
    config['dataset_params']['config']['filename'] = dataset_name.replace(
        'mice_AH.h5', 'mice_MH_3c.h5')
    new_fn = fn.replace('AH_', 'MH_3c_')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


filenames = [fn for fn in os.listdir('config/eff') if fn.startswith('all_b4_')]
assert len(filenames) == 20

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    config['architecture']['class_name'] = 'B4'
    new_fn = fn.replace('b4', 'b1')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


filenames = [fn for fn in os.listdir('config/eff') if fn.startswith('AH_')]
assert len(filenames) == 20

for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset_params']['config']['filename']
    config['dataset_params']['config']['filename'] = dataset_name.replace(
        'mice_AH.h5', 'mice_MH_AH.h5')
    new_fn = fn.replace('AH_', 'MH_AH_')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)


for fn in filenames:
    with open(f'config/eff/{fn}', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset_params']['config']['filename']
    config['dataset_params']['config']['filename'] = dataset_name.replace(
        'mice_AH.h5', 'mice_MH_unsharp.h5')
    new_fn = fn.replace('AH_', 'MH_unsharp_')
    with open(f'config/eff/{new_fn}', 'w') as f:
        json.dump(config, f)

fold_list = []
for i in range(4, -1, -1):
    for j in range(4, -1, -1):
        if i == j:
            continue
        fold_list.append([k for k in range(5) if k != i and k != j] + [j, i])

# efficientnet group
preprocessor_group = 'efficientnet'
learning_rates = ['0001', '0005', '001', '005']
model_names = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'S', 'M', 'L']
with open('config/templates/pretrained_template.json', 'r') as f:
    template = json.load(f)

for lr, model in itertools.product(learning_rates, model_names):
    base_template = template.copy()
    base_template['architecture']['class_name'] = model
    base_template['model_params']['optimizer']['config']['learning_rate'] = float(
        f'0.{lr}')
    base_template['dataset_params']['config']['preprocessors'][-1]['config']['model'] = preprocessor_group

    for fold in fold_list:
        base_template['dataset_params']['config']['train_folds'] = fold[:-2]
        base_template['dataset_params']['config']['val_folds'] = fold[-2:-1]
        base_template['dataset_params']['config']['test_folds'] = fold[-1:]
        filename = f'config/{preprocessor_group}/{model}_lr{lr}_f{"".join(map(str,fold))}.json'
        with open(filename, 'w') as f:
            json.dump(base_template, f)

# resnet group
preprocessor_group = 'resnet'
learning_rates = ['0001', '0005', '001', '005']
model_names = ['ResNet50', 'ResNet101', 'ResNet152']
with open('config/templates/pretrained_template.json', 'r') as f:
    template = json.load(f)
for lr, model in itertools.product(learning_rates, model_names):
    base_template = template.copy()
    base_template['architecture']['class_name'] = model
    base_template['model_params']['optimizer']['config']['learning_rate'] = float(
        f'0.{lr}')
    base_template['dataset_params']['config']['preprocessors'][-1]['config']['model'] = preprocessor_group

    for fold in fold_list:
        base_template['dataset_params']['config']['train_folds'] = fold[:-2]
        base_template['dataset_params']['config']['val_folds'] = fold[-2:-1]
        base_template['dataset_params']['config']['test_folds'] = fold[-1:]
        filename = f'config/{preprocessor_group}/{model}_lr{lr}_f{"".join(map(str,fold))}.json'
        with open(filename, 'w') as f:
            json.dump(base_template, f)

# vgg group
preprocessor_group = 'vgg'
learning_rates = ['0001', '0005', '001', '005']
model_names = ['VGG16', 'VGG19']
with open('config/templates/pretrained_template.json', 'r') as f:
    template = json.load(f)
for lr, model in itertools.product(learning_rates, model_names):
    base_template = template.copy()
    base_template['architecture']['class_name'] = model
    base_template['model_params']['optimizer']['config']['learning_rate'] = float(
        f'0.{lr}')
    base_template['dataset_params']['config']['preprocessors'][-1]['config']['model'] = model.lower()

    for fold in fold_list:
        base_template['dataset_params']['config']['train_folds'] = fold[:-2]
        base_template['dataset_params']['config']['val_folds'] = fold[-2:-1]
        base_template['dataset_params']['config']['test_folds'] = fold[-1:]
        filename = f'config/{preprocessor_group}/{model}_lr{lr}_f{"".join(map(str,fold))}.json'
        with open(filename, 'w') as f:
            json.dump(base_template, f)

# mobilenet group
preprocessor_group = 'mobilenet'
learning_rates = ['0001', '0005', '001', '005']
model_names = ['MobileNet']
with open('config/templates/pretrained_template.json', 'r') as f:
    template = json.load(f)
for lr, model in itertools.product(learning_rates, model_names):
    base_template = template.copy()
    base_template['architecture']['class_name'] = model
    base_template['model_params']['optimizer']['config']['learning_rate'] = float(
        f'0.{lr}')
    base_template['dataset_params']['config']['preprocessors'][-1]['config']['model'] = preprocessor_group

    for fold in fold_list:
        base_template['dataset_params']['config']['train_folds'] = fold[:-2]
        base_template['dataset_params']['config']['val_folds'] = fold[-2:-1]
        base_template['dataset_params']['config']['test_folds'] = fold[-1:]
        filename = f'config/{preprocessor_group}/{model}_lr{lr}_f{"".join(map(str,fold))}.json'
        with open(filename, 'w') as f:
            json.dump(base_template, f)

# inception group
preprocessor_group = 'inception'
learning_rates = ['0001', '0005', '001', '005']
model_names = ['Inception']
with open('config/templates/pretrained_template.json', 'r') as f:
    template = json.load(f)
for lr, model in itertools.product(learning_rates, model_names):
    base_template = template.copy()
    base_template['architecture']['class_name'] = model
    base_template['model_params']['optimizer']['config']['learning_rate'] = float(
        f'0.{lr}')
    base_template['dataset_params']['config']['preprocessors'][-1]['config']['model'] = preprocessor_group

    for fold in fold_list:
        base_template['dataset_params']['config']['train_folds'] = fold[:-2]
        base_template['dataset_params']['config']['val_folds'] = fold[-2:-1]
        base_template['dataset_params']['config']['test_folds'] = fold[-1:]
        filename = f'config/{preprocessor_group}/{model}_lr{lr}_f{"".join(map(str,fold))}.json'
        with open(filename, 'w') as f:
            json.dump(base_template, f)
