from deoxys.customize import custom_architecture, custom_preprocessor
from deoxys.loaders.architecture import BaseModelLoader
from deoxys.data.preprocessor import BasePreprocessor
from deoxys.experiment.postprocessor import DefaultPostProcessor
from deoxys.utils import deep_copy

from tensorflow.keras.applications import efficientnet_v2, resnet_v2, vgg16, vgg19, mobilenet_v2, inception_v3
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model

import numpy as np
import h5py
import os
import shutil

@custom_architecture
class PretrainModelLoader(BaseModelLoader):
    map_name = {
        'B0': efficientnet_v2.EfficientNetV2B0,
        'B1': efficientnet_v2.EfficientNetV2B1,
        'B2': efficientnet_v2.EfficientNetV2B2,
        'B3': efficientnet_v2.EfficientNetV2B3,
        'B4': efficientnet_v2.EfficientNetV2B4,
        'B5': efficientnet_v2.EfficientNetV2B5,
        'B6': efficientnet_v2.EfficientNetV2B6,
        'B7': efficientnet_v2.EfficientNetV2B7,
        'L': efficientnet_v2.EfficientNetV2L,
        'M': efficientnet_v2.EfficientNetV2M,
        'S': efficientnet_v2.EfficientNetV2S,
        'ResNet50': resnet_v2.ResNet50V2,
        'ResNet101': resnet_v2.ResNet101V2,
        'ResNet152': resnet_v2.ResNet152V2,
        'VGG16': vgg16.VGG16,
        'VGG19': vgg19.VGG19,
        'MobileNet': mobilenet_v2.MobileNetV2,
        'InceptionV3': inception_v3.InceptionV3,

    }

    def __init__(self, architecture, input_params):
        self._input_params = deep_copy(input_params)
        self.options = architecture

    def load(self):
        """

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network of sequential layers
            from the configured layer list.
        """
        num_class = self.options['num_class']
        shape = self._input_params['shape']
        pretrain_model = self.map_name[self.options['class_name']]

        if num_class <= 2:
            num_class = 1
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        model = pretrain_model(include_top=False, classes=num_class,
                                classifier_activation=activation, input_shape=shape, pooling='avg')
        dropout_out = Dropout(0.3)(model.output)
        pred = Dense(num_class, activation=activation)(dropout_out)
        model = Model(model.inputs, pred)

        return model


@custom_architecture
class EfficientNetModelLoader(BaseModelLoader):
    """
    Create a sequential network from list of layers
    """
    map_name = {
        'B0': efficientnet_v2.EfficientNetV2B0,
        'B1': efficientnet_v2.EfficientNetV2B1,
        'B2': efficientnet_v2.EfficientNetV2B2,
        'B3': efficientnet_v2.EfficientNetV2B3,
        'B4': efficientnet_v2.EfficientNetV2B4,
        'B5': efficientnet_v2.EfficientNetV2B5,
        'B6': efficientnet_v2.EfficientNetV2B6,
        'B7': efficientnet_v2.EfficientNetV2B7,
        'L': efficientnet_v2.EfficientNetV2L,
        'M': efficientnet_v2.EfficientNetV2M,
        'S': efficientnet_v2.EfficientNetV2S,
    }

    def __init__(self, architecture, input_params):
        self._input_params = deep_copy(input_params)
        self.options = architecture

    def load(self):
        """

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network of sequential layers
            from the configured layer list.
        """
        num_class = self.options['num_class']
        pretrained = self.options['pretrained']
        shape = self._input_params['shape']
        efficientNet = self.map_name[self.options['class_name']]

        if num_class <= 2:
            num_class = 1
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        if pretrained:
            model = efficientNet(include_top=False, classes=num_class,
                                 classifier_activation=activation, input_shape=shape, pooling='avg')
            dropout_out = Dropout(0.3)(model.output)
            pred = Dense(num_class, activation=activation)(dropout_out)
            model = Model(model.inputs, pred)
        else:
            model = efficientNet(weights=None, include_top=True, classes=num_class,
                                 classifier_activation=activation, input_shape=shape)

        return model



@custom_preprocessor
class RGBConverter(BasePreprocessor):
    def transform(self, images, targets=None):
        # efficientNet requires input between [0-255]
        images = images * 255
        # pretrain require 3 channel
        if images.shape[-1] == 1:
            new_images = np.concatenate([images, images, images], axis=-1)
        elif images.shape[-1] == 3:
            new_images = images
        else:
            raise ValueError(
                'Input image must have either 1 channel or 3 channel')
        if targets is None:
            return new_images
        else:
            return new_images, targets

@custom_preprocessor
class PretrainedModelPreprocessor(BasePreprocessor):
    map_name = {
        'resnet': resnet_v2.preprocess_input,
        'vgg16': vgg16.preprocess_input,
        'vgg19': vgg19.preprocess_input,
        'mobilenet': mobilenet_v2.preprocess_input,
        'inception': inception_v3.preprocess_input,
        'efficientnet': efficientnet_v2.preprocess_input,
    }

    def __init__(self, model='efficientnet'):
        self.model = model

    def transform(self, images, targets=None):
        images = np.copy(images)
        if self.model in self.map_name:
            images = self.map_name[self.model](images)
        if targets is None:
            return images
        return images, targets


@custom_preprocessor
class PretrainedEfficientNet(BasePreprocessor):
    def transform(self, images, targets):
        # efficientNet requires input between [0-255]
        images = images * 255
        # pretrain require 3 channel
        if images.shape[-1] == 1:
            new_images = np.concatenate([images, images, images], axis=-1)
        elif images.shape[-1] == 3:
            new_images = images
        else:
            raise ValueError(
                'Input image must have either 1 channel or 3 channel')

        return new_images, targets


@custom_preprocessor
class OneHot(BasePreprocessor):
    def __init__(self, num_class=2):
        if num_class <= 2:
            num_class = 1
        self.num_class = num_class

    def transform(self, images, targets):
        # labels to one-hot encode
        new_targets = np.zeros((len(targets), self.num_class))
        if self.num_class == 1:
            new_targets[..., 0] = targets
        else:
            for i in range(self.num_class):
                new_targets[..., i][targets == i] = 1

        return images, new_targets


class EnsemblePostProcessor(DefaultPostProcessor):
    def __init__(self, log_base_path='logs',
                 log_path_list=None,
                 map_meta_data=None, **kwargs):

        self.log_base_path = log_base_path
        self.log_path_list = []
        for path in log_path_list:
            merge_file = path + self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            if os.path.exists(merge_file):
                self.log_path_list.append(merge_file)
            else:
                print('Missing file from', path)

        # check if there are more than 1 to ensemble
        assert len(self.log_path_list) > 1, 'Cannot ensemble with 0 or 1 item'

        if map_meta_data:
            if type(map_meta_data) == str:
                self.map_meta_data = map_meta_data.split(',')
            else:
                self.map_meta_data = map_meta_data
        else:
            self.map_meta_data = ['patient_idx']

        # always run test
        self.run_test = True

    def ensemble_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        output_file = output_folder + self.PREDICT_TEST_NAME
        if not os.path.exists(output_file):
            print('Copying template for output file')
            shutil.copy(self.log_path_list[0], output_folder)

        print('Creating ensemble results...')
        y_preds = []
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                y_preds.append(hf['predicted'][:])

        with h5py.File(output_file, 'a') as mf:
            mf['predicted'][:] = np.mean(y_preds, axis=0)
        print('Ensembled results saved to file')

        return self

    def concat_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        # first check the template
        with h5py.File(self.log_path_list[0], 'r') as f:
            ds_names = list(f.keys())
        ds = {name: [] for name in ds_names}

        # get the data
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                for key in ds:
                    ds[key].append(hf[key][:])

        # now merge them
        print('creating merged file')
        output_file = output_folder + self.PREDICT_TEST_NAME
        with h5py.File(output_file, 'w') as mf:
            for key, val in ds.items():
                mf.create_dataset(key, data=np.concatenate(val, axis=0))
