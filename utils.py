# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalized data
(3) rounding: Handle categorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''

# Necessary packages
import numpy as np
import tensorflow.compat.v1 as tf

# Disable TensorFlow v2 behaviors
tf.disable_v2_behavior()

def normalization(data, parameters=None):
    '''Normalize data in [0, 1] range.'''
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] -= min_val[i]
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] /= (max_val[i] + 1e-6)
        norm_parameters = {'min_val': min_val, 'max_val': max_val}
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        for i in range(dim):
            norm_data[:, i] -= min_val[i]
            norm_data[:, i] /= (max_val[i] + 1e-6)
        norm_parameters = parameters

    return norm_data, norm_parameters

def renormalization(norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.'''
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] *= (max_val[i] + 1e-6)
        renorm_data[:, i] += min_val[i]

    return renorm_data

def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.'''
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    return rounded_data

def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between original and imputed data.'''
    ori_data, norm_params = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_params)
    nom = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denom = np.sum(1 - data_m)
    return np.sqrt(nom / float(denom))

def xavier_init(size):
    '''Xavier initialization.'''
    in_dim = size[0]
    stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=stddev)

def binary_sampler(p, rows, cols):
    '''Sample binary random variables.'''
    return (np.random.uniform(0., 1., size=[rows, cols]) < p).astype(np.float32)

def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.'''
    return np.random.uniform(low, high, size=[rows, cols])

def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.'''
    return np.random.permutation(total)[:batch_size]

