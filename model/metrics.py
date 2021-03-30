import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K


def prep_metrics():
    metrics_ = {
        'accuracy': tf.keras.metrics.CategoricalAccuracy(),  # one hot
        'precision': tf.keras.metrics.Precision(),  # one hot
        'recall': tf.keras.metrics.Recall()}  # one hot

    return metrics_


def prep_metrics_meta():
    metrics_ = {
        'pre_update_accuracy': tf.keras.metrics.CategoricalAccuracy(),
        'updated_accuracy': tf.keras.metrics.CategoricalAccuracy()  # one hot
    }
    return metrics_


def update_state_meta(metrics_, labels, final_predictions, pre_predictions):
    metrics_['pre_update_accuracy'].update_state(labels, pre_predictions)
    metrics_['updated_accuracy'].update_state(labels, final_predictions)


def update_state(metrics_, labels, predictions):
    metrics_['accuracy'].update_state(labels, predictions)
    metrics_['precision'].update_state(labels, predictions)
    metrics_['recall'].update_state(labels, predictions)


def result_meta(metrics_, as_numpy=False):
    metrics_res = dict()
    if as_numpy:
        metrics_res['pre_update_accuracy'] = metrics_['pre_update_accuracy'].result().numpy()
        metrics_res['updated_accuracy'] = metrics_['updated_accuracy'].result().numpy()

    else:
        metrics_res['pre_update_accuracy'] = metrics_['pre_update_accuracy'].result()
        metrics_res['updated_accuracy'] = metrics_['updated_accuracy'].result()

    return metrics_res


def result(metrics_, as_numpy=False):
    metrics_res = dict()
    if as_numpy:
        metrics_res['accuracy'] = metrics_['accuracy'].result().numpy()
        metrics_res['precision'] = metrics_['precision'].result().numpy()
        metrics_res['recall'] = metrics_['recall'].result().numpy()
    else:
        metrics_res['accuracy'] = metrics_['accuracy'].result()
        metrics_res['precision'] = metrics_['precision'].result()
        metrics_res['recall'] = metrics_['recall'].result()

    return metrics_res


def reset_states(metrics_):
    [met.reset_states() for met in metrics_.values()]
