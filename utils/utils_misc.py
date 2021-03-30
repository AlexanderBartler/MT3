import logging
import tensorflow as tf
import json


def set_loggers(path_log=None, logging_level=0, b_stream=True, b_debug=False):
    """
    Setup logger
    :param path_log:
    :param logging_level:
    :param b_stream:
    :param b_debug:
    :return:
    """
    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)
        logger_tf.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)


def clear_loggers():
    # get logger and remove handler
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger = tf.get_logger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])






def save_result_json(filename, results):
    for key, value in results.items():
        for metric, value_m in value.items():
            results[key][metric] = str(value_m)
    with open(filename, 'w') as f:
        json.dump(results, f,indent=4)
