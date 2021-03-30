import os
import datetime
import logging
import gin
import tensorflow as tf

@gin.configurable(denylist = ['path_model_id'])
def gen_run_folder(path_model_id='', target='models'):
    """
    Generate folder structure for saving results
    :param path_model_id: if not start from scratch
    :return: path dict
    """
    run_paths = dict()
    if not os.path.isdir(path_model_id):
        path_model_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'experiments', target))
        date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        run_id = 'run_' + date_creation
        run_paths = dict()
        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
    else:
        run_paths['path_model_id'] = path_model_id
    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'train', 'run.log')
    run_paths['path_logs_eval'] = os.path.join(run_paths['path_model_id'], 'logs', 'eval', 'run.log')
    run_paths['path_logs_test'] = os.path.join(run_paths['path_model_id'], 'logs', 'test', 'run.log')
    run_paths['path_graphs_train'] = os.path.join(run_paths['path_model_id'], 'graphs')
    # run_paths['path_graphs_eval'] = os.path.join(run_paths['path_model_id'], 'graphs', 'eval')
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
    # run_paths['path_ckpts_eval'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'eval')
    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')

    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ['path_model', 'path_graphs', 'path_ckpts']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass  # atm file creation is sufficient

    return run_paths


def inject_gin(config_names, bindings=(), path_model_id=''):
    """
    Inject gin file and bindings
    :param config_names:
    :param bindings: additional config changes as list of strings, e.g ['evaluate.num_episodes_eval = 10']
    :return:
    """
    # parse config.gin
    configs = list()
    for conf_name_ in config_names:
        if os.path.isfile(os.path.join(path_model_id, conf_name_)):
            configs.append(os.path.abspath(os.path.join(path_model_id, conf_name_)))
        else:
            configs.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'configs', conf_name_)))

    logging.info(f'Injecting gin configs: {configs}.')
    gin.parse_config_files_and_bindings(configs, bindings=bindings, skip_unknown=True)


# Simple wrapper functions - as gin-config is not fully tf2.0 rdy for saving / loading
def save_gin(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)
