import logging
from model.inputs.input_fn_byol import gen_pipeline_train_byol, gen_pipeline_eval_byol, \
    gen_pipeline_test_byol
from model import model_fn, train_byol
from utils import utils_params, utils_misc
import argparse


def set_up_train(path_model_id='', config_names=['config.gin'], bindings=[]):
    """
    Setup complete environment including datasets, models and start training
    :param path_model_id: use if continue from existing model
    :param config_names: name of config file(s) for gin config
    :param bindings: to overwrite single parameters defined in the gin config file(s)
    :return: return the last validation metric
    """
    # inject config
    utils_params.inject_gin(config_names, path_model_id=path_model_id,
                            bindings=bindings)  # bindings=['train_and_eval.n_epochs = 3','train_and_eval.save_period = 1']

    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # Define input pipeline depending on the type of training
    logging.info('Setup input pipeline...')
    train_ds, train_ds_info = gen_pipeline_train_byol()
    eval_ds, eval_ds_info = gen_pipeline_eval_byol()
    test_ds, test_info = gen_pipeline_test_byol()

    # Define model
    logging.info("Setup model...")
    online_model = model_fn.gen_model(n_classes=train_ds_info.features['label'].num_classes)
    target_model = model_fn.gen_model(n_classes=train_ds_info.features['label'].num_classes)

    # Train and eval
    logging.info('Start training...')
    results = train_byol.train_and_eval_byol(online_model, target_model, train_ds, train_ds_info, eval_ds, test_ds,
                                             run_paths)
    return results


if __name__ == '__main__':
    # Define model path
    parser = argparse.ArgumentParser(description='BYOL training.')
    parser.add_argument('--path', type=str, default='', required=False, help='Result path')
    args = parser.parse_args()

    if args.path:
        path_model_id = args.path
    else:
        path_model_id = ''

    # gin config files and bindings
    config_names = ['configByol.gin']
    # Get bindings
    bindings = []

    # start training
    set_up_train(path_model_id=path_model_id, config_names=config_names, bindings=bindings)
