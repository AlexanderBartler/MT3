import logging
from model.inputs.input_fn_meta import gen_pipeline_train_meta, gen_pipeline_eval_meta
from model import model_fn, train_meta_obj
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
    train_ds, train_ds_info, repetition = gen_pipeline_train_meta()
    eval_ds, eval_ds_info = gen_pipeline_eval_meta()

    # Define model
    logging.info("Setup model...")
    model = model_fn.gen_model # not passing an instance, since multiple instances are needed later

    # Train and eval
    logging.info('Init M3T trainer...')
    trainer = train_meta_obj.MT3Trainer(model, train_ds, train_ds_info, eval_ds, run_paths, repetition)
    logging.info('Start M3T trainer...')
    results = trainer.train()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MT3 training.')
    parser.add_argument('--path',type=str, default=[], required=False, help='result folder')
    args = parser.parse_args()

    if args.path:
        path_model_id = args.path
    else:
        path_model_id = ''

    # gin config files and bindings
    config_names = ['configMT3.gin']
    bindings = []

    # start training
    results = set_up_train(path_model_id=path_model_id, config_names=config_names, bindings=bindings)
