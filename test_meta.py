import logging
from model.inputs.input_fn_meta import gen_pipeline_test_meta
from model import model_fn, test_meta
from utils import utils_params, utils_misc
import os
import argparse
from prettytable import PrettyTable


def run_test(dataset='', run_paths=''):
    """
    Setup complete environment including datasets, models and start testing
    """

    # Define input pipeline depending on the type of training
    logging.info('Setup input pipeline...')
    test_ds, test_info = gen_pipeline_test_meta(ds_name=dataset)

    # Define model
    logging.info("Setup model...")
    target_model = model_fn.gen_model(n_classes=test_info.features['label'].num_classes)
    online_model = model_fn.gen_model(n_classes=test_info.features['label'].num_classes)

    # Train and eval
    logging.info('Start testing...')

    result = test_meta.test_meta(target_model, online_model, test_ds, test_info, run_paths)
    del target_model, online_model
    return result


if __name__ == '__main__':
    # Define model path
    parser = argparse.ArgumentParser(description='Meta Test training.')
    parser.add_argument('--path', type=str, default='', required=False, help='Result path')
    args = parser.parse_args()

    if args.path:
        path_model_id = args.path
    else:
        path_model_id = '/home/m134/Documents/Arbeit/UnsupervisedRepresentationLearning/Models/MTTT/MT3/checkpoints/mt3/run1/'  # define model for test

    LEVEL = 5
    test_datasets = [f'cifar10_corrupted/brightness_{LEVEL}',
                     f'cifar10_corrupted/contrast_{LEVEL}',
                     f'cifar10_corrupted/defocus_blur_{LEVEL}']
                     #f'cifar10_corrupted/elastic_{LEVEL}',
                     #f'cifar10_corrupted/fog_{LEVEL}',
                     #f'cifar10_corrupted/frost_{LEVEL}',
                     #f'cifar10_corrupted/frosted_glass_blur_{LEVEL}',
                     #f'cifar10_corrupted/gaussian_blur_{LEVEL}',
                     #f'cifar10_corrupted/impulse_noise_{LEVEL}',
                     #f'cifar10_corrupted/jpeg_compression_{LEVEL}',
                     #f'cifar10_corrupted/motion_blur_{LEVEL}',
                     #f'cifar10_corrupted/pixelate_{LEVEL}',
                     #f'cifar10_corrupted/shot_noise_{LEVEL}',
                     #f'cifar10_corrupted/snow_{LEVEL}',
                     #f'cifar10_corrupted/zoom_blur_{LEVEL}']
    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # gin config files and bindings
    config_names = ['configMetaTest.gin']
    bindings = []

    # inject config
    utils_params.inject_gin([run_paths['path_gin']] + config_names, path_model_id=path_model_id,
                            bindings=bindings)  # bindings=['train_and_eval.n_epochs = 3','train_and_eval.save_period = 1']

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_test'], logging.INFO)

    # run testing over all datasets
    results = {}
    for dataset in test_datasets:
        # start testing for one dataset
        result = run_test(dataset=dataset, run_paths=run_paths)
        results[dataset] = result
    # restructure results for saving
    result_all = {}
    num_inner_steps = len(results[dataset]) - 1

    result_dicts = []
    for k in range(num_inner_steps + 1):
        # one file for initial and one for each inner step
        result_ = dict()
        for dataset_, values_ in results.items():
            result_[dataset_] = values_[k]
        result_dicts.append(result_)

    # Convert all results to json and print all results
    for k, res_ in enumerate(result_dicts):
        utils_misc.save_result_json(os.path.join(run_paths['path_model_id'], f'test_results_{k}.json'), res_)

    # Print results as table
    t = PrettyTable(['Dataset', 'Before', 'After'])
    acc_before_mean = 0.0
    acc_after_mean = 0.0
    for key, value in results.items():
        acc_0 = float(value[0]['accuracy']) * 100
        acc_last = float(value[-1]['accuracy']) * 100
        # add to overall
        acc_before_mean += acc_0
        acc_after_mean += acc_last
        t.add_row([key, acc_0, acc_last])

    # Calc avg
    acc_before_mean /= len(results)
    acc_after_mean /= len(results)
    # print avg
    t.add_row(['Avg.', acc_before_mean, acc_after_mean])

    # Nice format
    # print(t)
    list_of_table_lines = t.get_string().split('\n')
    # Use the first line (+---+-- ...) as horizontal rule to insert later
    horizontal_line = list_of_table_lines[0]
    # Print the table
    # Treat the last n lines as "result lines" that are seperated from the
    # rest of the table by the horizontal line
    result_lines = 1
    print("\n".join(list_of_table_lines[:-(result_lines + 1)]))
    print(horizontal_line)
    print("\n".join(list_of_table_lines[-(result_lines + 1):]))
