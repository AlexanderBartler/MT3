import logging
from model.inputs.input_fn_byol import gen_pipeline_test_byol
from model import model_fn, test_baselineBYOL
from utils import utils_params, utils_misc
import os, argparse
from prettytable import PrettyTable


def run_test(dataset='', run_paths=''):
    """
    Setup complete environment including datasets, models and start testing
    """

    # Define input pipeline depending on the type of training
    logging.info('Setup input pipeline...')
    test_ds, test_info = gen_pipeline_test_byol(ds_name=dataset)

    # Define model
    logging.info("Setup model...")
    model = model_fn.gen_model(n_classes=test_info.features['label'].num_classes)

    # Train and eval
    logging.info('Start testing...')

    result = test_baselineBYOL.test_baseline(model, test_ds, test_info, run_paths)
    return result


if __name__ == '__main__':
    # Define model path
    parser = argparse.ArgumentParser(description='BYOL Test')
    parser.add_argument('--path', type=str, default='', required=False, help='Result path')
    args = parser.parse_args()

    if args.path:
        path_model_id = args.path
    else:
        path_model_id = ''  # define model for manual testing

    # Define test datasets
    LEVEL = 5
    test_datasets = [f'cifar10_corrupted/brightness_{LEVEL}',
                     f'cifar10_corrupted/contrast_{LEVEL}',
                     f'cifar10_corrupted/defocus_blur_{LEVEL}',
                     f'cifar10_corrupted/elastic_{LEVEL}',
                     f'cifar10_corrupted/fog_{LEVEL}',
                     f'cifar10_corrupted/frost_{LEVEL}',
                     f'cifar10_corrupted/frosted_glass_blur_{LEVEL}',
                     f'cifar10_corrupted/gaussian_noise_{LEVEL}',
                     f'cifar10_corrupted/impulse_noise_{LEVEL}',
                     f'cifar10_corrupted/jpeg_compression_{LEVEL}',
                     f'cifar10_corrupted/motion_blur_{LEVEL}',
                     f'cifar10_corrupted/pixelate_{LEVEL}',
                     f'cifar10_corrupted/shot_noise_{LEVEL}',
                     f'cifar10_corrupted/snow_{LEVEL}',
                     f'cifar10_corrupted/zoom_blur_{LEVEL}']

    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # gin config files and bindings
    config_names = []
    bindings = []

    # inject config
    utils_params.inject_gin([run_paths['path_gin']], path_model_id=path_model_id,
                            bindings=bindings)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_test'], logging.INFO)

    # run testing over all datasets
    results = {}
    for dataset in test_datasets:
        # start testing for one dataset
        result = run_test(dataset=dataset, run_paths=run_paths)
        results[dataset] = result

    # Convert all results to json and print all results
    utils_misc.save_result_json(os.path.join(run_paths['path_model_id'], 'test_results.json'), results)

    # Print results as table
    t = PrettyTable(['Dataset', 'Acc'], float_format='.2')
    acc_mean = 0.0
    for key, value in results.items():
        acc = float(value['accuracy']) * 100
        # add to overall
        acc_mean += acc
        t.add_row([key, acc])

    # Calc avg
    acc_mean /= len(results)
    # print avg
    t.add_row(['Avg.', acc_mean])

    # Nice format
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
