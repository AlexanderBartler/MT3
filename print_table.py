import os
import json
import numpy as np
from prettytable import PrettyTable


def get_short_name(dataset, level=5):
    LEVEL = level
    test_datasets = {f'cifar10': 'orig',
                     f'cifar10_corrupted/brightness_{LEVEL}': 'brit',
                     f'cifar10_corrupted/contrast_{LEVEL}': 'contr',
                     f'cifar10_corrupted/defocus_blur_{LEVEL}': 'defoc',
                     f'cifar10_corrupted/elastic_{LEVEL}': 'elast',
                     f'cifar10_corrupted/fog_{LEVEL}': 'fog',
                     f'cifar10_corrupted/frost_{LEVEL}': 'frost',
                     f'cifar10_corrupted/frosted_glass_blur_{LEVEL}': 'glass',
                     f'cifar10_corrupted/gaussian_blur_{LEVEL}': 'gauss',
                     f'cifar10_corrupted/impulse_noise_{LEVEL}': 'impul',
                     f'cifar10_corrupted/jpeg_compression_{LEVEL}': 'jpeg',
                     f'cifar10_corrupted/motion_blur_{LEVEL}': 'motn',
                     f'cifar10_corrupted/pixelate_{LEVEL}': 'pixel',
                     f'cifar10_corrupted/saturate_{LEVEL}': 'satu',
                     f'cifar10_corrupted/shot_noise_{LEVEL}': 'shot',
                     f'cifar10_corrupted/snow_{LEVEL}': 'snow',
                     f'cifar10_corrupted/spatter_{LEVEL}': 'spat',
                     f'cifar10_corrupted/speckle_noise_{LEVEL}': 'speck',
                     f'cifar10_corrupted/zoom_blur_{LEVEL}': 'zoom'}
    return test_datasets[dataset]


def get_data_from_folder(folder, result_name='', skip=[]):
    # get all result files
    files = []
    if result_name:
        for subfolder in os.listdir(folder):
            subfolder_abs = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_abs):
                f_files = os.listdir(subfolder_abs)
                if result_name in f_files:
                    files.append(os.path.join(folder, subfolder, result_name))
    else:
        # direclty pass result file
        files = [folder]
    # get data from files
    data_list = []
    for file in files:
        with open(file) as json_file:
            data_list.append(json.load(json_file))

    # process data
    acc_dict = dict()
    all_accs = []
    for data in data_list:
        for dataset, result in data.items():
            acc = float(result['accuracy']) * 100
            if acc > 0.0 and dataset not in skip:
                # use only results which are used
                all_accs.append(acc)
                if acc_dict.__contains__(dataset):
                    acc_dict[dataset].append(acc)
                else:
                    acc_dict[dataset] = [acc]

    # sort dict to be consitend between methods
    items = acc_dict.items()
    acc_dict = dict(sorted(items))

    # calc mean, std, max, min for each dataset and mean over all used datasets
    data_res_dict = dict()
    all_means = []
    all_stds = []
    all_accs_per_run = []
    for dataset, accs in acc_dict.items():
        data_res_dict[dataset] = dict()
        # Metrics
        data_res_dict[dataset]['mean'] = np.mean(accs)
        all_means.append(np.mean(accs))
        data_res_dict[dataset]['std'] = np.std(accs)
        all_stds = np.std(accs)
        data_res_dict[dataset]['min'] = np.min(accs)
        data_res_dict[dataset]['max'] = np.max(accs)
        all_accs_per_run.append(accs)

    global_mean = np.mean(np.mean(all_accs_per_run, axis=0))
    global_std = np.std(np.mean(all_accs_per_run, axis=0))
    return data_res_dict, global_mean, global_std


def get_baseline(folder, skip=[]):
    data_res_dict, global_mean_means, global_std_means = get_data_from_folder(folder, result_name='test_results.json',
                                                                              skip=skip)
    return data_res_dict, global_mean_means, global_std_means


def get_jt(folder, ttt=False, skip=[]):
    if ttt == False:
        data_res_dict, global_mean_means, global_std_means = get_data_from_folder(folder,
                                                                                  result_name='test_results_0.json',
                                                                                  skip=skip)
    else:
        data_res_dict, global_mean_means, global_std_means = get_data_from_folder(folder,
                                                                                  result_name='test_results_1.json',
                                                                                  skip=skip)
    return data_res_dict, global_mean_means, global_std_means


def get_mt3(folder, ttt=False, skip=[]):
    if ttt == False:
        data_res_dict, global_mean_means, global_std_means = get_data_from_folder(folder,
                                                                                  result_name='test_results_0.json',
                                                                                  skip=skip)
    else:
        data_res_dict, global_mean_means, global_std_means = get_data_from_folder(folder,
                                                                                  result_name='test_results_1.json',
                                                                                  skip=skip)
    return data_res_dict, global_mean_means, global_std_means


def get_row_string(dataset, ttt_baseline_dict, ttt_jt_dict, ttt_ttt_dict, baseline_dict, jt_dict, jt_ttt_dict,
                   mt3_before_dict, mt3_after_dict):
    dataset_short = get_short_name(dataset)
    str = []

    str.append(f"{dataset_short}")

    str.append(f"{ttt_baseline_dict[dataset]['mean']:2.1f}")
    str.append(f"{ttt_jt_dict[dataset]['mean']:2.1f}")
    str.append(f"{ttt_ttt_dict[dataset]['mean']:2.1f}")
    str.append(f"{baseline_dict[dataset]['mean']:2.1f} +- {baseline_dict[dataset]['std']:2.2f}")
    str.append(f"{jt_dict[dataset]['mean']:2.1f} +- {jt_dict[dataset]['std']:2.2f}")
    str.append(f"{jt_ttt_dict[dataset]['mean']:2.1f} +- {jt_ttt_dict[dataset]['std']:2.2f}")
    str.append(f"{mt3_before_dict[dataset]['mean']:2.1f} +- {mt3_before_dict[dataset]['std']:2.2f}")
    str.append(f"{mt3_after_dict[dataset]['mean']:2.1f} +- {mt3_after_dict[dataset]['std']:2.2f}")
    return str


def get_row_string_btm(ttt_baseline_mean, ttt_jt_mean, ttt_ttt_mean, baseline_gmean, baseline_gstd, jt_gmean, jt_gstd,
                       jt_ttt_gmean, jt_ttt_gstd, mt3_before_gmean, mt3_before_gstd, mt3_after_gmean, mt3_after_gstd):
    str = []

    str.append(f"avg")

    str.append(f"{ttt_baseline_mean:2.1f}")
    str.append(f"{ttt_jt_mean:2.1f}")
    str.append(f"{ttt_ttt_mean:2.1f}")

    str.append(f"{baseline_gmean:2.1f} +- {baseline_gstd:2.2f}")
    str.append(f"{jt_gmean:2.1f} +- {jt_gstd:2.2f}")
    str.append(f"{jt_ttt_gmean:2.1f} +- {jt_ttt_gstd:2.2f}")
    str.append(f"{mt3_before_gmean:2.1f} +- {mt3_before_gstd:2.2f}")
    str.append(f"{mt3_after_gmean:2.1f} +- {mt3_after_gstd:2.2f}")
    return str


if __name__ == '__main__':

    # Plot results for ablation studies
    # configs
    # ignore additional datasets
    IGNORE = ['cifar10', 'cifar10_corrupted/spatter_5', 'cifar10_corrupted/speckle_noise_5',
              'cifar10_corrupted/saturate_5']
    # Folders
    folder_baseline = './final_results/baseline'
    folder_jt = './final_results/byol'  # used w/o tttt
    folder_mt3 = './final_results/mt3'  # used for before and after

    # files of sota method
    file_ttt_baseline = './final_results/ttt_paper/ttt_paper_baseline.json'
    file_ttt_jt = './final_results/ttt_paper/ttt_paper_jointtraining.json'
    file_ttt_ttt = './final_results/ttt_paper/ttt_paper_TTT.json'

    # get mean, std for all datasets
    baseline_dict, baseline_gmean, baseline_gstd = get_baseline(folder_baseline, skip=IGNORE)
    jt_dict, jt_gmean, jt_gstd = get_jt(folder_jt, ttt=False, skip=IGNORE)
    jt_ttt_dict, jt_ttt_gmean, jt_ttt_gstd = get_jt(folder_jt, ttt=True, skip=IGNORE)
    mt3_before_dict, mt3_before_gmean, mt3_before_gstd = get_mt3(folder_mt3, ttt=False, skip=IGNORE)
    mt3_after_dict, mt3_after_gmean, mt3_after_gstd = get_mt3(folder_mt3, ttt=True, skip=IGNORE)

    # get dict of sota methods
    ttt_baseline_dict, ttt_baseline_mean, _ = get_data_from_folder(file_ttt_baseline, skip=IGNORE)
    ttt_jt_dict, ttt_jt_mean, _ = get_data_from_folder(file_ttt_jt, skip=IGNORE)
    ttt_ttt_dict, ttt_ttt_mean, _ = get_data_from_folder(file_ttt_ttt, skip=IGNORE)

    # plot table for ablation study
    t = PrettyTable(
        ['', 'TTT Baseline [32]', ' TTT JT [32]', 'TTT [32]', 'Baseline (ours)', 'JT (ours)', 'TTT (ours)', 'MT (ours)',
         'MT3 (ours)'])

    for dataset, accs_before in baseline_dict.items():
        dataset_short = get_short_name(dataset)
        t.add_row(
            get_row_string(dataset, ttt_baseline_dict, ttt_jt_dict, ttt_ttt_dict, baseline_dict, jt_dict, jt_ttt_dict,
                           mt3_before_dict, mt3_after_dict))

    t.add_row(get_row_string_btm(ttt_baseline_mean, ttt_jt_mean, ttt_ttt_mean, baseline_gmean, baseline_gstd, jt_gmean,
                                 jt_gstd, jt_ttt_gmean, jt_ttt_gstd, mt3_before_gmean, mt3_before_gstd, mt3_after_gmean,
                                 mt3_after_gstd))

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
