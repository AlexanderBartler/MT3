# Meta Test-Time Training MT3
Implementation of the paper https://arxiv.org/abs/2103.16201
## Environment
Requirement Python3.x and Cuda 11.0 

`pip install -r requirements.txt`

Note: All datasets will be automatically downloaded to ~/tensorflow_datasets. Can be changed in the config/config*.gin files
## Run Baseline training (Baseline (ours))
Configuration of training parameters: config/configBaseline.gin

#### run with default parameters:

Run training: 
`python train_baseline.py --path <result path>`

Test baseline: 
`python test_baseline.py --path <result path>`

#### run train and test (results saved to ./results/baseline_*):
`./train_test_baseline.sh` 

Note: Results will be printed (as table) or stored in <path>/test_results.json

## Run BYOL baseline (JT (ours))
Configuration of training parameters: config/configBYOL.gin
#### run with default parameters:

Run training: 
`python train_byol.py --path <result path>`

Test baseline: 
`python test_byol.py --path <result path>`

#### run train and test (results saved to ./results/byol_*):
`./train_test_byol.sh` 

#### run test time adaption for byol (TTT(ours))
`python test_byol_ttt.py --path <result path>`

Note: Results will be printed (as table) or stored in <path>/test_results.json



## Run MT3 (MT(ours), MT3 (ours))
Configuration of training parameters: config/MT3.gin
#### run with default parameters:

Run training: 
`python train_meta.py --path <result path>`

Test baseline: 
`python test_meta.py --path <result path>`

#### run train and test (results saved to ./results/meta_*):
`./train_test_meta.sh` 

Note: Results will be printed (as table) or stored in <path>/test_results_*.json

## Reproduce test-time adaption of MT3  
Checkpoints for our method MT3 are given for test-time training

`python test_meta.py --path $(pwd)/checkpoints/mt3/run1/`

Note: Results maybe changing to the reported one since due tue test-time adaption random augmentations are applied (seed not fixed)

## Generate Table 2 results using jsons of all runs 
We provide the result files (json) of all runs

Print table related to table 2:

`python print_table.py`

|       | TTT Baseline [32] |  TTT JT [32] | TTT [32] | Baseline (ours) |  JT (ours)   |  TTT (ours)  |  MT (ours)   |  MT3 (ours)  |
|-------|-------------------|--------------|----------|-----------------|--------------|--------------|--------------|--------------|
|  brit |        86.5       |     87.4     |   87.8   |   86.7 +- 0.44  | 86.5 +- 0.13 | 86.6 +- 0.26 | 84.3 +- 1.15 | 86.2 +- 0.47 |
| contr |        75.0       |     74.7     |   76.1   |   54.0 +- 6.42  | 75.4 +- 2.02 | 75.1 +- 2.38 | 69.3 +- 2.63 | 77.6 +- 1.21 |
| defoc |        76.3       |     75.8     |   78.2   |   68.1 +- 2.34  | 84.7 +- 0.11 | 84.7 +- 0.09 | 82.7 +- 1.33 | 84.4 +- 0.44 |
| elast |        72.6       |     76.0     |   77.4   |   74.3 +- 0.27  | 74.6 +- 0.80 | 74.4 +- 1.19 | 74.2 +- 1.08 | 76.3 +- 1.18 |
|  fog  |        71.9       |     72.5     |   74.9   |   70.7 +- 0.98  | 70.3 +- 0.86 | 70.4 +- 0.67 | 72.0 +- 1.03 | 75.9 +- 1.26 |
| frost |        65.6       |     67.5     |   70.0   |   65.2 +- 0.93  | 79.8 +- 0.62 | 79.5 +- 0.73 | 76.6 +- 1.16 | 81.2 +- 0.20 |
| glass |        48.3       |     51.5     |   53.9   |   50.7 +- 2.96  | 62.8 +- 0.97 | 61.9 +- 1.10 | 62.8 +- 1.35 | 66.3 +- 1.24 |
| gauss |        49.5       |     50.6     |   54.4   |   49.9 +- 3.17  | 71.7 +- 1.13 | 70.4 +- 1.08 | 63.6 +- 1.17 | 69.9 +- 0.34 |
| impul |        43.9       |     46.6     |   50.0   |   43.4 +- 4.31  | 59.3 +- 3.04 | 58.5 +- 3.17 | 50.3 +- 1.68 | 58.2 +- 1.25 |
|  jpeg |        70.2       |     71.3     |   72.8   |   76.0 +- 0.86  | 78.6 +- 0.37 | 79.0 +- 0.44 | 75.2 +- 0.06 | 77.3 +- 0.26 |
|  motn |        75.7       |     75.2     |   77.0   |   71.6 +- 0.46  | 70.7 +- 0.45 | 69.8 +- 0.46 | 72.6 +- 3.17 | 77.2 +- 2.37 |
| pixel |        44.2       |     48.4     |   52.8   |   60.1 +- 2.73  | 65.0 +- 0.32 | 62.1 +- 0.44 | 67.8 +- 5.13 | 72.4 +- 2.29 |
|  shot |        52.8       |     54.7     |   58.2   |   52.3 +- 2.17  | 72.3 +- 1.36 | 71.0 +- 1.09 | 64.0 +- 1.24 | 70.5 +- 0.72 |
|  snow |        74.4       |     75.0     |   76.1   |   74.5 +- 0.46  | 77.2 +- 0.58 | 77.2 +- 0.55 | 77.1 +- 0.51 | 79.8 +- 0.63 |
|  zoom |        73.7       |     73.6     |   76.1   |   67.4 +- 1.70  | 81.6 +- 0.69 | 81.7 +- 0.66 | 78.7 +- 1.72 | 81.3 +- 0.58 |
|  avg  |        65.4       |     66.7     |   69.0   |   64.3 +- 0.42  | 74.0 +- 0.77 | 73.5 +- 0.80 | 71.4 +- 0.42 | 75.6 +- 0.30 |
