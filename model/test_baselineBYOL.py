import tensorflow as tf
import gin
import logging
import os
from model import metrics


@gin.configurable(denylist=['model', 'test_ds', 'test_ds_info', 'run_paths'])
def test_baseline(model, test_ds, test_ds_info, run_paths):
    # Load ckpts and ckpt manager
    # manager automatically handles model reloading if directory contains ckpts
    # First build model, otherwise not all variables will be loaded
    model.build(input_shape=tuple([None] + test_ds._flat_shapes[0][1:].as_list()))
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'], max_to_keep=2)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    if ckpt_manager.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
        epoch_start = int(os.path.basename(ckpt_manager.latest_checkpoint).split('-')[1])
    else:
        assert ('No checkpoint for testing...')

    # Prepare Metrics
    metrics_test = metrics.prep_metrics()

    # Testing
    for images, labels in test_ds:
        eval_step(model, images, labels, metrics_test)

    # fetch & reset metrics
    metrics_res_test = metrics.result(metrics_test, as_numpy=True)

    metrics.reset_states(metrics_test)

    logging.info(f'Result: metrics_test: {metrics_res_test}.')

    return metrics_res_test


@tf.function
def eval_step(model, images, labels, metrics_used):
    # Note: wrap whole test step, otherwise you end up with a tape on the CPU.
    with tf.device('/gpu:*'):
        _, _, predictions = model(images, training=False)

    metrics.update_state(metrics_used, labels, predictions)

    return
