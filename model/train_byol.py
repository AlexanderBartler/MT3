# noinspection PyInterpreter
import tensorflow as tf
import gin
import logging
import tensorflow.keras as ks
import os
from model import metrics
from utils import utils_params
import sys
from time import time
import numpy as np


@gin.configurable(
    denylist=['online_model', 'target_model', 'train_ds', 'train_ds_info', 'eval_ds', 'test_ds', 'run_paths'])
def train_and_eval_byol(online_model,
                        target_model,
                        train_ds,
                        train_ds_info,
                        eval_ds,
                        test_ds,
                        run_paths,
                        n_epochs=200,
                        lr_base=0.1,
                        lr_momentum=0.9,
                        beta_base=0.996,
                        gamma_byol=1.0,
                        lr_drop_boundaries=[1, 80, 120],
                        lr_factors=[0.1, 1, 0.1, 0.01],
                        save_period=1):
    # generate summary writer
    writer_train = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_train']))
    writer_eval = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_eval']))
    writer_test = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_test']))
    logging.info(f"saving train log to {os.path.dirname(run_paths['path_logs_train'])}")

    # loss
    ce_loss_obj = ks.losses.CategoricalCrossentropy()

    # define optimizer with learning rate schedule
    steps_per_epoch = 50000 // train_ds._flat_shapes[0][0]
    max_num_steps = (train_ds_info._splits['train'].num_examples // train_ds._flat_shapes[0][0]) * n_epochs

    boundaries = [k * steps_per_epoch for k in lr_drop_boundaries]
    lr_values = [k * lr_base for k in lr_factors]
    learning_rate_schedule = ks.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries,
                                                                            values=lr_values)
    optimizer = ks.optimizers.SGD(learning_rate=learning_rate_schedule, momentum=lr_momentum)

    # define ckpts and ckpt manager
    # manager automatically handles model reloading if directory contains ckpts
    # First build model, otherwise not all variables will be loaded
    online_model.build(input_shape=tuple([None] + train_ds._flat_shapes[0][1:].as_list()))
    target_model.build(input_shape=tuple([None] + train_ds._flat_shapes[0][1:].as_list()))

    ckpt = tf.train.Checkpoint(model=online_model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'], max_to_keep=2)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    if ckpt_manager.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
        epoch_start = int(os.path.basename(ckpt_manager.latest_checkpoint).split('-')[1])
    else:
        logging.info("Initializing from scratch.")
        epoch_start = 0

    # metrics
    metric_loss = tf.keras.metrics.Mean()
    metric_ce_loss = tf.keras.metrics.Mean()
    metric_byol_loss = tf.keras.metrics.Mean()
    metrics_train = metrics.prep_metrics()
    metrics_eval = metrics.prep_metrics()
    metrics_test = metrics.prep_metrics()
    logging.info(f"Training from epoch {epoch_start + 1} to {n_epochs}.")

    # use tf variable for epoch passing - so no new trace is triggered
    # if using normal range (instead of tf.range) assign a epoch_tf tensor, otherwise function gets recreated every turn
    epoch_tf = tf.Variable(1, dtype=tf.int32)
    # define tf variable for beta (ema momentum update)
    beta_tf = tf.Variable(1, dtype=tf.float32)

    # Global training time in s
    total_time = 0.0
    # Note: using tf.range also seems to create a graph
    for epoch in range(epoch_start, int(n_epochs)):
        # Start epoch timer
        start = time()
        eta = (n_epochs - epoch) * (total_time / (epoch + 1e-12)) / 60
        # assign tf variable, so graph building doesn't get triggered
        epoch_tf.assign(epoch)

        # perform training for one epoch
        logging.info(
            f"Epoch {epoch + 1}/{n_epochs}: starting training, LR:  {optimizer.learning_rate(optimizer.iterations.numpy()).numpy():.5f},  ETA: {eta:.1f} minutes.")

        for image_aug1, image_aug2, image_sup, labels in train_ds:
            # Update momentum for ema (beta)
            beta = 1. - (1 - beta_base) * (np.cos(np.pi * optimizer.iterations.numpy() / max_num_steps) + 1) / 2
            beta_tf.assign(beta)  # assign to use in graph

            train_step(online_model,
                       target_model,
                       image_aug1,
                       image_aug2,
                       image_sup,
                       labels,
                       optimizer,
                       ce_loss_obj,
                       metric_loss,
                       metric_ce_loss,
                       metric_byol_loss,
                       metrics_train,
                       epoch_tf=epoch_tf,
                       beta_tf=beta_tf,
                       gamma_byol=gamma_byol,
                       b_verbose=False)

        # print model summary once - done after training on first epoch, so model is already built.
        if epoch <= 0:
            online_model.summary()

        # save train metrics
        loss_avg = metric_loss.result()
        loss_ce_avg = metric_ce_loss.result()
        loss_byol_avg = metric_byol_loss.result()
        metrics_res_train = metrics.result(metrics_train, as_numpy=True)

        with writer_train.as_default():
            tf.summary.scalar('loss_average', loss_avg, step=epoch)
            tf.summary.scalar('ce_loss_average', loss_ce_avg, step=epoch)
            tf.summary.scalar('byol_loss_average', loss_byol_avg, step=epoch)
            [tf.summary.scalar(k, v, step=epoch) for (k, v) in metrics_res_train.items()]

        # Reset metrics
        metric_loss.reset_states()
        metric_ce_loss.reset_states()
        metric_byol_loss.reset_states()
        metrics.reset_states(metrics_train)

        # Eval epoch
        for images, labels in eval_ds:
            eval_step(online_model, images, labels, metrics_eval)

        # fetch & reset metrics
        metrics_res_eval = metrics.result(metrics_eval, as_numpy=True)
        with writer_eval.as_default():
            [tf.summary.scalar(k, v, step=epoch) for (k, v) in metrics_res_eval.items()]

        metrics.reset_states(metrics_eval)

        # Test epoch
        for images, labels in test_ds:
            eval_step(online_model, images, labels, metrics_test)

        # fetch & reset metrics
        metrics_res_test = metrics.result(metrics_test, as_numpy=True)
        with writer_test.as_default():
            [tf.summary.scalar(k, v, step=epoch) for (k, v) in metrics_res_test.items()]

        metrics.reset_states(metrics_test)

        logging.info(
            f'Epoch {epoch + 1}/{n_epochs}: loss_average: {loss_avg}, metrics_train: {metrics_res_train}, metrics_eval: {metrics_res_eval}, metrics_test: {metrics_res_test}.')

        # saving checkpoints after first epoch, last epoch and save_period epochs
        if ((epoch + 1) % save_period == 0) | (epoch == n_epochs - 1):
            logging.info(f'Saving checkpoint to {run_paths["path_ckpts_train"]}.')
            ckpt_manager.save(checkpoint_number=epoch)

        # write config after everything has been established
        if epoch <= 0:
            gin_string = gin.operative_config_str()
            logging.info(f'Fetched config parameters: {gin_string}.')
            utils_params.save_gin(run_paths['path_gin'], gin_string)
        # Calc total run_time
        total_time += time() - start
    return metrics_res_eval


@tf.function
def train_step(online_model, target_model, image_aug1, image_aug2, image_sup, labels, optimizer, ce_loss_obj,
               metric_loss, metric_ce_loss, metric_byol_loss, metrics_used, epoch_tf, beta_tf, gamma_byol,
               b_verbose=False):
    logging.info(f'Trace indicator - train epoch - eager mode: {tf.executing_eagerly()}.')

    # Note: wrap whole training step, otherwise you end up with a tape on the CPU.
    with tf.device('/gpu:*'):
        # Get targets (not backproped for optimization)
        _, target_aug1, _ = target_model(image_aug1, use_predictor=False, training=True)
        _, target_aug2, _ = target_model(image_aug2, use_predictor=False, training=True)

        with tf.GradientTape() as tape:
            _, online_aug1, _ = online_model(image_aug1, use_predictor=True, training=True)
            _, online_aug2, _ = online_model(image_aug2, use_predictor=True, training=True)
            _, _, predictions = online_model(image_sup, use_predictor=True, training=True)

            # Calculate losses
            byol_loss = tf.reduce_mean(
                byol_loss_fn(online_aug1, tf.stop_gradient(target_aug2)) + byol_loss_fn(online_aug2,
                                                                                        tf.stop_gradient(target_aug1)))
            ce_loss = ce_loss_obj(labels, predictions)
            loss = ce_loss + gamma_byol * byol_loss

        gradients = tape.gradient(loss, online_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, online_model.trainable_variables))
        # Update target model weights with MAVG
        for k in range(0, len(target_model.weights)):
            if not target_model.weights[k].dtype == tf.bool:
                target_model.weights[k].assign(
                    target_model.weights[k] * beta_tf + online_model.weights[k] * (1 - beta_tf))
    # update metrics
    metric_loss.update_state(loss)
    metric_ce_loss.update_state(ce_loss)
    metric_byol_loss.update_state(byol_loss)

    metrics.update_state(metrics_used, labels, predictions)

    if b_verbose:
        tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - loss: ", loss,
                 " - ce_loss: ", ce_loss, " - byol_loss: ", byol_loss,
                 output_stream=sys.stdout)
    return


@tf.function
def eval_step(model, images, labels, metrics_used):
    # Note: wrap whole test step, otherwise you end up with a tape on the CPU.
    with tf.device('/gpu:*'):
        _, _, predictions = model(images, training=False)

    metrics.update_state(metrics_used, labels, predictions)

    return


# Helper functions

def byol_loss_fn(x, y):
    x = tf.math.l2_normalize(x, axis=-1)
    y = tf.math.l2_normalize(y, axis=-1)
    return 2 - 2 * tf.math.reduce_sum(x * y, axis=-1)
