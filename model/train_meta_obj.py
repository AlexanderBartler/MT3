import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import gin
from time import time
import sys
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from model import metrics
from utils import utils_params


@gin.configurable(denylist=['target_model', 'run_paths', 'ds_train', 'inner_repetition', 'ds_train_info', 'ds_val'])
class MT3Trainer:
    """
    Class for MTTT Trainer
    """

    def __init__(self,
                 target_model,
                 ds_train,
                 ds_train_info,
                 ds_val,
                 run_paths,
                 inner_repetition,
                 meta_epochs,
                 meta_lr,
                 beta_byol,
                 num_inner_steps,
                 inner_lr,
                 use_lr_drop,
                 lr_drop_boundaries,
                 lr_factors,
                 use_inner_clipping,
                 use_outer_clipping,
                 clipping_norm,
                 debug=True,
                 keep_ckp=2,
                 save_period=5):
        """
        Init meta traininer
        """

        # All parameter
        self.run_paths = run_paths
        self.meta_epochs = meta_epochs
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.inner_lr = inner_lr
        self.save_period = save_period
        self.inner_repetition = inner_repetition
        self.lr_drop_boundaries = lr_drop_boundaries
        self.lr_factors = lr_factors
        self.use_lr_drop = use_lr_drop
        self.use_inner_clipping = use_inner_clipping
        self.use_outer_clipping = use_outer_clipping
        self.clipping_norm = clipping_norm
        self.beta_byol = beta_byol
        self.debug = debug
        self.keep_ckp = keep_ckp

        # datasets
        self.ds_train = ds_train
        self.ds_train_info = ds_train_info
        self.ds_val = ds_val

        # get shapes, batch sizes, steps per epoch
        self.meta_batch_size = ds_train._flat_shapes[0][0]
        self.inner_batch_size = ds_train._flat_shapes[0][1]
        self.input_shape = ds_train._flat_shapes[0][2:].as_list()
        self.num_classes = ds_train._flat_shapes[3][-1]
        if self.inner_repetition and ds_train_info.name == 'cifar10':
            self.steps_per_epoch = round(50000 / self.meta_batch_size)
        elif not self.inner_repetition and ds_train_info.name == 'cifar10':
            self.steps_per_epoch = round(50000 / (self.meta_batch_size * self.inner_batch_size))

        # init target model and call one time for correct init
        logging.info("Building models...")
        self.target_model = target_model(n_classes=self.num_classes)
        # self.target_model.build(input_shape=tuple([None] + self.input_shape))
        self.target_model(tf.zeros(shape=tuple([1] + self.input_shape)))
        self.target_model(tf.zeros(shape=tuple([1] + self.input_shape)), use_predictor=True)

        # init one instance for each inner step (and step 0)
        self.updated_models = list()
        for _ in range(self.num_inner_steps + 1):
            updated_model = target_model(n_classes=self.num_classes)
            # updated_model.build(input_shape=tuple([None] + self.input_shape))
            updated_model(tf.zeros(shape=tuple([1] + self.input_shape)))
            updated_model(tf.zeros(shape=tuple([1] + self.input_shape)), use_predictor=True)
            self.updated_models.append(updated_model)

        # define optimizer
        logging.info("Setup optimizer...")
        if self.use_lr_drop:
            boundaries = [k * self.steps_per_epoch for k in self.lr_drop_boundaries]
            lr_values = [k * self.meta_lr for k in self.lr_factors]
            learning_rate_schedule = ks.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries,
                                                                                    values=lr_values)
            self.meta_optimizer = ks.optimizers.SGD(learning_rate=learning_rate_schedule, momentum=0.9)

        else:
            self.meta_optimizer = ks.optimizers.SGD(learning_rate=self.meta_lr, momentum=0.9)

        # Checkpoint
        self.target_ckpt = tf.train.Checkpoint(model=self.target_model, optimizer=self.meta_optimizer)
        self.target_ckpt_manager = tf.train.CheckpointManager(self.target_ckpt, directory=run_paths['path_ckpts_train'],
                                                              max_to_keep=self.keep_ckp)

        # Logging tb
        # generate summary writer
        self.writer_train = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_train']))
        self.writer_eval = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_eval']))
        logging.info(f"saving train log to {os.path.dirname(run_paths['path_logs_train'])}")

        # metrics and losses
        self.ce_loss_obj = ks.losses.CategoricalCrossentropy()
        self.metric_ce_loss = tf.keras.metrics.Mean()
        self.metric_byol_loss = tf.keras.metrics.Mean()
        self.metric_loss = tf.keras.metrics.Mean()
        self.metrics_train = metrics.prep_metrics_meta()
        self.metrics_eval = metrics.prep_metrics()

    def train(self):
        """
        Training method
        :return:
        """

        # checkpoint and maybe restore model
        self.target_ckpt.restore(self.target_ckpt_manager.latest_checkpoint)
        if self.target_ckpt_manager.latest_checkpoint:
            logging.info(f"Restored from {self.target_ckpt_manager.latest_checkpoint}.")
            epoch_start = int(os.path.basename(self.target_ckpt_manager.latest_checkpoint).split('-')[1])
        else:
            logging.info("Initializing from scratch.")
            epoch_start = 0

        # use tf variable for epoch passing - so no new trace is triggered
        epoch_tf = tf.Variable(1, dtype=tf.int32)

        # global time counter for eta estimation
        total_time = 0.0

        for epoch in range(epoch_start, int(self.meta_epochs)):
            # Start epoch timer
            start = time()
            eta = (self.meta_epochs - epoch) * (total_time / (epoch + 1e-12)) / 60
            # assign tf variable, so graph building doesn't get triggered
            epoch_tf.assign(epoch)
            # Log start of epoch and ETA
            if self.use_lr_drop:
                logging.info(
                    f"Epoch {epoch + 1}/{self.meta_epochs}: starting training, LR:  {self.meta_optimizer.learning_rate(self.meta_optimizer.iterations.numpy()).numpy():.5f},  ETA: {eta:.1f} minutes.")
            else:
                logging.info(f"Epoch {epoch + 1}/{self.meta_epochs}: starting training,  ETA: {eta:.1f} minutes.")

            # Start iteration over meta batches
            step_cnt = 1
            for images_aug_1, images_aug_2, images, labels in self.ds_train:
                # Update one (meta) step
                start_step = time()
                self.meta_train_step(images_aug_1, images_aug_2, images, labels)
                if self.debug:
                    logging.info(
                        f"Step {step_cnt} finished in: {time() - start_step}s, ce_loss: {self.metric_ce_loss.result()}, byol_loss: {self.metric_byol_loss.result()}")
                step_cnt += 1

            # Eval (target) model
            for images, labels in self.ds_val:
                self.eval_step(images, labels)

            # maybe saving checkpoint
            if (epoch % self.save_period == 0) | (epoch + 1 == self.meta_epochs):
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                self.target_ckpt_manager.save(checkpoint_number=epoch)

            # get metrics and losses
            ce_loss = self.metric_ce_loss.result()
            byol_loss = self.metric_byol_loss.result()
            loss = self.metric_loss.result()
            metrics_res_train = metrics.result_meta(self.metrics_train, as_numpy=True)
            metrics_res_val = metrics.result(self.metrics_eval, as_numpy=True)

            # logging of metrics
            logging.info(
                f'Epoch {epoch + 1}/{self.meta_epochs}: loss: {loss}, ce_loss: {ce_loss}, byol_loss: {byol_loss}, metrics_train: {metrics_res_train}, metrics_eval: {metrics_res_val}')

            # Saving results into tensorboard
            with self.writer_train.as_default():
                tf.summary.scalar('loss', loss, step=epoch)
                tf.summary.scalar('ce_loss', ce_loss, step=epoch)
                tf.summary.scalar('byol_loss', byol_loss, step=epoch)
                [tf.summary.scalar(k, v, step=epoch) for (k, v) in metrics_res_train.items()]

            with self.writer_eval.as_default():
                [tf.summary.scalar(k, v, step=epoch) for (k, v) in metrics_res_val.items()]

            # reset metrics
            self.metric_ce_loss.reset_states()
            self.metric_byol_loss.reset_states()
            self.metric_loss.reset_states()
            metrics.reset_states(self.metrics_train)
            metrics.reset_states(self.metrics_eval)

            # save gin config and summarize model
            if epoch <= 0:
                gin_string = gin.operative_config_str()
                logging.info(f'Fetched config parameters: {gin_string}.')
                utils_params.save_gin(self.run_paths['path_gin'], gin_string)
                self.target_model.summary()

            # estimate epoch time
            total_time += time() - start

        return metrics_res_val

    @tf.function
    def meta_train_step(self, images_aug_1, images_aug_2, images, labels):
        """
        Performs one meta train step which updates self.target_model
        :param images_aug_1:
        :param images_aug_2:
        :param images:
        :param labels:
        :return:
        """
        with tf.GradientTape(persistent=False) as outer_tape:
            # final loss: "test loss" of meta training
            # test_predictions: predictions of updated (inner loop already performed) models
            # meta_predictions: predictions of target model before updating
            final_loss, final_ce_loss, final_byol_loss, test_predictions, meta_predictions = self.parallel_mapper(
                images_aug_1, images_aug_2, images, labels)
        # Calc gradients
        outer_gradients = outer_tape.gradient(final_loss, self.target_model.trainable_variables)

        # maybe clip gradients by norm
        if self.use_outer_clipping:
            outer_gradients = [tf.clip_by_norm(g, self.clipping_norm) if g is not None else g for g in outer_gradients]

        # Apply gradients
        self.meta_optimizer.apply_gradients(zip(outer_gradients, self.target_model.trainable_variables))

        # Update metrics
        self.metric_loss(final_loss)
        self.metric_ce_loss(final_ce_loss)
        self.metric_byol_loss(final_byol_loss)
        metrics.update_state_meta(self.metrics_train, labels, test_predictions, meta_predictions)

    def parallel_mapper(self, images_aug_1, images_aug_2, images, labels):
        """
        Apply the inner batch optimization via parallel map for each element (task/domain) in the meta batch
        :param images_aug_1:
        :param images_aug_2:
        :param images:
        :param labels:
        :return:
        """
        final_losses, final_ce_loss, final_byol_loss, final_predictions, final_meta_predictions = tf.map_fn(
            self.get_losses_of_inner_batch,
            elems=(
                images_aug_1,
                images_aug_2,
                images,
                labels
            ),
            fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
            parallel_iterations=self.meta_batch_size if self.meta_batch_size <= 4 else 4,
        )
        final_loss = tf.reduce_mean(final_losses)
        final_ce_loss = tf.reduce_mean(final_ce_loss)
        final_byol_loss = tf.reduce_mean(final_byol_loss)

        return final_loss, final_ce_loss, final_byol_loss, final_predictions, final_meta_predictions

    def sequential_mapper(self, images_aug_1, images_aug_2, images, labels):
        """
        Apply the inner batch optimization via parallel map for each element (task/domain) in the meta batch
        :param images_aug_1:
        :param images_aug_2:
        :param images:
        :param labels:
        :return:
        """
        final_losses = []
        final_predictions = []
        final_meta_predictions = []
        final_ce_loss = []
        final_byol_loss = []
        for k in range(self.meta_batch_size):
            losses, ce_loss, byol_loss, predictions, meta_predictions = self.get_losses_of_inner_batch(
                (images_aug_1[k, :, :, :, :], images_aug_2[k, :, :, :, :], images[k, :, :, :, :], labels[k, :, :]))
            final_losses.append(losses)
            final_ce_loss.append(ce_loss)
            final_byol_loss.append(byol_loss)
            final_predictions.append(predictions)
            final_meta_predictions.append(meta_predictions)

        # back to tensors
        final_losses = tf.stack(final_losses)
        final_ce_loss = tf.stack(final_ce_loss)
        final_byol_loss = tf.stack(final_byol_loss)
        final_predictions = tf.stack(final_predictions)
        final_meta_predictions = tf.stack(final_meta_predictions)

        # calc means for losses
        final_loss = tf.reduce_mean(final_losses)
        final_ce_loss = tf.reduce_mean(final_ce_loss)
        final_byol_loss = tf.reduce_mean(final_byol_loss)

        return final_loss, final_ce_loss, final_byol_loss, final_predictions, final_meta_predictions

    def get_losses_of_inner_batch(self, inputs):
        """
        Updated model (inner loop) and get "test" loss for optimization
        :param images_aug_1:
        :param images_aug_2:
        :param images:
        :param labels:
        :return:
        """
        # inputs
        images_aug_1, images_aug_2, images, labels = inputs
        # Apply inner loop
        updated_model = self.inner_train_looop(images_aug_1, images_aug_2)

        if self.inner_repetition:
            # if inner repetition, only evaluate on one train test image
            # get prediction of meta model (to calc the pre_update acc
            _, _, meta_prediction = self.target_model(images[:1, :, :, :])
            # Get new test predictions of updated model
            _, _, test_prediction = updated_model(images[:1, :, :, :])
            # Calc test loss
            ce_loss = self.ce_loss_obj(labels[:1, :], test_prediction)
        else:
            _, _, test_prediction = updated_model(images)
            _, _, meta_prediction = self.target_model(images)

            # Calc test loss
            ce_loss = self.ce_loss_obj(labels, test_prediction)

        # Get final byol loss
        # Get targets (not backproped for optimization)
        _, target_aug1, _ = updated_model(images_aug_1, use_predictor=False, training=True)  # TODO: use target model?
        _, target_aug2, _ = updated_model(images_aug_2, use_predictor=False, training=True)
        _, online_aug1, _ = updated_model(images_aug_1, use_predictor=True, training=True)
        _, online_aug2, _ = updated_model(images_aug_2, use_predictor=True, training=True)

        # Calculate byol loss
        byol_loss = tf.reduce_mean(
            self.byol_loss_fn(online_aug1, tf.stop_gradient(target_aug2)) + self.byol_loss_fn(online_aug2,
                                                                                              tf.stop_gradient(
                                                                                                  target_aug1)))

        # calc final test loss
        test_loss = ce_loss + self.beta_byol * byol_loss

        return test_loss, ce_loss, byol_loss, test_prediction, meta_prediction

    def inner_train_looop(self, images_aug_1, images_aug_2):
        """
        Inner training loop, here unsupervised with byol like loss
        :param images_aug_1:
        :param images_aug_2:
        :return:
        """

        # Copy target model (s.t. this is differentiable)
        # TODO: Maybe faster with changing update meta model
        gradients = list()
        for variable in self.target_model.trainable_variables:
            gradients.append(tf.zeros_like(variable))
        self.update_meta_model(self.updated_models[0], self.target_model, gradients)

        for k in range(1, self.num_inner_steps + 1):
            # Get targets
            _, tar1, _ = self.updated_models[k - 1](images_aug_1, use_predictor=False, training=True)
            _, tar2, _ = self.updated_models[k - 1](images_aug_2, use_predictor=False, training=True)

            # Perform inner optimization
            with tf.GradientTape(persistent=False) as train_tape:
                train_tape.watch(self.updated_models[k - 1].meta_trainable_variables)
                _, prediction1, _ = self.updated_models[k - 1](images_aug_1, use_predictor=True, training=True)
                _, prediction2, _ = self.updated_models[k - 1](images_aug_2, use_predictor=True, training=True)
                # Calc byol loss
                loss1 = self.byol_loss_fn(prediction1, tf.stop_gradient(tar2))
                loss2 = self.byol_loss_fn(prediction2, tf.stop_gradient(tar1))
                loss = tf.reduce_mean(loss1 + loss2)
            gradients = train_tape.gradient(loss, self.updated_models[k - 1].meta_trainable_variables)

            # Maybe clip gradients by norm
            if self.debug:
                norms = tf.stack([tf.norm(g) for g in gradients if g is not None])
                mean_norm = tf.reduce_mean(norms)
                std_norm = tf.math.reduce_std(norms)
                max_norm = tf.reduce_max(norms)
                # tf.print("Gradient Norms (mean, std (max)):", mean_norm, "+-", std_norm, " (", max_norm, ")",
                #         output_stream=sys.stdout)

            if self.use_inner_clipping:
                gradients = [tf.clip_by_norm(g, self.clipping_norm) if g is not None else g for g in gradients]

            # Gradient step for updated_models[k]
            self.update_meta_model(self.updated_models[k], self.updated_models[k - 1], gradients)

        return self.updated_models[-1]

    def update_meta_model(self, updated_model, model, gradients):
        k = 0
        variables = list()
        model_layers = list(self.flatten(model.layers))
        updated_model_layers = list(self.flatten(updated_model.layers))

        lr = self.inner_lr

        gradients = [tf.zeros(1) if v is None else v for v in gradients]
        for i in range(len(model_layers)):
            if isinstance(model_layers[i], tf.keras.layers.Conv2D):
                updated_model_layers[i].kernel = model_layers[i].kernel - lr * gradients[k]
                k += 1
                variables.append(updated_model_layers[i].kernel)
                if not updated_model_layers[i].bias is None:
                    updated_model_layers[i].bias = model_layers[i].bias - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].bias)
            elif isinstance(model_layers[i], tf.keras.layers.Dense):
                updated_model_layers[i].kernel = model_layers[i].kernel - lr * gradients[k]
                k += 1
                variables.append(updated_model_layers[i].kernel)
                if not updated_model_layers[i].bias is None:
                    updated_model_layers[i].bias = model_layers[i].bias - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].bias)

            elif isinstance(model_layers[i], tf.keras.layers.BatchNormalization):
                if hasattr(model_layers[i], 'moving_mean') and model_layers[i].moving_mean is not None:
                    updated_model_layers[i].moving_mean.assign(model_layers[i].moving_mean)
                if hasattr(model_layers[i], 'moving_variance') and model_layers[i].moving_variance is not None:
                    updated_model_layers[i].moving_variance.assign(model_layers[i].moving_variance)
                if hasattr(model_layers[i], 'gamma') and model_layers[i].gamma is not None:
                    updated_model_layers[i].gamma = model_layers[i].gamma - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].gamma)
                if hasattr(model_layers[i], 'beta') and model_layers[i].beta is not None:
                    updated_model_layers[i].beta = \
                        model_layers[i].beta - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].beta)

            elif isinstance(model_layers[i], tf.keras.layers.LayerNormalization):
                if hasattr(model_layers[i], 'gamma') and model_layers[i].gamma is not None:
                    updated_model_layers[i].gamma = model_layers[i].gamma - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].gamma)
                if hasattr(model_layers[i], 'beta') and model_layers[i].beta is not None:
                    updated_model_layers[i].beta = \
                        model_layers[i].beta - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].beta)
            elif isinstance(model_layers[i], tfa.layers.GroupNormalization):
                if hasattr(model_layers[i], 'gamma') and model_layers[i].gamma is not None:
                    updated_model_layers[i].gamma = model_layers[i].gamma - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].gamma)
                if hasattr(model_layers[i], 'beta') and model_layers[i].beta is not None:
                    updated_model_layers[i].beta = \
                        model_layers[i].beta - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].beta)
        setattr(updated_model, 'meta_trainable_variables', variables)

    def byol_loss_fn(self, x, y):
        x = tf.math.l2_normalize(x, axis=-1)
        y = tf.math.l2_normalize(y, axis=-1)
        return 2 - 2 * tf.math.reduce_sum(x * y, axis=-1)

    def flatten(self, l):
        for el in l:
            if hasattr(el, '_layers') and len(el._layers) > 0:
                yield from self.flatten(el._layers)
            else:
                yield el

    @tf.function
    def eval_step(self, images, labels):
        _, _, predictions = self.target_model(images)
        metrics.update_state(self.metrics_eval, labels, predictions)
        return 0
