import gin, logging, os
import tensorflow as tf
from model import metrics
from time import time


@gin.configurable(denylist=['target_model', 'online_model', 'test_ds', 'test_ds_info', 'run_paths'])
def test_meta(target_model, online_model, test_ds, test_ds_info, run_paths, test_lr, num_test_steps):
    # Load ckpts and ckpt manager
    # manager automatically handles model reloading if directory contains ckpts
    # First call model, otherwise not all variables will be loaded
    target_model(tf.ones(shape=tuple([1] + test_ds._flat_shapes[0][1:].as_list())), use_predictor=True)
    online_model(tf.ones(shape=tuple([1] + test_ds._flat_shapes[0][1:].as_list())), use_predictor=True)
    ckpt = tf.train.Checkpoint(model=target_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'], max_to_keep=2)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    if ckpt_manager.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
        epoch_start = int(os.path.basename(ckpt_manager.latest_checkpoint).split('-')[1])
    else:
        assert ('No checkpoint for testing...')

    # Prepare Metrics
    metrics_test = [metrics.prep_metrics() for _ in range(num_test_steps + 1)]

    # Get optimizer (similiar to inner loop, so no momentum and so on)
    optimizer = tf.keras.optimizers.SGD(learning_rate=test_lr, momentum=0.0)

    # def byol loss
    def byol_loss_fn(x, y):
        x = tf.math.l2_normalize(x, axis=-1)
        y = tf.math.l2_normalize(y, axis=-1)
        return 2 - 2 * tf.math.reduce_sum(x * y, axis=-1)

    @tf.function
    def inner_loop(images_aug_1, images_aug_2, images, labels):
        # copy weights for each image
        # online_model.set_weights(target_model.get_weights()) # slow
        for k in range(0, len(online_model.weights)):
            if not online_model.weights[k].dtype == tf.bool:
                online_model.weights[k].assign(target_model.weights[k])
        # acc without inner update
        _, _, predictions = online_model(images[:1, :, :, :], training=False)  # only one image since repetition
        metrics.update_state(metrics_test[0], labels[:1, :], predictions)

        # inner update and acc
        for k in range(num_test_steps):
            # calc target
            # Get targets
            _, tar1, _ = target_model(images_aug_1, use_predictor=False, training=True)
            _, tar2, _ = target_model(images_aug_2, use_predictor=False, training=True)

            # Perform inner optimization
            with tf.GradientTape(persistent=False) as test_tape:
                _, prediction1, _ = online_model(images_aug_1, use_predictor=True, training=True)
                _, prediction2, _ = online_model(images_aug_2, use_predictor=True, training=True)
                # Calc byol loss
                loss1 = byol_loss_fn(prediction1, tf.stop_gradient(tar2))
                loss2 = byol_loss_fn(prediction2, tf.stop_gradient(tar1))
                loss = tf.reduce_mean(loss1 + loss2)
            gradients = test_tape.gradient(loss, online_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, online_model.trainable_variables))
            # get predictions for test acc
            _, _, predictions = online_model(images[:1, :, :, :], training=False)  # only one image since repetition
            metrics.update_state(metrics_test[k + 1], labels[:1, :], predictions)
        return 0

    k = 1
    for images_aug_1, images_aug_2, images, labels in test_ds:
        inner_loop(images_aug_1, images_aug_2, images, labels)
        k += 1
        if k==3:
            break

    # fetch & reset metrics
    metrics_res_test = [metrics.result(metrics_, as_numpy=True) for metrics_ in metrics_test]

    [metrics.reset_states(metrics_) for metrics_ in metrics_test]

    logging.info(f'Result: metrics_test: {metrics_res_test}.')

    return metrics_res_test
