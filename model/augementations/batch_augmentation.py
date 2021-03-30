import gin
import tensorflow as tf
import tensorflow_addons as tfa


@gin.configurable(denylist=['image', 'stddev','flip_ud', 'flip_rl', 'brightness', 'blur' ])
def batch_augmentation(image, stddev, flip_ud, flip_rl, brightness, blur, flip_ud_rate=0.5, flip_rl_rate=0.5, blur_rate=0.2):
    # image must be scaled in [0, 1]

    # flip up/down
    if flip_ud < flip_ud_rate:
        image = tf.image.flip_up_down(image)

    # flip right/left
    if flip_rl < flip_rl_rate:
        image = tf.image.flip_left_right(image)

    # gaussian blur
    if blur < blur_rate:
        image = tfa.image.filters.gaussian_filter2d(image)

    # brightness
    image = tf.image.adjust_brightness(image, brightness)

    # gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)  # stddev=0.01
    noise_img = tf.add(image, noise)
    noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
    return noise_img