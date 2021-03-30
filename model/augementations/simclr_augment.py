"""inpspired by https://github.com/sthalles/SimCLR-tensorflow
   and https://github.com/Spijkervet/BYOL/blob/master/main.py"""
import tensorflow as tf


def resize_only(image, imsize):
    output = tf.image.resize(image, size=(imsize, imsize))
    return output


def distort_simclr(image):
    image = tf.cast(image, tf.float32) / 255.
    v1 = crop_and_resize(image)
    v1 = tf.image.random_flip_up_down(v1)
    v1 = color_distortion(v1)

    return v1


def crop_and_resize(image, imsize=32):
    size = tf.random.uniform(shape=[], minval=20, maxval=32, dtype=tf.dtypes.int32)
    image = tf.image.random_crop(image, size=(size, size, 3))
    output = tf.image.resize(image, size=(imsize, imsize))
    return output


def color_distortion(image, s=0.2):
    # image is a tensor with value range in [0, 1].
    # s is the strength of color distortion.

    def color_jitter(x):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8 * s)
        x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_hue(x, max_delta=0.2 * s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def color_drop(x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 3])
        return x

    image = tf.cond(
        tf.random.uniform(shape=(), minval=0, maxval=1) < 0.8,
        lambda selected_func=color_jitter: selected_func(image),
        lambda: image)

    image = tf.cond(
        tf.random.uniform(shape=(), minval=0, maxval=1) < 0.2,
        lambda selected_func=color_drop: selected_func(image),
        lambda: image)
    return image
