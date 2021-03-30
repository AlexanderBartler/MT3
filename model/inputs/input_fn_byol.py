import tensorflow as tf
import tensorflow_datasets as tfds
import gin
import model.inputs.cifar10_corrupted_train
from model.augementations.batch_augmentation import batch_augmentation
from model.augementations.auto_augment import distort_image_with_autoaugment
from model.augementations.simclr_augment import distort_simclr


@gin.configurable
def gen_pipeline_train_byol(ds_name='cifar10',
                            tfds_path='',
                            size_batch=128,
                            b_shuffle=True,
                            size_buffer_cpu=5,
                            shuffle_buffer_size=0,
                            dataset_cache=False,
                            augmentation_type='simclr',
                            batch_aug = False,
                            num_parallel_calls=-1):
    """
    Build input pipeline for baseline training
    :param ds_name: dataset name for tensorflow dataset
    :param tfds_path: path to tfds datasets
    :param size_batch: batch size for training
    :param b_shuffle: shuffle before each iteration
    :param size_buffer_cpu: number of batches to be prefetched
    :param shuffle_buffer_size: buffer size for shuffle
    :param dataset_cache: cache dataset in RAM
    :param augmentation_type: type of training augementation to be applied ('simple','batchaug')
    :param num_parallel_calls: how many parallel calls to be used (AUTOTUNE = -1)
    :return: dataset and dataset info
    """

    # Load and prepare tensorflow dataset
    data, info = tfds.load(name=ds_name,
                           data_dir=tfds_path,
                           split=tfds.Split.TRAIN,
                           shuffle_files=False,
                           with_info=True)

    def _map_data(*args):
        """
        Preprocessing for images (before data augmentation)
        :param args:
        :return:
        """
        image = args[0]['image']
        label = args[0]['label']
        label = tf.one_hot(label, info.features['label'].num_classes, dtype=tf.int32)
        return image, label

    def _simple_aug(image):
        image = tf.cast(image, tf.float32) / 255.
        # image = tf.image.random_flip_left_right(image)
        IMG_SIZE = image.shape[0]  # CIFAR10: 32
        # Add 4 pixels of padding
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4)
        # Random crop back to the original size
        image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
        return image

    def _map_augment_img(*inputs):
        """Data augmentation function to be applied"""
        image, label = inputs
        if augmentation_type == 'simclr':
            image1 = distort_simclr(image)
            image2 = distort_simclr(image)
            # image3 = distort_simclr(image)
            image3 = _simple_aug(image)
        elif augmentation_type == 'autoaug':
            image1 = distort_image_with_autoaugment(image, 'cifar10')
            image2 = distort_image_with_autoaugment(image, 'cifar10')
            image3 = _simple_aug(image)
            image1 = tf.cast(image1, tf.float32) / 255.0
            image2 = tf.cast(image2, tf.float32) / 255.0
        else:
            assert 'augmentation type not supported!'
        return image1, image2, image3, label

    def _map_augment_batch(*inputs):
        image1, image2, image3, label = inputs
        stddev = tf.random.uniform((1,), 0, 0.02, dtype=tf.float32)
        flip_ud = tf.random.uniform([], 0, 1)
        flip_rl = tf.random.uniform([], 0, 1)
        blur = tf.random.uniform([], 0, 1)
        brightness = tf.random.uniform((1,), -0.2, 0.2, dtype=tf.float32)
        image1 = batch_augmentation(image1, stddev, flip_ud, flip_rl, brightness, blur)
        image2 = batch_augmentation(image2, stddev, flip_ud, flip_rl, brightness, blur)
        image3 = batch_augmentation(image3, stddev, flip_ud, flip_rl, brightness, blur)
        return image1, image2, image3, label

    # Map data
    dataset = data.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)

    # Cache data
    if dataset_cache:
        dataset = dataset.cache()

    # Shuffle data
    if b_shuffle:
        if shuffle_buffer_size == 0:
            shuffle_buffer_size = info.splits['train'].num_examples
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    # Map Augmentation
    dataset = dataset.map(map_func=_map_augment_img, num_parallel_calls=num_parallel_calls)


    dataset = dataset.batch(batch_size=size_batch, drop_remainder=True)  #
    if batch_aug:
        # use batch aug
        dataset = dataset.map(_map_augment_batch, num_parallel_calls=num_parallel_calls)
    if size_buffer_cpu > 0:
        dataset = dataset.prefetch(buffer_size=size_buffer_cpu)

    return dataset, info


@gin.configurable
def gen_pipeline_eval_byol(ds_name='mnist',
                           tfds_path='~/tensorflow_datasets',
                           size_batch=100,
                           size_buffer_cpu=5,
                           dataset_cache=False,
                           num_parallel_calls=-1):
    """
    :param ds_name: dataset name for tensorflow datasets
    :param tfds_path: path of tfds
    :param size_batch: batch size
    :param size_buffer_cpu:number of batches to be prefetched
    :param dataset_cache: cache dataset in RAM
    :param num_parallel_calls: number of parallel calls
    :return: dataset and info
    """
    # Load and prepare tensorflow dataset
    data, info = tfds.load(name=ds_name,
                           data_dir=tfds_path,
                           split=tfds.Split.TEST,
                           shuffle_files=False,
                           with_info=True)

    @tf.function
    def _map_data(*args):
        image = args[0]['image']
        label = args[0]['label']

        # Cast image type and range to [0,1]
        image = tf.cast(image, tf.float32) / 255.

        label = tf.one_hot(label, info.features['label'].num_classes, dtype=tf.int32)
        return image, label

    # Map data
    dataset = data.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)

    # Cache data
    if dataset_cache:
        dataset = dataset.cache()

    # Batching
    dataset = dataset.batch(batch_size=size_batch,
                            drop_remainder=True)  # > 1.8.0: use drop_remainder=True
    if size_buffer_cpu > 0:
        dataset = dataset.prefetch(buffer_size=size_buffer_cpu)

    return dataset, info


@gin.configurable
def gen_pipeline_test_byol(ds_name='cifar10',
                           tfds_path='~/tensorflow_datasets',
                           size_batch=100,
                           size_buffer_cpu=5,
                           dataset_cache=False,
                           num_parallel_calls=-1):
    """
    :param ds_name: dataset name for tensorflow datasets
    :param tfds_path: path of tfds
    :param size_batch: batch size
    :param size_buffer_cpu:number of batches to be prefetched
    :param dataset_cache: cache dataset in RAM
    :param num_parallel_calls: number of parallel calls
    :return: dataset and info
    """
    # Load and prepare tensorflow dataset
    data, info = tfds.load(name=ds_name,
                           data_dir=tfds_path,
                           split=tfds.Split.TEST,
                           shuffle_files=False,
                           with_info=True)

    @tf.function
    def _map_data(*args):
        image = args[0]['image']
        label = args[0]['label']

        # Cast image type and range to [0,1]
        image = tf.cast(image, tf.float32) / 255.
        label = tf.one_hot(label, info.features['label'].num_classes, dtype=tf.int32)
        return image, label

    # Map data
    dataset = data.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)

    # Cache data
    if dataset_cache:
        dataset = dataset.cache()

    # Batching
    dataset = dataset.batch(batch_size=size_batch,
                            drop_remainder=True)  # > 1.8.0: use drop_remainder=True
    if size_buffer_cpu > 0:
        dataset = dataset.prefetch(buffer_size=size_buffer_cpu)

    return dataset, info
