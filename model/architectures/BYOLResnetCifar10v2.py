import gin
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow_addons import layers as tfalayers


class MLP(models.Model):
    """
    MLP for predictor/projector
    """
    def __init__(self,hidden_size,projection_size,weight_decay, group_norm_groups, batch_norm_epsilon,batch_norm_center,batch_norm_scale,use_mlp_norm):
        super().__init__()
        self._kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        self.dense_in = tf.keras.layers.Dense(hidden_size, activation='linear', kernel_regularizer=self._kernel_regularizer)
        # maybe using groupnorm?
        self.use_mlp_norm = use_mlp_norm
        if self.use_mlp_norm:
            self._bn1 = tfalayers.GroupNormalization(groups=group_norm_groups,
                                                 axis=-1,
                                                 epsilon=batch_norm_epsilon,
                                                 center=batch_norm_center,
                                                 scale=batch_norm_scale)
        self.dense1_out = tf.keras.layers.Dense(projection_size, kernel_regularizer=self._kernel_regularizer)

    def call(self, x, training=False):
        output = self.dense_in(x)
        if self.use_mlp_norm:
            output = self._bn1(output)
        output = tf.nn.relu(output)
        output = self.dense1_out(output)
        return output


@gin.configurable('BYOLResnetCifar10v2', denylist=['n_classes'])
class Architecture(models.Model):
    """ResNet for CIFAR10 dataset."""
    "Adapted from: https://github.com/chao-ji/tf-resnet-cifar10/blob/master/v2/model.py"

    def __init__(self,
                 n_classes,
                 num_layers=20,
                 num_initial_filters=16,
                 shortcut_connection=True,
                 weight_decay=2e-4,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=1e-3,
                 batch_norm_center=True,
                 batch_norm_scale=True,
                 use_mlp_norm = True,
                 group_norm_groups=16,
                 proj_hidden_size=256,
                 proj_size=128):
        """Constructor.
        Args:
          num_layers: int scalar, num of layers.
          shortcut_connection: bool scalar, whether to add shortcut connection in
            each Resnet unit. If False, degenerates to a 'Plain network'.
          weight_decay: float scalar, weight for l2 regularization.
          batch_norm_momentum: float scalar, the moving avearge decay.
          batch_norm_epsilon: float scalar, small value to avoid divide by zero.
          batch_norm_center: bool scalar, whether to center in the batch norm.
          batch_norm_scale: bool scalar, whether to scale in the batch norm.
        """
        super().__init__()
        if num_layers not in (20, 26, 32, 44, 56, 110):
            raise ValueError('num_layers must be one of 20, 32, 44, 56 or 110.')

        self._num_layers = num_layers
        self._num_initial_filters = num_initial_filters
        self._shortcut_connection = shortcut_connection
        self._weight_decay = weight_decay
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._batch_norm_center = batch_norm_center
        self._batch_norm_scale = batch_norm_scale
        self._group_norm_groups = group_norm_groups

        self._num_units = (num_layers - 2) // 6

        self._kernel_regularizer = regularizers.l2(weight_decay)

        self._init_conv = layers.Conv2D(self._num_initial_filters, 3, 1, 'same', use_bias=False,
                                        kernel_regularizer=self._kernel_regularizer, name='init_conv')

        self._block1 = models.Sequential([ResNetUnit(
            self._num_initial_filters,
            1,
            shortcut_connection,
            True if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale,
            group_norm_groups,
            'res_net_unit_%d' % (i + 1)) for i in range(self._num_units)],
            name='block1')
        self._block2 = models.Sequential([ResNetUnit(
            self._num_initial_filters*2,
            2 if i == 0 else 1,
            shortcut_connection,
            False if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale,
            group_norm_groups,
            'res_net_unit_%d' % (i + 1)) for i in range(self._num_units)],
            name='block2')
        self._block3 = models.Sequential([ResNetUnit(
            self._num_initial_filters*4,
            2 if i == 0 else 1,
            shortcut_connection,
            False if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale,
            group_norm_groups,
            'res_net_unit_%d' % (i + 1)) for i in range(self._num_units)],
            name='block3')

        self._global_avg = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAvgPooling")

        self._last_bn = tfalayers.GroupNormalization(groups=group_norm_groups,
                                                     axis=-1,
                                                     epsilon=batch_norm_epsilon,
                                                     center=batch_norm_center,
                                                     scale=batch_norm_scale)

        self._pre_hidden = tf.keras.layers.Dense(proj_hidden_size, use_bias=False, activation='linear',kernel_regularizer=self._kernel_regularizer)
        #self._pre_hidden_norm = tfalayers.GroupNormalization(groups=group_norm_groups,
        #                                             axis=-1,
        #                                             epsilon=batch_norm_epsilon,
        #                                             center=batch_norm_center,
        #                                             scale=batch_norm_scale)

        self._mlp_dense_out = tf.keras.layers.Dense(n_classes, use_bias=True, name="Head_Dense", activation='softmax',
                                                    kernel_regularizer=self._kernel_regularizer)

        # BYOL Projector and Predictor
        self.projector = MLP(hidden_size=proj_hidden_size,projection_size=proj_size,weight_decay=weight_decay,group_norm_groups=group_norm_groups, batch_norm_epsilon=batch_norm_epsilon,batch_norm_center=batch_norm_center,batch_norm_scale=batch_norm_scale,use_mlp_norm=use_mlp_norm )

        # Only need a predictor for the online model
        self.predictor = MLP(hidden_size=proj_hidden_size,projection_size=proj_size,weight_decay=weight_decay,group_norm_groups=group_norm_groups, batch_norm_epsilon=batch_norm_epsilon,batch_norm_center=batch_norm_center,batch_norm_scale=batch_norm_scale,use_mlp_norm=use_mlp_norm)

    def call(self, inputs, training=False, use_predictor=False):
        """Execute the forward pass.
        Args:
          inputs: float tensor of shape [batch_size, 32, 32, 3], the preprocessed,
            data-augmented, and batched CIFAR10 images.
        Returns:
          logits: float tensor of shape [batch_size, 10], the unnormalized logits.
        """
        h = inputs
        h = self._init_conv(h)

        h = self._block1(h, training=training)
        h = self._block2(h, training=training)
        h = self._block3(h, training=training)
        h = tf.nn.relu(self._last_bn(h))
        h = self._global_avg(h)
        # Use one hidden layer as in simclr v2
        #h = self._pre_hidden_norm(h)
        h = tf.nn.relu(self._pre_hidden(h))

        # Use projector / predictor after global avg layer
        proj = self.projector(h)
        if use_predictor:
            proj = self.predictor(proj)
        # Classifier on top of feature space (here h), not on projection/prediction space
        g = self._mlp_dense_out(h, training=training)

        # return feature space h, projection/prediction proj and classiciton prediction g
        return h, proj, g

class ResNetUnit(layers.Layer):
    """A ResNet Unit contains two conv2d layers interleaved with Batch
    Normalization and ReLU.
    """

    def __init__(self,
                 depth,
                 stride,
                 shortcut_connection,
                 shortcut_from_preact,
                 weight_decay,
                 batch_norm_momentum,
                 batch_norm_epsilon,
                 batch_norm_center,
                 batch_norm_scale,
                 group_norm_groups,
                 name):
        """Constructor.
        Args:
          depth: int scalar, the depth of the two conv ops in each Resnet unit.
          stride: int scalar, the stride of the first conv op in each Resnet unit.
          shortcut_connection: bool scalar, whether to add shortcut connection in
            each Resnet unit. If False, degenerates to a 'Plain network'.
          shortcut_from_preact: bool scalar, whether the shortcut connection starts
            from the preactivation or the input feature map.
          weight_decay: float scalar, weight for l2 regularization.
          batch_norm_momentum: float scalar, the moving average decay.
          batch_norm_epsilon: float scalar, small value to avoid divide by zero.
          batch_norm_center: bool scalar, whether to center in the batch norm.
          batch_norm_scale: bool scalar, whether to scale in the batch norm.
        """
        super(ResNetUnit, self).__init__(name=name)
        self._depth = depth
        self._stride = stride
        self._shortcut_connection = shortcut_connection
        self._shortcut_from_preact = shortcut_from_preact
        self._weight_decay = weight_decay

        self._kernel_regularizer = regularizers.l2(weight_decay)

        self._bn1 = tfalayers.GroupNormalization(groups=group_norm_groups,
                                                 axis=-1,
                                                 epsilon=batch_norm_epsilon,
                                                 center=batch_norm_center,
                                                 scale=batch_norm_scale)
        self._conv1 = layers.Conv2D(depth,
                                    3,
                                    stride,
                                    'same',
                                    use_bias=True,
                                    kernel_regularizer=self._kernel_regularizer,
                                    name='conv1')

        self._bn2 = tfalayers.GroupNormalization(groups=group_norm_groups,
                                                 axis=-1,
                                                 epsilon=batch_norm_epsilon,
                                                 center=batch_norm_center,
                                                 scale=batch_norm_scale)
        self._conv2 = layers.Conv2D(depth,
                                    3,
                                    1,
                                    'same',
                                    use_bias=True,
                                    kernel_regularizer=self._kernel_regularizer,
                                    name='conv2')

    def call(self, inputs, training=False):
        """Execute the forward pass.
        Args:
          inputs: float tensor of shape [batch_size, height, width, depth], the
            input tensor.
        Returns:
          outouts: float tensor of shape [batch_size, out_height, out_width,
            out_depth], the output tensor.
        """
        depth_in = inputs.shape[3]  # depth_in = num_initial_filters
        depth = self._depth
        preact = tf.nn.relu(self._bn1(inputs))

        shortcut = preact if self._shortcut_from_preact else inputs

        if depth != depth_in:
            shortcut = tf.nn.avg_pool2d(
                shortcut, (2, 2), strides=(1, 2, 2, 1), padding='SAME')
            shortcut = tf.pad(
                shortcut, [[0, 0], [0, 0], [0, 0], [(depth - depth_in) // 2] * 2])

        residual = tf.nn.relu(self._bn2(self._conv1(preact)))
        residual = self._conv2(residual)

        outputs = residual + shortcut if self._shortcut_connection else residual

        return outputs
