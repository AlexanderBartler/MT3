import gin
from model.architectures import BYOLResnetCifar10v2, ResnetCifar10v2

@gin.configurable()
def gen_model(Architecture, **kwargs):
    """Model function defining the graph operations.
    :param architecture: architecture module containing Architecture class (tf.keras.Model)
    :param kwargs: additional keywords passed directly to model
    """

    model = Architecture(**kwargs)

    return model
