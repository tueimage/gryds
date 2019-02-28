import theano.tensor as T

from lasagne.layers import Layer
from lasagne.utils import as_tuple


class Upscale3DLayer(Layer):
    """
    3D upscaling layer
    Performs 3D upscaling over the three trailing axes of a 5D input tensor.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a cubic scale factor region. If an iterable, it should have three
        elements.
    mode : {'repeat', 'dilate'}
        Upscaling mode: repeat element values or upscale leaving zeroes between
        upscaled elements. Default is 'repeat'.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, mode='repeat', **kwargs):
        super(Upscale3DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 3)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1 or \
           self.scale_factor[2] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

        if mode not in {'repeat', 'dilate'}:
            msg = "Mode must be either 'repeat' or 'dilate', not {0}"
            raise ValueError(msg.format(mode))
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        if output_shape[4] is not None:
            output_shape[4] *= self.scale_factor[2]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b, c = self.scale_factor
        upscaled = input
        if self.mode == 'repeat':
            if c > 1:
                upscaled = T.extra_ops.repeat(upscaled, c, 4)
            if b > 1:
                upscaled = T.extra_ops.repeat(upscaled, b, 3)
            if a > 1:
                upscaled = T.extra_ops.repeat(upscaled, a, 2)
        elif self.mode == 'dilate':
            if c > 1 or b > 1 or a > 1:
                output_shape = self.get_output_shape_for(input.shape)
                upscaled = T.zeros(shape=output_shape, dtype=input.dtype)
                upscaled = T.set_subtensor(
                    upscaled[:, :, ::a, ::b, ::c], input)
        return upscaled
