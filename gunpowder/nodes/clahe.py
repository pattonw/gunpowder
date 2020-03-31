import numpy as np

from gunpowder.batch_request import BatchRequest

from .batch_filter import BatchFilter


class CLAHE(BatchFilter):
    """Node utilizing scikit-images CLAHE implementation:
    https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist

    Args:

        arrays (List, :class:`ArrayKey`):

            The arrays to modify.

        kernel_size (int or array_like :int:):

            See scikit documentation

        clip_limit (float):

            See scikit documentation

        nbins (int):

            See scikit documentation
    """

    def __init__(self, arrays, kernel_size, clip_limit=0.01, nbins=256):
        self.arrays = arrays
        self.kernel_size = np.array(kernel_size)
        self.clip_limit = clip_limit
        self.nbins = nbins

    def setup(self):
        self.enable_autoskip()
        for key in self.arrays():
            self.updates(key, self.spec[key])

    def prepare(self, request):
        deps = BatchRequest()
        growth = self.kernel_size // 2
        for key in self.arrays:
            spec = request[key].copy()
            spec.roi = spec.roi.grow(growth, growth)
            deps[key] = spec
        return deps

    def process(self, batch, request):

        for key, array in batch.items():
            data = array.data
            shape = data.shape
            data_dims = len(shape)
            kernel_dims = len(self.kernel)
            channels_shape = shape[: data_dims - kernel_dims]
            spatial_shape = shape[data_dims - kernel_dims :]

            for channel_index in itertools.product([range(d) for d in channels_shape]):
                data[channel_index] = scikit
