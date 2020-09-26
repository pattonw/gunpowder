import numpy as np
import skimage

from gunpowder.batch_request import BatchRequest

from .batch_filter import BatchFilter

import random


class NoiseAugment(BatchFilter):
    '''Add random noise to an array. Uses the scikit-image function skimage.util.random_noise.
    See scikit-image documentation for more information on arguments and additional kwargs.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

        mode (``string``):

            Type of noise to add, see scikit-image documentation.

        seed (``int``):

            Optionally set a random seed, see scikit-image documentation.

        clip (``bool``):

            Whether to preserve the image range (either [-1, 1] or [0, 1]) by clipping values in the end, see
            scikit-image documentation

    Kwargs:

        Given a key-word argument pair such as {"mean": 0.5}, this kwarg will get passed
        directly to the scipy noise function.
        Given a key-word argument pair with a key prefixed by "range_", it is assumed
        that you want to sample this kwarg from a uniform range. i.e. given kwarg
        {"range_mean": (0, 1)}, first a value is sampled uniformly from range(0, 1)
        and then kwarg {"mean": value} will be passed to scipy noise function.
    '''

    def __init__(self, array, mode='gaussian', seed=None, clip=True, **kwargs):
        self.array = array
        self.mode = mode
        self.seed = seed
        self.clip = clip
        self.kwargs = kwargs

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, "Noise augmentation requires float types for the raw array (not " + str(raw.data.dtype) + "). Consider using Normalize before."
        assert raw.data.min() >= -1 and raw.data.max() <= 1, "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."

        kwargs = {}
        for kwarg, value_or_range in self.kwargs.items():
            if kwarg.startswith("range_"):
                kwarg = kwarg[6:]
                value = random.uniform(value_or_range[0], value_or_range[1])
            else:
                value = value_or_range
            kwargs[kwarg] = value

        raw.data = skimage.util.random_noise(
            raw.data,
            mode=self.mode,
            seed=self.seed,
            clip=self.clip,
            **kwargs).astype(raw.data.dtype)
