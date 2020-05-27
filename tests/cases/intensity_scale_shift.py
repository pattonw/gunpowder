from .provider_test import ProviderTest
from gunpowder import IntensityScaleShift, ArrayKeys, build, Normalize

import numpy as np


class TestIntensityScaleShift(ProviderTest):
    def test_shift(self):

        pipeline = (
            self.test_source
            + Normalize(ArrayKeys.RAW)
            + IntensityScaleShift(ArrayKeys.RAW, 0.5, 10)
        )

        with build(pipeline):
            batch = pipeline.request_batch(self.test_request)

            x = batch.arrays[ArrayKeys.RAW].data
            assert np.isclose(x.min(), 10)
            assert np.isclose(x.max(), 10)
