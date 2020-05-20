import pytest
import numpy as np

import deepCABAC


@pytest.mark.parametrize('arr', [np.random.rand(10).astype(np.float32),
                                 np.random.rand(8, 16, 3, 3).astype(np.float32),
                                 np.random.rand(8, 16).astype(np.float32)])
def test_deepcabac(arr):
    stepsize = 2 ** (-0.5 * 15)

    encoder = deepCABAC.Encoder()
    encoder.encodeWeightsRD(arr, 0.1, stepsize, 0.)
    a_enc = encoder.finish().tobytes()

    dec = deepCABAC.Decoder()
    dec.getStream(np.frombuffer(a_enc, dtype=np.uint8))
    arr_rec = dec.decodeWeights()
    dec.finish()

    arr_quant = (arr / stepsize).round() * stepsize
    np.testing.assert_equal(arr_quant, arr_rec)
