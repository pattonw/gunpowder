import numpy as np
from gunpowder.nodes.clahe import equalize_adapthist
from mclahe import mclahe


def test_adapthist_grayscale_Nd():
    """
    Test for n-dimensional consistency with float images
    Note: Currently if img.ndim == 3, img.shape[2] > 4 must hold for the image
    not to be interpreted as a color image by @adapt_rgb
    """
    # construct a 2d image and a stack of it
    np.random.seed(0)
    img2d = np.random.random((25, 25))
    img3d = np.array([img2d] * 25)
    adapted2d = equalize_adapthist(img2d, kernel_size=5, clip_limit=0.01)
    adapted3d = equalize_adapthist(img3d, kernel_size=5, clip_limit=0.01)

    # check that dimensions of input and output match
    assert img2d.shape == adapted2d.shape
    assert img3d.shape == adapted3d.shape

    # check that the result from the stack of 2d images is similar
    # to the underlying 2d image
    assert np.mean(np.abs(adapted2d - adapted3d[12])) < 0.06

    """
    img3d = np.random.random((25, 25, 25))
    adapted3d = equalize_adapthist(img3d, kernel_size=(5, 5))

    img4d = np.random.random((1, 1, 25, 25))
    img5d = img4d.reshape((1, 1, 1, 25, 25))
    adapted4d = equalize_adapthist(img4d, kernel_size=(1, 1, 5, 5))
    adapted5d = equalize_adapthist(img5d, kernel_size=(1, 1, 1, 5, 5))

    assert all(np.isclose(adapted4d, adapted5d[0]))
    """


def test_mclahe():
    # construct a 2d image and a stack of it
    np.random.seed(0)
    img2d = np.random.random((25, 25))
    img3d = np.array([img2d] * 25)
    adapted2d = mclahe(
        img2d,
        kernel_size=(5, 5),
        n_bins=128,
        clip_limit=0.01,
        adaptive_hist_range=False,
        use_gpu=False,
    )
    adapted3d = mclahe(
        img3d,
        kernel_size=(5, 5, 5),
        n_bins=128,
        clip_limit=0.01,
        adaptive_hist_range=False,
        use_gpu=False,
    )

    # check that dimensions of input and output match
    assert img2d.shape == adapted2d.shape
    assert img3d.shape == adapted3d.shape

    # check that the result from the stack of 2d images is similar
    # to the underlying 2d image
    assert np.mean(np.abs(adapted2d - adapted3d[12])) < 0.06

    img4d = np.random.random((1, 1, 25, 25))
    img5d = img4d.reshape((1, 1, 1, 25, 25))
    adapted4d = mclahe(
        img4d,
        kernel_size=(1, 1, 5, 5),
        n_bins=128,
        clip_limit=0.01,
        adaptive_hist_range=False,
        use_gpu=False,
    )
    adapted5d = mclahe(
        img5d,
        kernel_size=(1, 1, 1, 5, 5),
        n_bins=128,
        clip_limit=0.01,
        adaptive_hist_range=False,
        use_gpu=False,
    )

    assert all(np.isclose(adapted4d, adapted5d[0]).flatten())


def test_similarity():

    img2d = np.random.random((25, 25))
    img3d = np.array([img2d] * 25)
    clahed2d = mclahe(
        img2d,
        kernel_size=(5, 5),
        n_bins=256,
        clip_limit=0.01,
        adaptive_hist_range=False,
        use_gpu=False,
    )
    clahed3d = mclahe(
        img3d,
        kernel_size=(5, 5, 5),
        n_bins=256,
        clip_limit=0.01,
        adaptive_hist_range=False,
        use_gpu=False,
    )

    adapted2d = equalize_adapthist(img2d, kernel_size=5, clip_limit=0.01)
    adapted3d = equalize_adapthist(img3d, kernel_size=5, clip_limit=0.01)

    diff = abs(adapted2d - clahed2d)
    print(adapted2d.min(), adapted2d.max(), adapted2d.mean())
    print(clahed2d.min(), clahed2d.max(), clahed2d.mean())
    print(diff.min(), diff.max(), diff.mean())
    assert all(np.isclose(clahed2d, adapted2d).flatten())
    assert all(np.isclose(clahed3d, adapted3d).flatten())
