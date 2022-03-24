from random import getrandbits, uniform
import numpy as np
import dataset_iterator.helpers as dih
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from ..model.utils import ensure_multiplicity

IMAGE_TYPE = ["TRANSMITTED LIGHT", "FLUORESCENCE", "PHASE CONTRAST"]

def get_normalization_center_scale_ranges(histogram, bins, center_centile_extent, scale_centile_range, verbose=False):
    assert dih is not None, "dataset_iterator package is required for this method"
    mode_value = dih.get_modal_value(histogram, bins)
    mode_centile = dih.get_percentile_from_value(histogram, bins, mode_value)
    print("model value={}, model centile={}".format(mode_value, mode_centile))
    assert mode_centile<scale_centile_range[0], "mode centile is {} and must be lower than lower bound of scale_centile_range={}".format(mode_centile, scale_centile_range)
    centiles = [max(0, mode_centile-center_centile_extent), min(100, mode_centile+center_centile_extent)]
    scale_centile_range = ensure_multiplicity(2, scale_centile_range)
    if isinstance(scale_centile_range, tuple):
        scale_centile_range = list(scale_centile_range)
    centiles = centiles + scale_centile_range
    values = dih.get_percentile(histogram, bins, centiles)
    mode_range = [values[0], values[1] ]
    scale_range = [values[2] - mode_value, values[3] - mode_value]
    if verbose:
        print("normalization_center_scale: modal value: {}, center_range: [{}; {}] scale_range: [{}; {}]".format(mode_value, mode_range[0], mode_range[1], scale_range[0], scale_range[1]))
    return mode_range, scale_range

def get_center_scale_range(dataset, raw_feature_name:str = "/raw", image_type:str=IMAGE_TYPE[0], tl_sd_factor:float=3., fluo_centile_range:list=[75, 99.9], fluo_centile_extent:float=5):
    """Computes a range for center and for scale factor for data augmentation.
    Image can then be normalized using a random center C in the center range and a random scaling factor in the scale range: I -> (I - C) / S

    Parameters
    ----------
    dataset : datasetIO/path(str) OR list/tuple of datasetIO/path(str)
    raw_feature_name : str
        name of the dataset
    image_type : str in ["TRANSMITTED LIGHT", "FLUORESCENCE", "PHASE CONTRAST"]
        in fluoresence mode:
            mode M is computed, corresponding to the Mp centile: M = centile(Mp). center_range = [centile(Mp-fluo_centile_extent), centile(Mp+fluo_centile_extent)]
            scale_range = [centile(fluo_centile_range[0]) - M, centile(fluo_centile_range[0]) + M ]
        in transmitted light mode: center_range = [mean - tl_sd_factor*sd, mean + tl_sd_factor*sd]; scale_range = [sd/tl_sd_factor., sd*tl_sd_factor]
        in phase contrast mode:
    tl_sd_factor : float
        Description of parameter `tl_sd_factor`.
    fluo_centile_range : list
        in fluoresence mode, interval for scale range in centiles
    fluo_centile_extent : float
        in fluoresence mode, extent for center range in centiles
    Returns
    -------
    scale_range (list(2)) , center_range (list(2))

    """
    if isinstance(dataset, (list, tuple)):
        scale_range, center_range = [], []
        for ds in dataset:
            sr, cr = get_center_scale_range(ds, raw_feature_name, fluoresence, tl_sd_factor, fluo_centile_range, fluo_centile_extent)
            scale_range.append(sr)
            center_range.append(cr)
        if len(dataset)==1:
            return scale_range[0], center_range[0]
        return scale_range, center_range
    if fluoresence:
        bins = dih.get_histogram_bins_IPR(*dih.get_histogram(dataset, raw_feature_name, bins=1000), n_bins=256, percentiles=[0, 95], verbose=True)
        histo, _ = dih.get_histogram(dataset, "/raw", bins=bins)
        center_range, scale_range = get_normalization_center_scale_ranges(histo, bins, fluo_centile_extent, fluo_centile_range, verbose=True)
        print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        return center_range, scale_range
    else:
        mean, sd = dih.get_mean_sd(dataset, "/raw", per_channel=True)
        mean, sd = np.mean(mean), np.mean(sd)
        print("mean: {} sd: {}".format(mean, sd))
        center_range, scale_range = [mean - tl_sd_factor*sd, mean + tl_sd_factor*sd], [sd/tl_sd_factor, sd*tl_sd_factor]
        print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        return center_range, scale_range

def gaussian_blur(img, sig):
    if len(img.shape)>2 and img.shape[-1]==1:
        return np.expand_dims(gaussian_filter(img.squeeze(-1), sig), -1)
    else:
        return gaussian_filter(img, sig)

def random_gaussian_blur(img, sig_min=1, sig_max=2):
    sig = uniform(sig_min, sig_max)
    return gaussian_blur(img, sig)

def add_gaussian_noise(img, sigma=0.035, scale_sigma_to_image_range=True):
    if is_list(sigma):
        if len(sigma)==2:
            sigma = uniform(sigma[0], sigma[1])
        else:
            raise ValueError("Sigma  should be either a list/tuple of lenth 2 or a scalar")
    if scale_sigma_to_image_range:
        sigma *= (img.max() - img.min())
    gauss = np.random.normal(0,sigma,img.shape)
    return img + gauss

def get_illumination_aug_fun(gaussian_blur_range:list, noise_sigma:float):
    """Returns illumination augmentation function.

    Parameters
    ----------
    gaussian_blur_range : list
        gaussian blur sigma is drawn randomly in [sigma_min, sigma_max]
        if None: no blurring
    noise_sigma : float
        noise intensity

    Returns
    -------
    function
        input/output batch (B, Y, X, C), applies random gaussian blur and adds random noise to each image YXC

    """
    def img_fun(img):
        # blur
        if getrandbits(1) and gaussian_blur_range is not None:
            img = random_gaussian_blur(img, gaussian_blur_range[0], gaussian_blur_range[1])
        # noise
        img = add_gaussian_noise(img, noise_sigma)
        return img
    return lambda batch : np.stack([img_fun(batch[i]) for i in range(batch.shape[0])])

def get_normalization_fun(center_range:list, scale_range:list):
    """Returns Normalization function.

    Parameters
    ----------
    center_range : list
        normalization center C is drawn randomly in interval center_range = [center min, center max]
    scale_range : list
        normalization scale S is drawn randomly in interval scale_range = [scale min, scale max]

    Returns
    -------
    function
        input/output batch (B, Y, X, C), applies random normalization

    """

    def img_fun(img):
        center = uniform(center_range[0], center_range[1])
        scale = uniform(scale_range[0], scale_range[1])
        return (img - center) / scale
    return img_fun

def adjust_histogram_range(img, min:float=0, max:float=1, initial_range:list=None):
    """Adjust intensity distribution range by a linear transformation to [min, max].

    Parameters
    ----------
    img : numpy array
        image to normalize
    min : float
        minimal value of target range
    max : float
        maximal value of target range
    initial_range : type
        intensity range of img

    Returns
    -------
    type
        numpy nd array with intensity in range [min, max]

    """
    if initial_range is None:
        initial_range=[img.min(), img.max()]
    return np.interp(img, initial_range, (min, max))

def get_phase_contrast_normalization_fun(min_range:float=0.1, range:list=[0,1]):
    """Adjust image range to a randomly drown range.

    Parameters
    ----------
    min_range : float
        contraint on target range: max-min >= min_range
    range : list
        range (min, max) in which target range is drawn

    Returns
    -------
    callable that inputs a numpy array
        normalization function

    """

    if range[1]-range[0]<min_range:
        raise ValueError("Range must be superior to min_range")

    def img_fun(img):
        vmin = uniform(range[0], range[1]-min_range)
        vmax = uniform(vmin+min_range, range[1])
        return adjust_histogram_range(img, vmin, vmax)
    return img_fun

################### helpers ##################################

def is_list(l):
    return isinstance(l, (list, tuple, np.ndarray))
