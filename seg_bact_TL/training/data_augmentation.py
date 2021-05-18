from random import getrandbits, uniform
import numpy as np
import dataset_iterator.helpers as dih
import numpy as np

def get_histogram_normalization_center_scale_ranges(histogram, bins, center_percentile_extent, scale_percentile_range, verbose=False):
    assert dih is not None, "dataset_iterator package is required for this method"
    mode_value = dih.get_modal_value(histogram, bins)
    mode_percentile = dih.get_percentile_from_value(histogram, bins, mode_value)
    print("model value={}, model percentile={}".format(mode_value, mode_percentile))
    assert mode_percentile<scale_percentile_range[0], "mode percentile is {} and must be lower than lower bound of scale_percentile_range={}".format(mode_percentile, scale_percentile_range)
    percentiles = [max(0, mode_percentile-center_percentile_extent), min(100, mode_percentile+center_percentile_extent)]
    scale_percentile_range = ensure_multiplicity(2, scale_percentile_range)
    if isinstance(scale_percentile_range, tuple):
        scale_percentile_range = list(scale_percentile_range)
    percentiles = percentiles + scale_percentile_range
    values = dih.get_percentile(histogram, bins, percentiles)
    mode_range = [values[0], values[1] ]
    scale_range = [values[2] - mode_value, values[3] - mode_value]
    if verbose:
        print("normalization_center_scale: modal value: {}, center_range: [{}; {}] scale_range: [{}; {}]".format(mode_value, mode_range[0], mode_range[1], scale_range[0], scale_range[1]))
    return mode_range, scale_range

def get_center_scale_range(dataset, raw_feature_name = "/raw", fluoresence=False):
    bins = dih.get_histogram_bins_IPR(*dih.get_histogram(dataset_path, raw_feature_name, bins=1000), n_bins=256, percentiles=[0, 95], verbose=True)
    histo, _ = dih.get_histogram(dataset_path, "/raw", bins=bins)
    if fluoresence:
        center_range, scale_range = get_histogram_normalization_center_scale_ranges(histo, bins, 0, [75, 99.9], verbose=True)
        print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        return center_range, scale_range
    else:
        mean, sd = dih.get_mean_sd(dataset_path, "/raw", per_channel=True)
        mean, sd = np.mean(mean), np.mean(sd)
        center_range, scale_range = get_histogram_normalization_center_scale_ranges(histo, bins, 0, [75, 99.9], verbose=True)
        print("mean: {} sd: {}".format(mean, sd))
        print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0]- sd, center_range[0] + sd, sd*0.5, sd*2))
        return [center_range[0]- 3*sd, center_range[0] + 3*sd], [sd/3., sd*3]

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

def get_illumination_aug_fun(center_range, scale_range, gaussian_blur_range, noise_sigma):
    def img_fun(img):
        # normalization
        center = uniform(center_range[0], center_range[1])
        scale = uniform(scale_range[0], scale_range[1])
        img = (img - center) / scale
        # blur
        if getrandbits(1) and gaussian_blur_range is not None:
            img = random_gaussian_blur(img, gaussian_blur_range[0], gaussian_blur_range[1])
        # noise
        img = pp.add_gaussian_noise(img, noise_sigma)
        return img
    return lambda batch : np.stack([img_fun(batch[i]) for i in range(batch.shape[0])])
