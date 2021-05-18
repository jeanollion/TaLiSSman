import edt
import numpy as np
import h5py
from dataset_iterator import MultiChannelIterator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error as mse
from dataset_iterator.tile_utils import extract_tile_random_zoom_function
from .data_augmentation import *
from ..model import get_unet

def get_edt_fun():
    return lambda labels : np.stack([edt.edt(labels[i,...,0], black_border=False)[...,np.newaxis] for i in range(labels.shape[0])])

def channels_postpprocessing(fluo_fun, label_fun):
    def fun(batch_by_channel):
        batch_by_channel[0] = fluo_fun(batch_by_channel[0])
        batch_by_channel[1] = label_fun(batch_by_channel[1])
    return fun

def get_train_test_iterators(dataset,
    center_range, scale_range,
    tile_params = dict(tile_shape=(256, 256), n_tiles=9, zoom_range=[0.6, 1.6], aspect_ratio_range=[0.75, 1.5] ) },
    elasticdeform_parameters = {},
    raw_feature_name="/raw", label_feature_name="/regionLabels"
    training_selection_name="train/", validation_selection_name="eval/" ):

    extract_tile_function = extract_tile_random_zoom_function(**tile_params) if tile_params is not None else None
    def random_channel_slice(nchan): # random offset on chosen slices to simulate focus variations
        halfnchan = nchan//2
        nuo=5
        noff=halfnchan-1-(nuo-1)*2 + 1
        off = np.random.randint(noff)
        idx = [off + 2*i for i in range(nuo)] + [off + 2*i + halfnchan for i in range(nuo)]
        return idx
    # def random_channel_slice(nchan):
    #     center = (nchan-1)//2
    #     intervalRange=12
    #     step = 2
    #     off_range = 5
    #     min_off = center - intervalRange
    #     off_range = min(min_off, off_range)
    #     off = np.random.randint(-off_range, off_range+1) + center - intervalRange
    #     idx = [off + step*i for i in range((intervalRange*2)//step+1)]
    #     return idx

    params = dict(dataset=dataset,
        channel_keywords=[raw_feature_name, label_feature_name], # channel keyword must correspond to the name of the extracted features
        output_channels= [1],
        mask_channels = [1],
        channel_slicing_channels = {0:random_channel_slice},
        elasticdeform_parameters = elasticdeform_parameters,
        extract_tile_function = extract_tile_function,
        channels_postprocessing_function = channels_postpprocessing(get_illumination_aug_fun(center_range, scale_range, None, 0.035), get_edt_fun()),
        batch_size=4,
        perform_data_augmentation=True,
        shuffle=True)

    train_it = MultiChannelIterator(group_keyword=training_selection_name, **params)
    test_it = MultiChannelIterator(group_keyword=validation_selection_name, **params)
    return train_it, test_it

def get_model(model_params=dict(n_filters=64, n_z=11), learning_rate = 2e-4, saved_weights_file=None):
    model = get_unet(**model_params)
    if saved_weights_file is not None:
        model.load_weights(saved_weights_file)
    model.compile(optimizer=Adam(learning_rate), loss=mse)
    return model
