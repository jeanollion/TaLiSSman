from tensorflow.keras.layers import Conv2D, Conv3D, Input, MaxPool2D, MaxPool3D, UpSampling2D, Concatenate, Permute
from tensorflow.keras.models import Model
import tensorflow as tf
from .padding import PaddingYX3D

def get_unet(n_filters=64, depth=4, n_z=1, conv_channel=True, conv_z=True, stride_z=1, name = "unet"):
    input = Input(shape = (None, None, n_z ), name=name+"_input")
    residual = []
    downsampled = [input]
    nf = 64
    for i in range(depth-1):
        down, res = downsampling_blockZ(input, nf, n_z, stride_z, conv_channel, name, 0) if i==0 and n_z>1 and conv_z else downsampling_block(downsampled[-1], nf, True, name, i)
        downsampled.append(down)
        residual.append(res)
        nf = nf*2
    last_down = downsampling_block(downsampled[-1], nf, False, name, depth-1)
    upsampled = [last_down]
    for i in range(depth-2, -1, -1):
        nf=nf//2
        up = upsampling_block(upsampled[-1], residual[i], nf, name, i)
        upsampled.append(up)

    conv = Conv2D(filters=n_filters, kernel_size=1, padding='same', activation="relu", name=name+"_conv1x1_1")(upsampled[-1])
    conv = Conv2D(filters=n_filters, kernel_size=1, padding='same', activation="relu", name=name+"_conv1x1_2")(conv)
    output = Conv2D(filters=1, kernel_size=1, padding='same', name=name+"_output")(conv)
    return Model(input, output, name=name)

def downsampling_block(input, n_filters, maxpool, name, l_idx):
    conv = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation="relu", name=name+"_down_conv_{}_1".format(l_idx))(input)
    res = Conv2D(filters=n_filters if maxpool else n_filters//2, kernel_size=3, padding='same', activation="relu", name=name+"_down_conv_{}_2".format(l_idx))(conv)
    if maxpool:
        down = MaxPool2D(pool_size=2, name=name+"_down_{}".format(l_idx))(res)
        return down, res
    else:
        return res

def downsampling_blockZ(input, n_filters, n_z ,stride_z, conv_channel, name, l_idx):
    res = reduceConv3D_block(input, n_z, n_filters, stride_z, name)
    if conv_channel:
        conv = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation="relu", name=name+"_down_conv_{}_1".format(l_idx))(input)
        conv2 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation="relu", name=name+"_down_conv_{}_2".format(l_idx))(conv)
        conv = Concatenate(axis=-1, name=name+"_concatZ")([conv2, res])
        res = Conv2D(filters=n_filters, kernel_size=1, padding='same', activation="relu", name=name+"_down_conv_{}_1x1".format(l_idx))(conv)

    down = MaxPool2D(pool_size=2, name=name+"_down_{}".format(l_idx))(res)
    return down, res

def upsampling_block(input, residual, n_filters, name, l_idx):
    up = UpSampling2D(size=2, interpolation='nearest', name = name+"_up_{}".format(l_idx))(input)
    upconv = Conv2D(filters=n_filters, kernel_size=2, padding='same', activation="relu", name=name+"_upconv_{}".format(l_idx))(up)
    concat = Concatenate(axis=-1, name =name+"_concat_{}".format(l_idx))([residual, upconv])
    conv = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation="relu", name=name+"_up_conv_{}_1".format(l_idx))(concat)
    return Conv2D(filters=n_filters if l_idx==0 else n_filters//2, kernel_size=3, padding='same', activation="relu", name=name+"_up_conv_{}_2".format(l_idx))(conv)

def reduceConv3D_block(input, n_z, n_filters, stride_z, name="unet"):
    assert n_z>1, "nz must be >1"
    # reshape to have channel dims as z
    input = Permute( dims=(3, 1, 2) )(input)
    conv = tf.expand_dims(input, axis=-1) # add a channel axis
    last_op_conv=False
    nz=n_z
    i=0
    while nz>1:
        if not last_op_conv:
            conv = PaddingYX3D((1,1))(conv)
            conv = Conv3D(filters=n_filters, kernel_size=3 if nz>=3 else (2, 3, 3), padding='valid', activation="relu", name=name+"conv3D_{}".format(i//2))(conv)
            nz=nz-2
            last_op_conv=True
        else:
            if nz-2>=stride_z:
                stride_z=1 
            conv = MaxPool3D(pool_size = (3, 1, 1)if nz>=3 else (2, 1, 1), strides=(stride_z, 1, 1) if nz-2>=stride_z else 1, padding='valid', name=name+"pool3D_{}".format(i//2))(conv)
            last_op_conv=False
            nz = (nz-2)//(stride_z if nz-2>=stride_z else 1) + (nz%stride_z if stride_z>1 else 0)
        i=i+1
        print("op: {}, shape: {}, nz: {}".format(i, tf.shape(conv), nz))
    return tf.squeeze(conv, [1])
