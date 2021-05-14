from tensorflow.keras.layers import Layer
from tensorflow import pad
from .utils import ensure_multiplicity

class PaddingYX3D(Layer):
  def __init__(self, paddingYX=(1, 1), mode="CONSTANT", **kwargs):
    paddingYX = ensure_multiplicity(2, paddingYX)
    self.padding = tuple(paddingYX)
    self.mode = mode
    super().__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1], input_shape[2] + 2 * self.padding[0], input_shape[3] + 2 * self.padding[1], input_shape[4])

  def call(self, input_tensor, mask=None):
    padding_height, padding_width = self.padding
    return pad(input_tensor, [[0,0], [0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], self.mode)

  def get_config(self):
      config = super().get_config().copy()
      config.update({"padding": self.padding, "mode":self.mode})
      return config
