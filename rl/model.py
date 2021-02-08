from keras import activations
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

_, tf, tfv = try_import_tf()

class HerdingModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(HerdingModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(self.inputs)
        x = tf.keras.layers.Conv1D(filters=16, kernel_size=4, strides=2)(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=70)(x)
        x = tf.keras.layers.Dense(units=30)(x)
        layer_out = tf.keras.layers.Dense(units=num_outputs, name='action')(x)

        value_out = tf.keras.layers.Dense(units=1)(x)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(tf.cast(input_dict["obs"], tf.float32))

        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
