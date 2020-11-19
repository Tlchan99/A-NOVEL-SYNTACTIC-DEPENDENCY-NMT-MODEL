import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    print('start TDD')
    print('x', x)
    print('w', w)
    print('b', b)
    print('timesteps', timesteps)
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]
    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)
    # collapse time dimension and batch dimension together
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)

    print('x', x)
    print('x_input_dim=', input_dim)
    print('timesteps=', timesteps)
    print('x_output_dim=', output_dim)
    print('end TDD')
    return x

class AttentionDecoder(Recurrent):

    def __init__(self, units, tar_timesteps, output_dim, src_timesteps,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states 
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space
        references:
            Bahdanau, Dzmitry, Kunghyun Cho, and Yoshua Bengio. 
            "Neural machine translation by jointly learning to align and translate." 
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.tar_timesteps = tar_timesteps
        self.output_dim = output_dim
        self.src_timesteps = src_timesteps
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)


        print('start _init_')
        print('self.units', self.units)
        print('self.output_dim', self.output_dim)
        print('self.return_probabilities', self.return_probabilities)
        print('self.tar_timesteps', self.tar_timesteps)
        print('self.src_timesteps', self.src_timesteps)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        print('self.name', self.name)
        self.return_sequences = True  # must return sequences
        print('end_initi_')


    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.

        """
        print('start build')
        print('self.src_timesteps', self.src_timesteps)
        print('K.input_shape', type(input_shape))
        print('input_shape', input_shape)


        self.batch_size, self.timesteps, self.input_dim = input_shape

        print('self.batch_size', self.batch_size)
        print('self.src_timesteps', self.src_timesteps)
        print('self.input_dim', self.input_dim)
        print('self.tar_timesteps', self.tar_timesteps)

        if self.stateful:
            super(AttentionDecoder, self).reset_states()
        self.states = [None, None]  # y, s


        """
            Matrices for creating the context vector
        """
        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        print('context V_a', self.V_a)
        print('context W_a', self.W_a)
        print('context U_a', self.U_a)
        print('context b_a', self.b_a)

        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        print('reset C_r', self.C_r)
        print('reset U_r', self.U_r)
        print('reset W_r', self.W_r)
        print('reset b_r', self.b_r)
        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        print('udpate C_z', self.C_z)
        print('udpate U_z', self.U_z)
        print('udpate W_z', self.W_z)
        print('udpate b_z', self.b_z)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        print('proposal self.C_p=', self.C_p)
        print('proposal self.U_p=', self.U_p)
        print('proposal self.W_p=', self.W_p)
        print('proposal self.b_p=', self.b_p)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        print('output C_o', self.C_o)
        print('output U_o', self.U_o)
        print('output W_o', self.W_o)
        print('output b_o', self.b_o)
        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        print('self.W_s',self.W_s)
        self.input_spec = [InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        print('self.batch_size', self.batch_size)
        print('self.timesteps', self.timesteps)
        print('self.input_dim', self.input_dim)
        print('end build')
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        print('start call')
        self.x_seq = x
        print('self.tar_timesteps',self.tar_timesteps)
        print('x_seq',x)
        # apply a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)
        print('self._uxpb',self._uxpb)
        print('self.U_a',self.U_a)
        print('self.b_a',self.b_a)
        print('self.timesteps',self.timesteps)
        print('x_seq after TDD',x)
        print('endcall')
        return super(AttentionDecoder, self).call(x)


    def get_initial_state(self, inputs):
        print('get initial state starts')
        print('inputs:', inputs)
        print('inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))
        print('s0=',s0)

        # from keras.layers.recurrent to initialize a vector of (batchsize, output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        print('y0=',y0)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        print('y0 after sum axis=(1,2)',y0)
        y0 = K.expand_dims(y0)  # (samples, 1)
        print('y0 after expand_dims(y0)', [y0])
        y0 = K.tile(y0, [1, self.output_dim])
        print('y0,s0', [y0,s0])
        print('end initial state')
        return [y0, s0]

    def step(self, x, states):
        print('starts step')
        ytm, stm = states
        print('ytm', ytm, ytm.shape)
        print('stm', stm, stm.shape)
        print('self.timesteps', self.timesteps)
        print('self.x_seq', x)

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)
        print('_stm*timesteps', _stm)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)
        print('Wxstm', _Wxstm)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))

        print('Wxstm + self._uxpb', _Wxstm + self._uxpb)
        print('et', et)

        at = K.exp(et)
        print('at=exp(et)', at)

        at_sum = K.sum(at, axis=1)
        print('at_sum=sum(at, axis=1)', at_sum)

        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        print('at_sum_repeated', at_sum_repeated)

        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)
        print('at/=at_sum_repeated',at)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:


        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1-zt)*stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(st, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        print('context =',context, context.shape)
        print('rt =',rt)
        print('zt =',zt)
        print('s_tp =',s_tp)
        print('st =',st)
        print('stm =',stm)
        print('yt =',yt)
        print('self.return_probabilities',self.return_probabilities)
        print('at', at)
        print('[yt,st]',[yt,st])

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]
        print('end step')


    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        print('start compute_output_shape')
        print('input_shape =', input_shape)
        print('self.timesteps =', self.timesteps)
        print('self.tar_timesteps =', self.tar_timesteps)
        print('self.output_dim =', self.output_dim)
        print('self.return_probabilities', self.return_probabilities)
        if self.return_probabilities:
            return (None, self.tar_timesteps, self.tar_timesteps)
        else:
            return (None, self.tar_timesteps, self.output_dim)
        print('end compute_output_shape')

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        print('start get_config')
        config = {'units': self.units,'tar_timesteps':self.tar_timesteps, 'output_dim': self.output_dim,'src_timesteps': self.src_timesteps,'return_probabilities': self.return_probabilities}
        base_config = super(AttentionDecoder, self).get_config()
        print('output_dim', self.output_dim)
        print('self.units', self.units)
        print('self.tar_timesteps', self.tar_timesteps)
        print('end get_config')
        return dict(list(base_config.items()) + list(config.items()))
