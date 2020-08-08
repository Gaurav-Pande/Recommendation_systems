import tensorflow.compat.v1 as tf 

class EncoderDecoder(object):
    def __init__(self, args):
        self.args = args
        self.weight_initializer = tf.random_normal_initializer(mean=0.0, stddev = 1.0)
        self.bias_initializer = tf.zeros_initializer()
    
    def _init_parameters(self):
        with tf.name_scope('weights'):
            input_size = self.args['input_size']
            hidden_dim1 = self.args['hidden_dim1']
            hidden_dim2 = self.args['hidden_dim2']
            self.encoder_1=tf.get_variable(name='encoder_1', shape=(input_size,hidden_dim1),initializer=self.weight_initializer)
            self.encoder_2=tf.get_variable(name='encoder_2', shape=(hidden_dim1,hidden_dim2),initializer=self.weight_initializer)
            self.decoder_1=tf.get_variable(name='decoder_1', shape=(hidden_dim2,hidden_dim1),initializer=self.weight_initializer)
            self.decoder_2=tf.get_variable(name='decoder_2', shape=(hidden_dim1,input_size),initializer=self.weight_initializer)
        with tf.name_scope('biases'):
            self.b1=tf.get_variable(name='bias_1', shape=(hidden_dim1),initializer=self.bias_initializer)
            self.b2=tf.get_variable(name='bias_2', shape=(hidden_dim2),initializer=self.bias_initializer)
            self.b3=tf.get_variable(name='bias_3', shape=(hidden_dim1),initializer=self.bias_initializer)

    def inference(self, x):
        with tf.name_scope('inference'):
        	# print("shape", x.shape)
        	a1=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, self.encoder_1),self.b1))
        	a2=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a1, self.encoder_2),self.b2))
        	a3=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a2, self.decoder_1),self.b3))
        	a4=tf.matmul(a3, self.decoder_2) 
        return a4
    
class AutoEncoders(object):
	def __init__(self, args, num_input):
		self.args = args
		# self.x = tf.placeholder(tf.float64, [None, num_input])
		self.num_input = num_input

	def init_parameter(self):
		hidden_dim1 = self.args['hidden_dim1']
		hidden_dim2 = self.args['hidden_dim2']
		weights = {
    	'encoder_h1': tf.Variable(tf.random_normal([self.num_input, hidden_dim1], dtype=tf.float64)),
    	'encoder_h2': tf.Variable(tf.random_normal([hidden_dim1, hidden_dim2], dtype=tf.float64)),
    	'decoder_h1': tf.Variable(tf.random_normal([hidden_dim2, hidden_dim1], dtype=tf.float64)),
    	'decoder_h2': tf.Variable(tf.random_normal([hidden_dim1, self.num_input], dtype=tf.float64)),
		}

		biases = {
		'encoder_b1': tf.Variable(tf.random_normal([hidden_dim1], dtype=tf.float64)),
		'encoder_b2': tf.Variable(tf.random_normal([hidden_dim2], dtype=tf.float64)),
		'decoder_b1': tf.Variable(tf.random_normal([hidden_dim1], dtype=tf.float64)),
		'decoder_b2': tf.Variable(tf.random_normal([self.num_input], dtype=tf.float64)),
		}


		return weights, biases

	def encoder(self, x, weights, biases):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
		return layer_2

	def decoder(self, x, weights, biases):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
		return layer_2

	def forward(self, X):
		weights, biases = self.init_parameter()
		encode = self.encoder(X, weights, biases)
		decode = self.decoder(encode, weights, biases)
		y_pred = decode
		y_true = X
		return y_pred, y_true