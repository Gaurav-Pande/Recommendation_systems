from model.autoencoders import EncoderDecoder
import tensorflow.compat.v1 as tf


class TrainModel(EncoderDecoder):
	# https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
	def __init__(self, args, name_scope):
		super(TrainModel,self).__init__(args)
		self._init_parameters()

	def compute_loss(self, predictions, labels, num_labels):
		"""
		Mean squared error
		"""
		with tf.name_scope('loss'):
			loss = tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions,labels))),num_labels)
			return loss

	def validation_loss(self, train, x_test):
		"""
		root mean squared error loss between the predicted and actual ratings
		"""
		# make predictions from the model
		predictions = self.inference(train)
		# check to see what values are zero in test set as the data is sparse
		mask=tf.where(tf.equal(x_test,0.0), tf.zeros_like(x_test), x_test)
		# num of non zero values in the test set
		num_test_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32)
		# convert to boolean 
		bool_mask=tf.cast(mask,dtype=tf.bool) 
		predictions = tf.where(bool_mask, predictions, tf.zeros_like(predictions))
		mse=self.compute_loss(predictions, x_test, num_test_labels)
		rmse=tf.sqrt(mse)
		ab_ops=tf.div(tf.reduce_sum(tf.abs(tf.subtract(x_test,predictions))),num_test_labels)
		return predictions, mse, rmse, ab_ops


	def train(self, x):
		# make predictions from the model
		predictions = self.inference(x)
		# check to see what values are zero in train set as the data is sparse
		mask=tf.where(tf.equal(x,0.0), tf.zeros_like(x), x)
		# num of non zero values in the train set
		num_test_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32)
		# convert to boolean 
		bool_mask=tf.cast(mask,dtype=tf.bool) 
		predictions = tf.where(bool_mask, predictions, tf.zeros_like(predictions))
		mse=self.compute_loss(predictions, x, num_test_labels)
		if self.args['l2_reg'] == True:
			l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
			mse = mse +  self.args['lambda_value'] * l2_loss
		optimizer = tf.train.AdamOptimizer(self.args['learning_rate']).minimize(mse)
		return optimizer, tf.sqrt(mse)

