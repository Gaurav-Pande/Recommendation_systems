import pandas as pd 
import numpy as np 
import tensorflow.compat.v1 as tf 
import os
from model.train_model import TrainModel 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess_data import  prepare_data
from model.autoencoders import AutoEncoders
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.set_random_seed(1234)


args = {
	'epochs':50,
	'learning_rate':2e-3,
	'batch_size':64,
	'l2_reg':False,
	'lambda_value':0.01,
	'input_size':252,
	'hidden_dim1':128,
	'hidden_dim2':64,
	'train_data_size':200,
	'checkpoints_path': os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'checkpoints/model.ckpt'))
}


def define_model_param(y_true, y_pred):
	loss = tf.losses.mean_squared_error(y_true, y_pred)
	# mAP = tf.metrics.average_precision_at_k(y_true,y_pred,10)
	mAP = 0
	optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)
	# optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)
	eval_x = tf.placeholder(tf.int32, )
	eval_y = tf.placeholder(tf.int32, )
	pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)
	init = tf.global_variables_initializer()
	local_init = tf.local_variables_initializer()
	return loss, mAP, optimizer, eval_x, eval_y, pre, pre_op, init, local_init

def get_top10_recommendation(preds, df, pred_data, users, items):
	pred_data = pred_data.append(pd.DataFrame(preds))
	pred_data = pred_data.stack().reset_index(name='EVENT_ID')
	pred_data.columns = ['USER_ID', 'ITEM_ID', 'EVENT_ID']
	pred_data['USER_ID'] = pred_data['USER_ID'].map(lambda value: users[value])
	pred_data['ITEM_ID'] = pred_data['ITEM_ID'].map(lambda value: items[value])
	keys = ['USER_ID', 'ITEM_ID']
	index_1 = pred_data.set_index(keys).index
	index_2 = df.set_index(keys).index
	top_ten_ranked = pred_data[~index_1.isin(index_2)]
	top_ten_ranked = top_ten_ranked.sort_values(['USER_ID', 'EVENT_ID'], ascending=[True, False])
	top_ten_ranked = top_ten_ranked.groupby('USER_ID').head(10)
	return top_ten_ranked

def compute_mse(predictions, true_labels, r, c):
	return tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions,true_labels))),r*c)


def fetch_recommendations(user_name, top_10_recommendation):
	recommendation = top_10_recommendation.loc[top_10_recommendation['USER_ID']==user_name]
	return  recommendation[['ITEM_ID','EVENT_ID']]


def main2():
	pred_data = pd.DataFrame()
	df, train_data, validation_data, test_data, unique_users, unique_items, users, items  = prepare_data()
	num_input = unique_items
	print(num_input)
	tf.compat.v1.disable_eager_execution()
	X = tf.placeholder(tf.float64, [None, num_input])
	model = AutoEncoders(args, num_input)
	y_pred, y_true = model.forward(X)
	loss, mAP, optimizer, eval_x, eval_y, pre, pre_op, init, local_init = define_model_param(y_true, y_pred)
	with tf.Session() as session:
		batch_size = args['batch_size']
		epochs = args['epochs']
		session.run(init)
		session.run(local_init)
		row, column = train_data.shape
		num_batches =  int( row / batch_size)
		user_item_matrix = np.array_split(train_data, num_batches)
		user_item_matrix = np.asarray(user_item_matrix)
		for epoch in range(epochs):
			average_cost = 0
			validation_cost = 0
			for batch in user_item_matrix:
				_, c = session.run([optimizer, loss], feed_dict={X: batch})
				# _, valid_c = session.run([optimizer, loss], feed_dict={X: np.asarray(validation_data)}) 
				average_cost+=c
				# validation_cost +=valid_c
			val_pred = session.run(y_pred, feed_dict={X: np.asarray(validation_data)})
			u, i = validation_data.shape
			val_loss = compute_mse(val_pred, validation_data, u, i)
			average_cost/=num_batches
			print("============")
			print("Epoch : {}".format(epoch+1))
			print("============")
			print("Train error: {} || Validation error: {}".format(average_cost, session.run(val_loss)))
		user_item_matrix_stacked = np.concatenate(user_item_matrix, axis=0)
		test_pred = session.run(y_pred, feed_dict={X: test_data})
		u,i = test_data.shape
		test_loss = compute_mse(test_pred, test_data, u, i)
		print("--------------------------------------------------------------------------")
		print("Test error {}".format(session.run(test_loss)))
		preds = session.run(y_pred, feed_dict={X: user_item_matrix_stacked})
		top_10_recommendation = get_top10_recommendation(preds, df, pred_data, users, items)
		# print(top_10_recommendation)
		return top_10_recommendation

def fetch_item_counts(recommendations):
	df = pd.read_csv('data/recom.csv')
	res={}
	for item in recommendations.ITEM_ID:
		res[item]=df.ITEM_ID[df.ITEM_ID == item].value_counts()[0]
	return pd.DataFrame(res.items(), columns=['ITEM_ID', 'Counts'])


def main():
	# get the data
	# This method is depricated, and for experimental purpose only
	num_batches = args['train_data_size']//args['batch_size']
	with tf.Graph().as_default():
		train_data, train_data_infer, test_data  = prepare_data()
		train_data = tf.data.Dataset.from_tensors(train_data.astype(np.float32))
		train_data_infer = tf.data.Dataset.from_tensors(train_data_infer.astype(np.float32))
		test_data = tf.data.Dataset.from_tensors(test_data.astype(np.float32))
		model=TrainModel(args, 'training')
		iter_train = train_data.make_initializable_iterator()
		iter_train_infer=train_data_infer.make_initializable_iterator()
		iter_test=test_data.make_initializable_iterator()
		x_train= iter_train.get_next()
		x_train_infer=iter_train_infer.get_next()
		x_test=iter_test.get_next()
		# print(x_train.shape)
		train_op, train_loss_op=model.train(x_train)
		prediction, labels, test_loss_op, mae_ops=model.validation_loss(x_train_infer, x_test)
		saver=tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			train_loss, test_loss, mae= 0, [], []
			for epoch in range(args['epochs']):
				sess.run(iter_train.initializer)
				sess.run(iter_train_infer.initializer)
				sess.run(iter_test.initializer)
				for batch_nr in range(num_batches):
					_, loss_=sess.run((train_op, train_loss_op))
					train_loss+=loss_
				for i in range(args['train_data_size']):
					pred, labels_, loss_, mae_=sess.run((prediction, labels, test_loss_op,mae_ops))
					test_loss.append(loss_)
					mae.append(mae_)   
				print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f, mean_abs_error: %.3f'
                      %(epoch,(train_loss/num_batches),np.mean(test_loss), np.mean(mae)))
				if np.mean(mae)<0.9:
					saver.save(sess, args['checkpoints_path'])
				train_loss=0
				test_loss=[]
				mae=[]
                    
if __name__ == "__main__":
    top_10_recommendation = main2()
    print("")
    user_input = input("Training done, Please enter the User name to show recommendations : ")
    recommendations = fetch_recommendations(user_input, top_10_recommendation)
    recommendations.rename(columns={'ITEM_ID': 'ITEM_ID', 'EVENT_ID': 'PROB_RECOM'}, inplace=True)
    item_counts = fetch_item_counts(recommendations)

    if recommendations.empty:
    	print("Cannot Recommend for this user, Please check if the userid entered correctly")
    else:
    	print("Top 10 recommendations for the user: {} are : \n{}".format(user_input, recommendations[['ITEM_ID','PROB_RECOM']]))
    	print("")
    	print("Popluarity of Recommended items\n{}".format(item_counts))
    yn = input("Do you want to continue? Y/N ")
    while yn == 'Y' or yn == 'y':
    	user_input = input("Please enter the User name to show recommendations : ")
    	recommendations = fetch_recommendations(user_input, top_10_recommendation)
    	if recommendations.empty:
    		print("Cannot Recommend for this user, Please check if the userid entered correctly")
    	else:
    		print("Top 10 recommendations for the user: {} are : \n{}".format(user_input, recommendations[['ITEM_ID','PROB_RECOM']]))
    		print("")
    		print("Popluarity of Recommended items\n{}".format(item_counts))
    	yn = input("Do you want to continue? Y/N ")
    else:
    	input("Press the <ENTER> key to exit the execution...")
    	print("Thank You..BBye!!")
    	exit(0)
    


