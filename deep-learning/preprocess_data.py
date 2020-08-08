import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import string

def fill_column(df):
	"""
	Method to introduce a normal distributed data as rating
	"""
	np.random.seed(1)
	lower = 1
	upper = 10
	mu = 6
	sigma = 2
	N = df.shape[0]
	samples = scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)
	df.EVENT_ID = np.round_(samples)
	return df

def random_char(y):
	return ''.join(random.choice(string.ascii_letters) for x in range(y))


def generate_users(df):
	df.USER_ID = random_char(7)+"@gmail.com"
	return df

def prepare_data():
	df = pd.read_csv('data/recom.csv')
	# prepare a summary of the data
	# df = generate_users(df)
	df = fill_column(df)
	# df.to_csv('data/final.csv')
	num_uniq_users, num_uniq_items = unique_user_item(df)
	popular_users = df.USER_ID.value_counts()
	popular_labs = df.ITEM_ID.value_counts()
	print("#####################################################")
	print("Number of unique users in the dataset",num_uniq_users)
	print("Number of unique labs in the dataset",num_uniq_items)
	print("")
	print("Most popular users in the dataset:\n",popular_users.head(20))
	print("")
	print("People who have taken very few labs:\n",popular_users.tail(10))
	print("")
	print("Most popular items/labs in the dataset:\n",popular_labs.head(20))
	print("#####################################################")
	input("Press the <ENTER> key to continue...")
	df = remove_extra_columns(df, columns = ['LAB_START', 'TIMELAPSE', 'LANGUAGE'])
	df = normalize_data(df)
	user_item_matrix, users, items = prep_user_item_matrix(df, 'click')
	return df, user_item_matrix[:4765], user_item_matrix[4765:5000], user_item_matrix[5000:], num_uniq_users, num_uniq_items, users, items

def remove_extra_columns(df, columns=['LAB_START', 'TIMELAPSE', 'LANGUAGE']):
	df = df.drop(columns=columns)
	return df

def normalize_data(df):
	scaler = MinMaxScaler()
	df['EVENT_ID'] = df['EVENT_ID'].values.astype(float)
	rating_scaled = pd.DataFrame(scaler.fit_transform(df['EVENT_ID'].values.reshape(-1,1)))
	df['EVENT_ID'] = rating_scaled
	return df


def unique_user_item(df):
	unique_users = df['USER_ID'].nunique()
	unique_items = df['ITEM_ID'].nunique()
	return unique_users, unique_items


def prep_user_item_matrix(df, type = 'rating'):
	if type == 'rating':
		df = df.drop_duplicates(['USER_ID', 'ITEM_ID'])
		user_item_matrix = df.pivot(index='USER_ID', columns='ITEM_ID', values='EVENT_ID')
		user_item_matrix.fillna(0, inplace=True)
		users = user_item_matrix.index.tolist()
		items = user_item_matrix.columns.tolist()
		user_item_matrix = user_item_matrix.to_numpy()
	elif type == 'click':
		df = df.drop_duplicates()
		grouped_df = df.groupby(['USER_ID', 'ITEM_ID']).sum().reset_index()
		user_item_matrix = grouped_df.pivot(index='USER_ID', columns='ITEM_ID', values='EVENT_ID')
		user_item_matrix.fillna(0, inplace=True)
		users = user_item_matrix.index.tolist()
		items = user_item_matrix.columns.tolist()
		user_item_matrix = user_item_matrix.to_numpy()
	else:
		print("User Item matrix conversion error!!")
	return user_item_matrix, users, items


