import socket
import numpy as np
import threading
import copy
import time
import Queue
import xgboost as xgb
from collections import OrderedDict
from time import gmtime, strftime

log_lock = threading.Lock()
def out(msg):
	with log_lock:
		with open('online_trainer_log.txt','a') as f:
			ctime = strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ': '
			f.write(ctime+msg+'\n')
			print ctime+msg


class Model(object):
	
	lock = threading.Lock()
	models = {}
        context_to_model = OrderedDict()

	def __init__(self, training_data, num_blocks):
    		self.param = {'bst:max_depth':10, 'bst:eta':1, 'silent':1, 'objective':'reg:linear'}
    		self.param['nthread'] = 4
    		self.param['eval_metric'] = 'rmse'
		self.dtrain = training_data
		self.eval_list = [(self.dtrain,'train')]
		self.num_round = 200	#keep it low
		self.model = None
		self.num_blocks = num_blocks

	def train(self):
		with Model.lock:
			self.model = xgb.train( self.param.items(), self.dtrain, self.num_round, self.eval_list, verbose_eval=False)
			if self.num_blocks not in Model.models:
				Model.models[self.num_blocks] = []
			Model.models[self.num_blocks].append(self)


	def predict(self, data, iteration):
		prediction = self.model.predict(data)
		errors = []
		actual = data.get_label()
		num_right, num_wrong = 0,0
		with open('model-window'+str(self.num_blocks)+'-'+str(iteration)+'.txt', "w+") as f:
			for x in range(0, len(prediction)):
				if int(round(prediction[x])) == int(actual[x]):
					num_right += 1
				else:
					num_wrong += 1   
				f.write('PREDICTION: '+str(round(prediction[x]))+' ACTUAL: '+str(actual[x])+'\n')
			accuracy = float(num_right)/(float(num_wrong) + float(num_right))
			f.write('ACCURACY: '+str(accuracy))
		
	@staticmethod
	def get_current_model(num_blocks):
		with Model.lock:
			if num_blocks not in Model.models:
				return None
			elif len(Model.models[num_blocks]) <= 0:
				return None
			else:
				return Model.models[num_blocks][-1]

class Block(object):
	BLOCK_NUM_COLUMN = 19
	blocks = OrderedDict()
        lock = threading.Lock()

	def __init__(self):
		self._block_number = None
		self.block_data = []
	
	def set_block_number(self, num):
		self._block_number = num

	def add(self, record):
		self.block_data.append(record)

	def __len__(self):
		return len(self.block_data) 

	def block_number(self):
		return self._block_number

	def __str__(self):
		out = 'Block Number: '+str(self._block_number)+'\n'
		out += 'Block Size: '+str(len(self))+'\n'
		return out

	@classmethod
	def all(cls):
		with Block.lock:
			all_blocks = []
			for x in Block.blocks:
				all_blocks.append(Block.blocks[x])
			return all_blocks

	@classmethod
	def last(cls, num):
		with Block.lock:
			last_x = []
			if num > len(Block.blocks):
				num = len(Block.blocks())
			last_blocks = reversed(Block.blocks)
			for x in range(0,num):
				last_x.insert(0,Block.blocks[last_blocks.next()])
			return last_x
	
	def serialize(self, exclude_columns = None, include_columns = None):
		if len(self.block_data) == 0:
			return []
		if include_columns == None:
			include_columns = []
			for x in range(0,len(self.block_data[0])):
				include_columns.append(x)
		out_data = []
		if exclude_columns is None:
			exclude_columns = ()
		if isinstance(exclude_columns,(int,long)):
			exclude_columns = (exclude_columns)
		if isinstance(include_columns, (int,long)):
			include_columns = (include_columns)
		for record in self.block_data:
			rec = []
			for x in range(0, len(record)):
				if x not in exclude_columns:
					if x in include_columns:
						rec.append(record[x])
			out_data.append(rec)
		return out_data

	@staticmethod
	def all_to_ndarray(blocks, label_column):
		feature_data, label_data = [],[]
		for x in blocks:
			feature_data = feature_data + (x.serialize(exclude_columns = (9,label_column,18,19)))
			label_data = label_data + (x.serialize(include_columns = (label_column,)))
		return np.array(feature_data), np.array(label_data)

	@staticmethod
	def all_to_DMatrix(blocks, label_column):
		feature, l = Block.all_to_ndarray(blocks, label_column)
		print(feature.shape)
		print(l.shape)
		return xgb.DMatrix(feature, label = l)
	

class RollingModelTrainer(object):
	def socket_reader(self):
		curr_block = Block()
		while self.reading is True:
			new_record_line = self.socket_file.readline()
			block_records, block_record_data = [],[]
			is_sentinel = False
 			# This would've failed. I changed it to get the first 
                        # column.
			if len(new_record_line.split(' ')) <= 0:
				continue
			elif new_record_line.split(' ')[0] == '':
				continue

			if float(new_record_line.split(' ')[0]) == 0.0:
				is_sentinel = True

			for elem in new_record_line.split(' '):
				block_record_data.append(float(elem))
			#out('Recieved Record: '+str(block_record_data))
			last_block_num_value = block_record_data[Block.BLOCK_NUM_COLUMN]
			if is_sentinel is True:
				self.recieved_blocks.put(curr_block)
				curr_block = Block()
			else:
				curr_block.set_block_number(last_block_num_value)
				curr_block.add(block_record_data)
			if self.host is not None:
				with open('block_stream.txt', 'a') as f:
					f.write(new_record_line)
				
	def model_trainer(self):
		last_block_trained = 0
		our_blocks = []
		while True:
			block_to_train = self.blocks_to_train.get()
			print("block num: ", block_to_train.block_number())
			self.blocks_to_train.task_done()
			our_blocks.append(block_to_train)
			for rolling_window_size in self.rolling:
				if len(our_blocks) >= rolling_window_size:
					training_data = Block.all_to_DMatrix(our_blocks[-rolling_window_size:],10)
					out('Training new model')
					m = Model(training_data, rolling_window_size )
					m.train()
					out('Trained new model')

	def model_predictor(self):
                iteration = 0
		while True:
			block_to_predict = self.recieved_blocks.get()
			self.recieved_blocks.task_done()
			for rolling_window_size in self.rolling:
				curr_model = Model.get_current_model(rolling_window_size)
				if curr_model is not None: #run predictions if we already have a trained_model
					out('Running predictions using latest model of window size'+str(rolling_window_size))
					curr_model.predict(Block.all_to_DMatrix([block_to_predict], 10), iteration)
			iteration += 1
			self.blocks_to_train.put(block_to_predict)



	def __init__(self, host = None, file_name  = None, rollover_blocks = (0, 1, 2, 4, 8, 16, 32, 64), feature_column = 10):
		self.trained_models = []
		self.socket, self.reading = socket.socket(socket.AF_INET, socket.SOCK_STREAM), False
		self.recv_buffer = ''
		self.recieved_blocks, self.blocks_to_train = Queue.Queue(1),Queue.Queue(1)		
		self.reading = True
		self.feature_column = feature_column
		self.max_block, self.last_training_block = None, 0
		self.new_model, self.model_lock = None, threading.Lock()
		self.rolling = rollover_blocks
		self.past_records = []
		self.host = host
		self.file = file_name
		if self.host is not None:
			self.socket.connect(self.host)
			self.socket_file = self.socket.makefile('r')
		else:
			self.socket_file = open(self.file, 'r')

	def run(self):
		#Threads
		print("KEK")
		reader = threading.Thread(target = self.socket_reader)

		reader.start()

		trainer = threading.Thread(target = self.model_trainer)
		trainer.start()

		predictor = threading.Thread(target = self.model_predictor)
		predictor.start()
		print('Running...')


if __name__ == "__main__":
	rolling_trainer= RollingModelTrainer(host = ('107.178.208.24',6541))
	rolling_trainer.run()
