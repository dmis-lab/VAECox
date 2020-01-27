import os
import gc
import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from progress.bar import Bar
from tqdm import trange
import math
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.cluster import KMeans
from vae_utils import *
import logging
from multiprocessing import Pool

cancer_list_dict = {
	'ching': ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD'],
	'wang': ['ACC', 'BLCA', 'BRCA', 'CESC', 'UVM', 'CHOL', 'ESCA', 'HNSC', 'KIRC', 'KIRP',
			 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC', 'UCS'],

	'all': ['ACC', 'BLCA', 'BRCA', 'CESC', 'UVM', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
			'HNSC', 'KICH', 'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
			'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'STES',
			'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']
}

acti_func_dict = {
	'ReLU': nn.ReLU(),
	'Tanh': nn.Tanh(),
	'LeakyReLU': nn.LeakyReLU(negative_slope=0.001),
	'Tanhshrink': nn.Tanhshrink(),
	'Hardtanh': nn.Hardtanh()
}


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cpu")

class AE(nn.Module):
	def __init__(self, config, logger, num_features):
		self.LOGGER = logger
		self.num_features = num_features
		self.save_path = './results/{}/{}/'.format(config.model_type, config.session_name)
		#if not os.path.exists(self.save_path):
		os.makedirs(self.save_path, exist_ok=True)
		self.max_epochs = config.max_epochs
		self.learning_rate = config.learning_rate
		self.opti_name = config.model_optimizer
		self.weight_sparsity = config.weight_sparsity
		self.weight_decay = config.weight_decay
		self.dropout_rate = config.dropout_rate
		self.save_mode = config.save_mode
		self.device_type = config.device_type
		self.exclude_imp = config.exclude_impute
		self.evaluation = dict()
		self.batch_size = config.batch_size
		self.batch_flag = False
		self.batch_index = 0
		self.global_train_loss = 0.0
		self.global_valid_loss = 0.0
		self.best_valid_loss = 999.999
		self.best_valid_flag = False

		super(AE, self).__init__()

		self.encode = nn.Sequential(
			nn.Linear(self.num_features, config.hidden_nodes),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, 128),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)
		self.decode = nn.Sequential(
			nn.Linear(128, config.hidden_nodes),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, self.num_features)
		)

		self.hp = 'hn_{}_af_{}_ms_{}_mt_{}_vd_{}'.format(config.hidden_nodes,
			config.acti_func,
			config.model_struct,
			config.model_type,
			config.vae_data)
		try:
			with open(self.save_path + 'best_loss.txt', "r") as fr:
				for line in fr.readlines():
					l = line.split('\t')
					self.best_valid_loss = float(l[1])
					print('BEST VALID LOSS: {}'.format(self.best_valid_loss))
		except IOError:
			with open(self.save_path + 'best_loss.txt', "w") as fw:
				fw.write(self.hp + '\t999.999')

	def write_best_loss(self):
		file_name = self.save_path + 'best_loss.txt'
		with open(file_name, "w") as fw:
			fw.write('{}\t{}'.format(self.hp, self.best_valid_loss))

	def init_layers(self):
		nn.init.xavier_normal_(self.encode[0].weight.data)
		nn.init.xavier_normal_(self.decode[0].weight.data)
		try:
			nn.init.xavier_normal_(self.encode[3].weight.data)
			nn.init.xavier_normal_(self.decode[3].weight.data)
		except:
			pass
		try:
			nn.init.xavier_normal_(self.encode[6].weight.data)
			nn.init.xavier_normal_(self.decode[6].weight.data)
		except:
			pass

	def _l1_norm(self, model):
		l1_loss = 0.0
		for param in model.parameters():
			l1_loss += torch.sum(torch.abs(param))
		return self.weight_sparsity * l1_loss

	def dimension_reduction(self, x):
		return self.encode(x)

	def forward(self, x, m=None, coo=None):
		z = self.encode(x)
		recon = self.decode(z)
		x = x * m if self.exclude_imp else x
		recon = recon * m if self.exclude_imp else recon

		if not self.exclude_imp:
			return get_mse_loss(recon, x)
		else:
			return get_mse_loss_masked(recon, x, m)

	def _switch_device(self, a, b):
		cpu_device = torch.device("cpu")
		gpu_device = self.device_type
		a = a.to(cpu_device)
		b = b.to(gpu_device)
		return a, b

	def fit(self, trainset, validset=None):
		self.init_layers()
		model = self.to(self.device_type)
		print(model)
		optimizer = get_optimizer(self.opti_name)(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
		print(optimizer)
		batch_num = int(trainset.num_samples / self.batch_size) if self.batch_size != 0 else 1
		batch_val = int(validset.num_samples / self.batch_size) if self.batch_size != 0 else 1

		t = trange(self.max_epochs + 1, desc='Training...')
		for epoch in t:
			self.batch_flag = False
			model.train()
			if self.batch_size != 0:
				# BATCH-WISE TRAINING PHASE
				self.global_train_loss, self.global_valid_loss = 0.0, 0.0
				for b in range(batch_num):
					i, j = (self.batch_size * b) % trainset.num_samples, (self.batch_size * (b+1)) % trainset.num_samples
					model.train()
					loss = model(trainset.X[i:j,:], trainset.m[i:j,:], trainset.coo)
					assert torch.isnan(loss).sum().sum() != 1
					self.global_train_loss += loss.item() * self.batch_size
					optimizer.zero_grad()
					loss += self._l1_norm(model)
					loss.backward()
					optimizer.step()
				self.batch_flag = False
				lb = trainset.num_samples % self.batch_size
				loss = model(trainset.X[:-b,:], trainset.m[:-b,:], None)
				loss += self._l1_norm(model)
				loss.backward()
				optimizer.step()
				assert torch.isnan(loss).sum().sum() != 1
				self.global_train_loss += loss.item() * lb
			else:
				# FULL_BATCH TRAINING PHASE
				loss = model(trainset.X, trainset.m, None)
				assert torch.isnan(loss).sum().sum() != 1
				self.global_train_loss = loss.item()
				optimizer.zero_grad()
				loss += self._l1_norm(model)
				loss.backward()
				optimizer.step()
			self.batch_flag = False
			if validset is not None:
				with torch.no_grad():
					model.eval()
					if self.batch_size != 0:
						# BATCH-WISE VALIDATION PHASE
						for b in range(batch_val):
							i, j = (self.batch_size * b) % validset.num_samples, (self.batch_size * (b+1)) % validset.num_samples
							vloss = model(validset.X[i:j,:], validset.m[i:j,:], trainset.coo)
							assert torch.isnan(vloss).sum().sum() != 1
							self.global_valid_loss += vloss.item() * self.batch_size
						self.batch_flag = False
						lb = validset.num_samples % self.batch_size
						vloss = model(validset.X[-lb:,:], validset.m[-lb:,:], None)
						assert torch.isnan(vloss).sum().sum() != 1
						self.global_valid_loss += vloss.item() * lb
					else:
						# FULL_BATCH VALIDATION PHASE
						vloss = model(validset.X, validset.m, None)
						assert torch.isnan(vloss).sum().sum() != 1
						self.global_valid_loss = vloss.item()
					if self.batch_size != 0:
						self.global_train_loss /= trainset.num_samples
						self.global_valid_loss /= validset.num_samples

					# SAVE BEST MODEL
					SAVE_PATH = '{}best_model'.format(self.save_path)
					if self.save_mode and (self.global_valid_loss < self.best_valid_loss):
						self.best_valid_loss = float(self.global_valid_loss)
						torch.save({'epoch': epoch,
									'model_state_dict': model.state_dict(),
									'optimizer_state_dict': optimizer.state_dict()}, SAVE_PATH)
						self.write_best_loss()
						self.best_valid_flag = True
			t.set_description('(Training: %g)' % float(math.sqrt(self.global_train_loss)) + '(Validation: %g)' % float(math.sqrt(self.global_valid_loss)))
		return model

	def predict(self, dataset, model):
		self.batch_flag = False
		loss = 0.0
		batch_val = int(dataset.num_samples / self.batch_size) if self.batch_size != 0 else dataset.num_samples
		model = self.to(self.device) if model is None else model
		with torch.no_grad():
			model.eval()
			if self.batch_size != 0:
				for b in range(batch_val):
					i, j = (self.batch_size * b) % dataset.num_samples, (self.batch_size * (b+1)) % dataset.num_samples
					model.eval()
					vloss = model(dataset.X[i:j,:], dataset.m[i:j,:], dataset.coo)
					loss += vloss.item() * self.batch_size
				self.batch_flag = False
				lb = dataset.num_samples % self.batch_size
				vloss = model(dataset.X[-lb:,:], dataset.m[-lb:,:], None)
				loss += vloss.item() * lb
			else:
				vloss = model(dataset.X, dataset.m, None)
				loss = vloss.item()
		if self.batch_size != 0:
			return loss / dataset.num_samples
		else:
			return loss

	def fit_predict(self, trainset, validset, testset):
		print("--------TRAINING--------")
		model = self.fit(trainset, validset)
		self.LOGGER.info('Best Loss Updated: {}'.format(self.best_valid_flag))
		if self.save_mode:
			SAVE_PATH = self.save_path + 'final_model'
			self.LOGGER.info('Saving Model....')
			torch.save({'model_state_dict': model.state_dict()}, SAVE_PATH)
		print("---------TESTING---------")
		return self.predict(testset, model)

class VAE(AE):
	def __init__(self, config, logger, num_features=20531):
		super().__init__(config, logger, num_features)
		self.encode = nn.Sequential(
			nn.Linear(self.num_features, config.hidden_nodes),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)
		self.encode_mu = nn.Sequential(
			nn.Linear(config.hidden_nodes, 128),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)
		self.encode_si = nn.Sequential(
			nn.Linear(config.hidden_nodes, 128),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)
		self.decode = nn.Sequential(
			nn.Linear(128, config.hidden_nodes),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, num_features)
		)

	def dimension_reduction(self, x, coo):
		if coo is None:
			h = self.encode(x)
			mu = self.encode_mu(h)
			return mu
		else:
			x = self._topological_conv(x, coo)
			h = self.encode(x)
			mu = self.encode_mu(h)
			return mu

	def _reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		assert not torch.isnan(std).any() and not torch.isnan(eps).any()
		return eps.mul(std).add_(mu)

	def forward(self, x, m=None, coo=None):
		h = self.encode(x)
		mu = self.encode_mu(h)
		logvar = self.encode_si(h)
		z = self._reparameterize(mu, logvar)
		recon = self.decode(mu)
		x = x * m if self.exclude_imp else x
		recon = recon if self.exclude_imp else recon

		if not self.exclude_imp:
			return get_mse_kld_loss(recon, x, mu, logvar)
		else:
			return get_mse_kld_loss_masked(recon, x, mu, logvar, m)

class DAE(AE):
	def __init__(self, config, logger, num_features=20531):
		super().__init__(config, logger, num_features)
		self.encode = nn.Sequential(
			nn.Linear(self.num_features, config.hidden_nodes),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, 128),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)
		self.decode = nn.Sequential(
			nn.Linear(128, config.hidden_nodes),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, self.num_features)
		)

	def init_layers(self):
		try:
			nn.init.xavier_normal_(self.encode[0].weight.data)
			nn.init.xavier_normal_(self.decode[3].weight.data)
		except:
			pass
		nn.init.xavier_normal_(self.decode[0].weight.data)

	def dimension_reduction(self, x):
		return self.encode(x)

	def forward(self, x, m=None, coo=None):
		x = torch.randn(x.size()).to(self.device_type) * 0.01 + x
		z = self.encode(xx)
		recon = self.decode(z)
		x = x * m if self.exclude_imp else x
		recon = recon * m if self.exclude_imp else recon

		if not self.exclude_imp:
			return get_mse_loss(recon, x)
		else:
			return get_mse_loss_masked(recon, x, m)






