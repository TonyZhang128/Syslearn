import argparse
import os
import torch
from zyn import utils

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--name', type=str, default='MNIST_Classification', help='name of the experiment')
		self.parser.add_argument('--dataset', type=str, default='MNIST', help='name of dataset')
		self.parser.add_argument('--dataset_path', type=str, default='/data/zyn/MNIST/processed', help='dataset pathdir')
		self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
		self.parser.add_argument('--nThreads', default=16, type=int, help='# workers for loading data')
		self.parser.add_argument('--seed', default=0, type=int, help='random seed')
		self.parser.add_argument('--num_epoch', default=20, type=int, help='epochs for running') 
		self.parser.add_argument('--transform', default=True, type=bool, help='whether using transform to input')

		self.initialized = True
	
	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.mode = self.mode

		# GPU ids
		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])
		
		# ckpt
		if self.opt.ckpt:
			self.opt.ckpt = os.path.join(self.opt.ckpt, self.opt.name)
			from zyn.utils import mkdirs
			mkdirs(self.opt.ckpt)


		# print args
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		utils.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		return self.opt
