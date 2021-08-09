import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class Learner(nn.Module):
	"""
	This is a learner class, which will accept a specific network module, such as OmniNet that define the network forward
	process. Learner class will create two same network, one as theta network and the other acts as theta_pi network.
	for each episode, the theta_pi network will copy its initial parameters from theta network and update several steps
	by meta-train set and then calculate its loss on meta-test set. All loss on meta-test set will be sumed together and
	then backprop on theta network, which should be done on metalaerner class.
	For learner class, it will be responsible for update for several steps on meta-train set and return with the loss on
	meta-test set.
	"""

	def __init__(self, net_cls, *args):
		"""
		It will receive a class: net_cls and its parameters: args for net_cls.
		:param net_cls: class, not instance
		:param args: the parameters for net_cls
		"""
		super(Learner, self).__init__()
		# pls make sure net_cls is a class but NOT an instance of class.
		assert net_cls.__class__ == type

		# we will create two class instance meanwhile and use one as theta network and the other as theta_pi network. 一个是元网络一个是子网络
		self.net = net_cls(*args)
		# you must call create_pi_net to create pi network additionally
		self.net_pi = net_cls(*args)
		# update theta_pi = theta_pi - lr * grad    更新公式
		# according to the paper, here we use naive version of SGD to update theta_pi
		# 0.1 here means the learner_lr


		#学习率=0。1，这里采用随机梯度
		self.optimizer = optim.SGD(self.net_pi.parameters(), 0.1)   #基础优化器

	def parameters(self):
		"""
		Override this function to return only net parameters for MetaLearner's optimize
		it will ignore theta_pi network parameters.
		:return:
		"""
		return self.net.parameters()

	def update_pi(self):
		"""
		copy parameters from self.net -> self.net_pi
		:return:
		"""
		for m_from, m_to in zip(self.net.modules(), self.net_pi.modules()):
			if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
				m_to.weight.data = m_from.weight.data.clone()
				if m_to.bias is not None:
					m_to.bias.data = m_from.bias.data.clone()

	def forward(self, support_x, support_y, query_x, query_y, num_updates):
		"""
		learn on current episode meta-train: support_x & support_y and then calculate loss on meta-test set: query_x&y
		元训练集（support_x 和support_y）   元测试集（query_x和query_y）
		:param support_x: [setsz, c_, h, w]
		:param support_y: [setsz]
		:param query_x:   [querysz, c_, h, w]
		:param query_y:   [querysz]
		:param num_updates: 5
		:return:
		"""
		# now try to fine-tune from current $theta$ parameters -> $theta_pi$
		# after num_updates of fine-tune, we will get a good theta_pi parameters so that it will retain satisfying
		# performance on specific task, that's, current episode.
		# firstly, copy theta_pi from theta network
		self.update_pi()

		# update for several steps
		for i in range(num_updates):
			# forward and backward to update net_pi grad.
			loss, pred = self.net_pi(support_x, support_y)
			self.optimizer.zero_grad()#调用基础优化器
			loss.backward()#产生梯度
			self.optimizer.step()#调用基础优化器梯度下降

		# Compute the meta gradient and return it, the gradient is from one episode
		# in metalearner, it will merge all loss from different episode and sum over it.
		loss, pred = self.net_pi(query_x, query_y)
		# pred: [setsz, n_way], indices: [setsz]
		_, indices = torch.max(pred, dim=1)
		correct = torch.eq(indices, query_y).sum().item()
		acc = correct / query_y.size(0)

		# gradient for validation on theta_pi
		# after call autorad.grad, you can not call backward again except for setting create_graph = True
		# as we will use the loss as dummpy loss to conduct a dummy backprop to write our gradients to theta network,
		# here we set create_graph to true to support second time backward.
		grads_pi = autograd.grad(loss, self.net_pi.parameters(), create_graph=True)#计算损失相对于网络权重参数的梯度总和

		return loss, grads_pi, acc

	def net_forward(self, support_x, support_y):
		"""
		This function is purely for updating net network. In metalearner, we need the get the loss op from net network
		to write our merged gradients into net network, hence will call this function to get a dummy loss op.
		:param support_x: [setsz, c, h, w]
		:param support_y: [sessz, c, h, w]
		:return: dummy loss and dummy pred
		"""
		loss, pred = self.net(support_x, support_y)
		return loss, pred


class MetaLearner(nn.Module):
	"""
	As we have mentioned in Learner class, the metalearner class will receive a series of loss on different tasks/episodes
	on theta_pi network, and it will merage all loss and then sum over it. The summed loss will be backproped on theta
	network to update theta parameters, which is the initialization point we want to find.
	"""

	def __init__(self, net_cls, net_cls_args, n_way, k_shot, meta_batchsz, beta, num_updates):
		"""

		:param net_cls: class, not instance. the class of specific Network for learner
		:param net_cls_args: tuple, args for net_cls, like (n_way, imgsz)
		:param n_way:
		:param k_shot:
		:param meta_batchsz: number of tasks/episode
		:param beta: learning rate for meta-learner
		:param num_updates: number of updates for learner
		"""
		super(MetaLearner, self).__init__()

		self.n_way = n_way
		self.k_shot = k_shot
		self.meta_batchsz = meta_batchsz
		self.beta = beta
		# self.alpha = alpha # set alpha in Learner.optimizer directly.
		self.num_updates = num_updates

		# it will contains a learner class to learn on episodes and gather the loss together.
		self.learner = Learner(net_cls, *net_cls_args)
		# the optimizer is to update theta parameters, not theta_pi parameters.


		self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)#元优化器（参数和学习率）

	def write_grads(self, dummy_loss, sum_grads_pi):
		"""
		write loss into learner.net, gradients come from sum_grads_pi.
		Since the gradients info is not calculated by general backward, we need this function to write the right gradients
		into theta network and update theta parameters as wished.
		:param dummy_loss: dummy loss, nothing but to write our gradients by hook
		:param sum_grads_pi: the summed gradients
		:return:
		"""

		# Register a hook on each parameter in the net that replaces the current dummy grad
		# with our grads accumulated across the meta-batch 经过整个元批次的梯度进行积累
		hooks = []

		for i, v in enumerate(self.learner.parameters()):
			print("i,v:",i,v)
			def closure():
				ii = i
				print("sum_grads_pi:",sum_grads_pi[ii])
				return lambda grad: sum_grads_pi[ii]

			# if you write: hooks.append( v.register_hook(lambda grad : sum_grads_pi[i]) )
			# it will pop an ERROR, i don't know why?
			hooks.append(v.register_hook(closure()))#利用钩子将总梯度写更改网络权重（初始化参数）

			print(hooks)
		# use our sumed gradients_pi to update the theta/net network,
		# since our optimizer receive the self.net.parameters() only.
		self.optimizer.zero_grad()#首先设置梯度为零
		dummy_loss.backward()#前馈产生梯度
		self.optimizer.step()#梯度下降

		# if you do NOT remove the hook, the GPU memory will expode!!!
		for h in hooks:
			h.remove()

	def forward(self, support_x, support_y, query_x, query_y):#钩子函数
		"""
		Here we receive a series of episode, each episode will be learned by learner and get a loss on parameters theta.
		we gather the loss and sum all the loss and then update theta network.
		setsz = n_way * k_shotf
		querysz = n_way * k_shot
		:param support_x: [meta_batchsz, setsz, c_, h, w]
		:param support_y: [meta_batchsz, setsz]
		:param query_x:   [meta_batchsz, querysz, c_, h, w]
		:param query_y:   [meta_batchsz, querysz]
		:return:
		"""
		sum_grads_pi = None
		meta_batchsz = support_y.size(0)

		# support_x[i]: [setsz, c_, h, w]
		# we do different learning task sequentially, not parallel.我们按顺序做任务，而不是平行的
		accs = []
		# for each task/episode.
		for i in range(meta_batchsz):
			_, grad_pi, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
			accs.append(episode_acc)
			if  sum_grads_pi is None:
				sum_grads_pi = grad_pi
			else:  # accumulate all gradients from different episode learner
				sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi)]#是一个多维的张量
		print("32个任务训练已结束")
		# As we already have the grads to update
		# We use a dummy forward / backward pass to get the correct grads into self.net
		# the right grads will be updated by hook, ignoring backward.
		# use hook mechnism to write sumed gradient into network.
		# we need to update the theta/net network, we need a op from net network, so we call self.learner.net_forward
		# to get the op from net network, since the loss from self.learner.forward will return loss from net_pi network.
		dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
		self.write_grads(dummy_loss, sum_grads_pi)

		return accs

	def pred(self, support_x, support_y, query_x, query_y):
		"""
		predict for query_x
		:param support_x:
		:param support_y:
		:param query_x:
		:param query_y:
		:return:
		"""
		meta_batchsz = support_y.size(0)

		accs = []
		# for each task/episode.
		# the learner will copy parameters from current theta network and then fine-tune on support set.
		for i in range(meta_batchsz):
			_, _, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
			accs.append(episode_acc)

		return np.array(accs).mean()
