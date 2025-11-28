import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob
import random
import numpy as np
import torch
from tqdm import tqdm

from tdmpc2.tdmpc2.common.buffer import Buffer
from tdmpc2.tdmpc2.trainer.base import Trainer


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
	
	#TODO change this eval per environmentType argument
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for _ in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(), False, 0, 0
				while not done:
					if torch.cuda.is_available() and self.device.type == 'cuda':
						torch.compiler.cudagraph_mark_step_begin()
					frame=self.env.render() 
					inputs=self.agent.vit_processor(images=frame,return_tensors='pt')
					outputs = self.agent.vit_model(**inputs)
					obs  = outputs.last_hidden_state

					ob=self.agent.img_projection_TDMPC2(obs)
					ob = ob.squeeze(0)
					ob=ob.mean(dim=0)
					
					action = self.agent.act(ob, t0=t==0, eval_mode=True, task=task_idx)

					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),})
		return results
	
	def _load_dataset(self,env2Train):
		"""Load dataset for offline training."""
		fps = list(self.cfg.data_dir.glob(f'*_{env2Train}.pt'))  # glob inside the directory
		assert len(fps) > 0, f'No data found at {self.cfg.data_dir}'
		print(f'Found {len(fps)} files in {self.cfg.data_dir}')
	
		# Create buffer for sampling
		_cfg = deepcopy(self.cfg)
		_cfg.steps = _cfg.buffer_size
		self.buffer = Buffer(_cfg)
		max_files = 50
		if len(fps) > max_files:
			fps = random.sample(fps, max_files)
		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp, weights_only=False)
			assert td.shape[0] == _cfg.episode_lengths[env2Train], \
				f'Expected episode length {td.shape[0]} to match config episode length {_cfg.episode_lengths[env2Train]}, ' \
				f'please double-check your config.'
			self.buffer.load(td)
			self.buffer.load(td)

		# expected_episodes = _cfg.buffer_size // _cfg.episode_length
		# if self.buffer.num_eps != expected_episodes:
		# 	print(f'WARNING: buffer has {self.buffer.num_eps} episodes, expected {expected_episodes} episodes for {self.cfg.task} task set.')

	def train(self,env2Train):
		"""Train a TD-MPC2 agent."""
		#assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
		self._load_dataset(env2Train)

		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in range(self.cfg.training_steps):

			# Update agent
			train_metrics = self.agent.update(self.buffer)

			# Evaluate agent periodically
			if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
				metrics = {
					'iteration': i,
					'elapsed_time': time() - self._start_time,
				}
				metrics.update(train_metrics)
				if i % self.cfg.eval_freq == 0:
					# metrics.update(self.eval())
					# self.logger.pprint_multitask(metrics, self.cfg)
					if i > 0:
						self.logger.save_agent(self.agent, identifier=f'{i}')
						print("saved")	
				self.logger.log(metrics, 'pretrain')
		self.logger.finish(self.agent)