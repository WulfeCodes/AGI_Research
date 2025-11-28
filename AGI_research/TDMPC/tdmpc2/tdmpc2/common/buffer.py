import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage,RandomSampler
from torchrl.data.replay_buffers.samplers import SliceSampler
import psutil

class SequentialSampler:
	def __init__(self,sequence_length,batch_size,max_attempts=5000):
		self.sequence_length = sequence_length+1
		self.max_attempts = max_attempts
		self.batch_size = batch_size

	def sample(self,storage):
		total_size = len(storage)
		max_start = total_size - self.sequence_length

		if max_start <=0:
			raise ValueError(f"Not enough data: need {self.sequence_length}, have {total_size}")
		valid_sequences = []
		attempts = 0
        
		while len(valid_sequences) < self.batch_size and attempts < self.max_attempts:
            # Sample a random starting position
			start_idx = torch.randint(0, max_start, ())
			sequence_indices = torch.arange(start_idx, start_idx + self.sequence_length)
            
            # Check if any position except the last has 'terminated' = True
			terminated_values = storage[sequence_indices]['terminated']
            
            # Only the last position can be terminated
			if not terminated_values[:-1].any():  # No early terminations
				valid_sequences.append(sequence_indices)
            
			attempts += 1
        
		if len(valid_sequences) < self.batch_size:
			print(f"Warning: Only found {len(valid_sequences)} valid sequences out of {self.batch_size}")
        # Stack sequences and flatten
		sequences = torch.stack(valid_sequences[:self.batch_size])
		return sequences.flatten(), {}
		


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		if torch.cuda.is_available():
			self._device = torch.device('cuda:0')
		else: 
			self._device = torch.device('cpu')
		self._capacity = cfg.buffer_size
		self._sampler = SequentialSampler(sequence_length=self.cfg.horizon
									,batch_size=self.cfg.batch_size)
			#SliceSampler(
			#num_slices=self.cfg.batch_size,
			#slice_len=self.cfg.horizon,
			#end_key='terminated',
			#traj_key='episode',
			#truncated_key='terminated',
			#strict_length=True,
			#cache_values=cfg.multitask,
		#)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			#sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		if torch.cuda.is_available():
			mem_free, _ = torch.cuda.mem_get_info()
		else:
			mem = psutil.virtual_memory()
			mem_free, _ = mem.available, mem.total

		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # if replaced w 2.5*total_bytes < mem_free
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		#td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task",strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task

	def sample(self):
		# Check all sampler attributes
		"""Sample a batch of subsequences from the buffer."""
		indices, info = self._sampler.sample(self._buffer._storage)
		td=self._buffer._storage[indices]
		td = td.view(-1, self.cfg.horizon+1).permute(1, 0)

		return self._prepare_batch(td)
