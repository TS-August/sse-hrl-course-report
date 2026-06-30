import threading
import torch
import os.path as osp
import numpy as np


def sample_her_transitions(buffer, reward_func, batch_size, future_step, future_p=1.0):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['mask'][:,0].copy().squeeze()
    
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon[ep_idxes])
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 
    
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    not_her_indexes = np.delete(np.arange(batch_size), her_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon[ep_idxes] - t_samples, future_step)).astype(int)
    
    future_t = np.minimum((t_samples + 1 + future_offset), buffer['a'].shape[1]-1)
    
    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    
    batch['offset'] = future_offset.copy()
    
    batch['r'] = reward_func(achieved_goal=batch['ag2'], goal=batch['bg'], info=None, ob=batch['o2'], ob_old=batch['ob'])
    
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_her_transitions_grid(buffer, reward_func, batch_size, future_step, future_p = 1.0, movement_pen = 1.0, movement_threshold = 0.5, future_offset_threshold = 30.):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 
    
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    not_her_indexes = np.delete(np.arange(batch_size), her_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    
    future_t = (t_samples + 1 + future_offset)
    
    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    
    batch['offset'] = future_offset.copy()
    
    batch['r'] = reward_func(achieved_goal=batch['ag2'], goal=batch['bg'], info=None, ob=batch['o2'], ob_old=batch['ob'])
    
    dist = batch['ag'][not_her_indexes] - buffer['ag'][ep_idxes[not_her_indexes], future_t[not_her_indexes]]
    dist2 = batch['bg'][not_her_indexes] - buffer['ag'][ep_idxes[not_her_indexes], future_t[not_her_indexes]]
    future_offset_test = future_t[not_her_indexes]
    
    movement_failure = not_her_indexes[0][np.where((dist < movement_threshold) & (dist2 > movement_threshold) & (future_offset_test > future_offset_threshold))]
    batch['r'][movement_failure] -= movement_pen
    
    
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch


def sample_her_transitions_with_subgoaltesting(buffer, reward_func, batch_size, graphplanner, future_step, cutoff, subgoaltest_p, subgoaltest_threshold, monitor, gradual_pen):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['mask'][:,0].copy().squeeze()
    
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon[ep_idxes])
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    original_batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    subgoaltesting_indexes = np.where(np.random.uniform(size=batch_size) < subgoaltest_p) 
    not_subgoaltesting_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon[ep_idxes] - t_samples, future_step)).astype(int)
    future_t = np.minimum((t_samples + 1 + future_offset), buffer['a'].shape[1]-1)

    batch['origin_bg'] = batch['bg'].copy()
    batch['origin_a'] = batch['a'].copy()

    batch['bg'] = buffer['ag'][ep_idxes, future_t]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(achieved_goal=batch['ag2'], goal=batch['bg'], info=None, ob=batch['o2'], ob_old=batch['ob'])
    dist = batch['a'][subgoaltesting_indexes] - batch['ag2'][subgoaltesting_indexes]
    batch['a'][not_subgoaltesting_indexes] = batch['ag2'][not_subgoaltesting_indexes]
      
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_transitions(buffer, batch_size):
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['mask'][:,0].copy().squeeze()
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon[ep_idxes])
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    return batch

def sample_transitions_high(buffer, batch_size):
    n_trajs = buffer['a'].shape[0]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    batch = {key: buffer[key][ep_idxes].copy() for key in buffer.keys()}
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    return batch

def sample_transitions_high_uniform(buffer0, buffer1, batch_size):
    unique_returns0 = np.unique(buffer0['normal_return'])
    unique_returns1 = np.unique(buffer1['normal_return']) if buffer1 else np.array([])  
    unique_returns = np.concatenate((unique_returns0, unique_returns1))
    
    if len(unique_returns) == 0:  
        raise ValueError("No unique rewards found in buffers!")  
    
    unique_returns = np.sort(np.unique(unique_returns))
    return_weights = np.ones_like(unique_returns)/len(unique_returns)
    num_per_returns = (batch_size * return_weights).astype(int)

    num_per_returns[-1] += batch_size - np.sum(num_per_returns)  

    selected_indices = {0: [], 1: []} 

    for buffer, buffer_id in zip([buffer0, buffer1], [0, 1]):
        if not buffer:  
            continue

        horizons = buffer['length'][:, 0] 

        r_flat = np.repeat(buffer['normal_return'], buffer['r'].shape[1], axis=1).flatten()
        mask = np.zeros_like(r_flat, dtype=bool)

        episode_size = buffer['r'].shape[1]
        valid_indices = np.arange(len(r_flat))
        for i, h in enumerate(horizons):
            mask[i * episode_size : i * episode_size + int(h)] = True  

        r_valid = r_flat[mask] 
        valid_indices = valid_indices[mask]  

        for uniq_return, num_samples in zip(unique_returns, num_per_returns):
            return_indices = valid_indices[r_valid == uniq_return]
            if len(return_indices) > 0:
                sampled_indices = np.random.choice(return_indices, size=num_samples, replace=True)
                selected_indices[buffer_id].extend(sampled_indices)
    remaining = batch_size - len(selected_indices[0]) - len(selected_indices[1])
    if remaining > 0 and buffer1:  
        extra_indices = np.random.choice(len(buffer1['r'].flatten()), size=remaining, replace=True)
        selected_indices[1].extend(extra_indices)
    elif remaining > 0:  
        extra_indices = np.random.choice(len(buffer0['r'].flatten()), size=remaining, replace=True)
        selected_indices[0].extend(extra_indices)
    if buffer0:
        buffer0['normal_return'] = np.repeat(buffer0['normal_return'], buffer0['r'].shape[1], axis=1).reshape(buffer0['r'].shape)
    if buffer1:
        buffer1['normal_return'] = np.repeat(buffer1['normal_return'], buffer1['r'].shape[1], axis=1).reshape(buffer1['r'].shape)
    batch = {key: np.concatenate(
                [buffer0[key].reshape(-1, buffer0[key].shape[-1])[selected_indices[0]]] + 
                ([buffer1[key].reshape(-1, buffer1[key].shape[-1])[selected_indices[1]]] if buffer1 else []), 
                axis=0)
             for key in ['ob','o2','ag','ag2','r','bg','a','mask', 'normal_return']}
    return batch






class LowReplay:
    def __init__(self, env_params, args, low_reward_func, agent=None, name='low_replay'):
        self.env_params = env_params
        self.args = args
        self.low_reward_func = low_reward_func
        self.agent = agent
        self.horizon = env_params['max_timesteps']
        self.size = args.buffer_size // self.horizon
        
        self.current_size = 0
        self.n_transitions_stored = 0
        
        self.buffers = dict(ob=np.zeros((self.size, self.horizon, self.env_params['obs'])),
                            o2=np.zeros((self.size, self.horizon, self.env_params['obs'])),
                            ag=np.zeros((self.size, self.horizon, self.env_params['sub_goal'])),
                            ag2=np.zeros((self.size, self.horizon, self.env_params['sub_goal'])),
                            bg=np.zeros((self.size, self.horizon, self.env_params['sub_goal'])),
                            a=np.zeros((self.size, self.horizon, self.env_params['l_action_dim'])), 
                            mask=np.zeros((self.size, self.horizon, 1))
                            )
        
        self.lock = threading.Lock()
        self._save_file = str(name) + '.pt'
    
    def store(self, episodes):
        ob_list, ag_list, bg_list, a_list, mask_list = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a'], episodes['mask']
        batch_size = ob_list.shape[0]
        episode_length = ob_list.shape[1] - 1
        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers['ob'][idxs][:episode_length] = ob_list[:,:-1].copy()
            self.buffers['o2'][idxs][:episode_length] = ob_list[:,1:].copy()
            self.buffers['ag'][idxs][:episode_length] = ag_list[:,:-1].copy()
            self.buffers['ag2'][idxs][:episode_length] = ag_list[:,1:].copy()
            self.buffers['bg'][idxs][:episode_length] = bg_list.copy()
            self.buffers['a'][idxs][:episode_length] = a_list.copy()
            self.buffers['mask'][idxs][:episode_length] = episode_length
            self.n_transitions_stored += self.horizon * batch_size
    
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions(temp_buffers, self.low_reward_func, batch_size,
                                             future_step=self.args.low_future_step,
                                             future_p=self.args.low_future_p)
        return transitions

    def sample_g(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions(temp_buffers, self.low_reward_func, batch_size,
                                             future_step=self.args.low_future_step,
                                             future_p=self.args.low_future_p_g)
        return transitions
    
    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx
    
    def get_all_data(self):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers
    
    def sample_regular_batch(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_transitions(temp_buffers, batch_size)
        return transitions
    
    def sample_search_batch(self, ag):
        distances = np.sum((self.buffers['ag'] - ag) ** 2, axis=-1)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return self.buffers['ob'][min_idx]
    
    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )
    
    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']
    
    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)




class HighReplay:
    def __init__(self, env_params, args, high_reward_func, monitor, agent=None, name='high_replay'):
        self.env_params = env_params
        self.args = args
        self.high_reward_func = high_reward_func
        self.monitor = monitor
        self.horizon = env_params['max_timesteps']
        self.size0 = args.buffer_size // self.horizon // 4
        self.size1 = args.buffer_size // self.horizon // 2
        self.agent = agent
        
        self.current_size0 = 0
        self.FIFO_window = 0
        self.current_size1 = 0
        self.n_transitions_stored = 0
        
        self.buffers0 = dict(ob=np.zeros((self.size0, self.horizon, self.env_params['obs']+1)),
                            o2=np.zeros((self.size0, self.horizon, self.env_params['obs']+1)),
                            ag=np.zeros((self.size0, self.horizon, self.env_params['sub_goal'])),
                            ag2=np.zeros((self.size0, self.horizon, self.env_params['sub_goal'])),
                            bg=np.zeros((self.size0, self.horizon, self.env_params['goal'])),
                            a=np.zeros((self.size0, self.horizon, self.env_params['h_action_dim'])),
                            r=np.zeros((self.size0, self.horizon, 1)),
                            mask=np.zeros((self.size0, self.horizon, 1)), 
                            gamma_return = np.zeros((self.size0, 1)), 
                            normal_return = np.zeros((self.size0, 1)),
                            length = np.zeros((self.size0, 1))
                            )
        self.buffers1 = dict(ob=np.zeros((self.size1, self.horizon, self.env_params['obs']+1)),
                            o2=np.zeros((self.size1, self.horizon, self.env_params['obs']+1)),
                            ag=np.zeros((self.size1, self.horizon, self.env_params['sub_goal'])),
                            ag2=np.zeros((self.size1, self.horizon, self.env_params['sub_goal'])),
                            bg=np.zeros((self.size1, self.horizon, self.env_params['goal'])),
                            a=np.zeros((self.size1, self.horizon, self.env_params['h_action_dim'])),
                            r=np.zeros((self.size1, self.horizon, 1)),
                            mask=np.zeros((self.size1, self.horizon, 1)), 
                            gamma_return = np.zeros((self.size1,1)), 
                            normal_return = np.zeros((self.size1, 1)),
                            length = np.zeros((self.size1, 1))
                            )
        
        self.buffers_for_adahind = dict(wp=np.zeros((self.size1//self.horizon, self.horizon, self.env_params['obs'])))

        self.lock = threading.Lock()
        self._save_file = str(name) + '.pt'


    def store0(self, episodes):
        ob_list, ag_list, bg_list, a_list, r_list, mask_list, o2_list, ag2_list  = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a'], episodes['r'], episodes['mask'], episodes['o2'], episodes['ag2']
        batch_size = ob_list.shape[0]
        episode_length = ob_list.shape[1]
        gamma_return = sum(r * (self.args.gamma_high ** i) for i, r in enumerate(r_list[0]))
        normal_return = sum(r for i, r in  enumerate(r_list[0]))
        with self.lock:
            idxs = self._get_storage_idx0(batch_size=batch_size)
            self.buffers0['ob'][idxs] = np.zeros((self.horizon, self.env_params['obs']+1))
            self.buffers0['ob'][idxs][:episode_length] = ob_list.copy()
            self.buffers0['o2'][idxs] = np.zeros((self.horizon, self.env_params['obs']+1))
            self.buffers0['o2'][idxs][:episode_length] = o2_list.copy()
            self.buffers0['ag'][idxs] = np.zeros((self.horizon, self.env_params['sub_goal']))
            self.buffers0['ag'][idxs][:episode_length] = ag_list.copy()
            self.buffers0['ag2'][idxs] = np.zeros((self.horizon, self.env_params['sub_goal']))
            self.buffers0['ag2'][idxs][:episode_length] = ag2_list.copy()
            self.buffers0['bg'][idxs] = np.zeros((self.horizon, self.env_params['goal']))
            self.buffers0['bg'][idxs][:episode_length] = bg_list.copy()
            self.buffers0['a'][idxs] = np.zeros((self.horizon, self.env_params['h_action_dim']))
            self.buffers0['a'][idxs][:episode_length] = a_list.copy()
            self.buffers0['r'][idxs] = np.zeros((self.horizon, 1))
            self.buffers0['r'][idxs][:episode_length] = r_list.copy()
            self.buffers0['mask'][idxs] = np.zeros((self.horizon, 1))
            self.buffers0['mask'][idxs][:episode_length] = mask_list.copy()
            self.buffers0['gamma_return'][idxs] = gamma_return
            self.buffers0['normal_return'][idxs] = normal_return
            self.buffers0['length'][idxs] = episode_length
            self.n_transitions_stored += episode_length
    def store1(self, episodes):
        ob_list, ag_list, bg_list, a_list, r_list, mask_list, o2_list, ag2_list = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a'], episodes['r'], episodes['mask'], episodes['o2'], episodes['ag2']
        batch_size = ob_list.shape[0]
        episode_length = ob_list.shape[1]
        gamma_return = sum(r * (self.args.gamma_high ** i) for i, r in enumerate(r_list[0]))
        normal_return = sum(r for i, r in  enumerate(r_list[0]))
        with self.lock:
            idxs = self._get_storage_as_return(batch_size=batch_size)
            self.buffers1['ob'][idxs] = np.zeros((self.horizon, self.env_params['obs']+1))
            self.buffers1['ob'][idxs][:episode_length] = ob_list.copy()
            self.buffers1['o2'][idxs] = np.zeros((self.horizon, self.env_params['obs']+1))
            self.buffers1['o2'][idxs][:episode_length] = o2_list.copy()
            self.buffers1['ag'][idxs] = np.zeros((self.horizon, self.env_params['sub_goal']))
            self.buffers1['ag'][idxs][:episode_length] = ag_list.copy()
            self.buffers1['ag2'][idxs] = np.zeros((self.horizon, self.env_params['sub_goal']))
            self.buffers1['ag2'][idxs][:episode_length] = ag2_list.copy()
            self.buffers1['bg'][idxs] = np.zeros((self.horizon, self.env_params['goal']))
            self.buffers1['bg'][idxs][:episode_length] = bg_list.copy()
            self.buffers1['a'][idxs] = np.zeros((self.horizon, self.env_params['h_action_dim']))
            self.buffers1['a'][idxs][:episode_length] = a_list.copy()
            self.buffers1['r'][idxs] = np.zeros((self.horizon, 1))
            self.buffers1['r'][idxs][:episode_length] = r_list.copy()
            self.buffers1['mask'][idxs] = np.zeros((self.horizon, 1))
            self.buffers1['mask'][idxs][:episode_length] = mask_list.copy()
            self.buffers1['gamma_return'][idxs] = gamma_return
            self.buffers1['normal_return'][idxs] = normal_return
            self.buffers1['length'][idxs] = episode_length
            self.n_transitions_stored += episode_length

    def _get_storage_as_return(self, batch_size):
        if self.current_size1 + batch_size <= self.size1:
            idx = np.arange(self.current_size1, self.current_size1 + batch_size)[0]
        else:
            normal_returns = self.buffers1["normal_return"].flatten()
            min_normal_indices = np.where(normal_returns == np.min(normal_returns))[0]
        
            if len(min_normal_indices) > 1:  
                gamma_returns = self.buffers1["gamma_return"].flatten()
                idx = min_normal_indices[np.argmin(gamma_returns[min_normal_indices])]
            else:
                idx = min_normal_indices[0]

        self.current_size1 = min(self.size1, self.current_size1 + batch_size)
        return idx

    def sample(self, batch_size, graphplanner):
        temp_buffers0 = {}
        temp_buffers1 = {}
        with self.lock:
            for key in self.buffers0.keys():
                if self.current_size0 > 0:
                    temp_buffers0[key] = self.buffers0[key][:self.current_size0]
                if self.current_size1 > 0:
                    temp_buffers1[key] = self.buffers1[key][:self.current_size1]
        transitions = sample_transitions_high_uniform(temp_buffers0, temp_buffers1, batch_size)
        return transitions
    
    def _get_storage_idx0(self, batch_size):
        if self.current_size0 + batch_size <= self.size0:
            idx = np.arange(self.current_size0, self.current_size0 + batch_size)
        else:
            idx = np.arange(self.FIFO_window, self.FIFO_window + batch_size) % self.size0
        self.current_size0 = min(self.size0, self.current_size0 + batch_size)
        self.FIFO_window = (self.FIFO_window + batch_size)%self.size0
        if batch_size == 1:
            idx = idx[0]
        return idx
    
    def sample_regular_batch(self, batch_size):
        temp_buffers0 = {}
        temp_buffers1 = {}
        with self.lock:
            for key in self.buffers0.keys():
                if self.current_size0 > 0:
                    temp_buffers0[key] = self.buffers0[key][:self.current_size0]
                if self.current_size1 > 0:
                    temp_buffers1[key] = self.buffers1[key][:self.current_size1]
        transitions = sample_transitions_high_uniform(temp_buffers0, temp_buffers1, batch_size)
        return transitions
    
    def state_dict(self):
        return dict(
            current_size0=self.current_size0,
            current_size1=self.current_size1,
            n_transitions_stored=self.n_transitions_stored,
            buffers0=self.buffers0,
            buffers1=self.buffers1
        )
    
    def load_state_dict(self, state_dict):
        self.current_size0 = state_dict['current_size0']
        self.current_size1 = state_dict['current_size1']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers0 = state_dict['buffers0']
        self.buffers1 = state_dict['buffers1']
    
    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
