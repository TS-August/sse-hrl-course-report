import numpy as np
import sys
import torch
import io
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from PIL import Image
from rl.algo.core import BaseAlgo
import os.path as osp
import os
from matplotlib.lines import Line2D 
import moviepy
import imageio
import wandb
import matplotlib.cm as cm
cmap = plt.cm.viridis
plt.rcParams['font.family'] = 'Times New Roman'
class Algo(BaseAlgo):
    def __init__(
        self,
        env, env_params, args,
        test_env, 
        low_agent, high_agent, low_replay, high_replay, monitor, low_learner, high_learner,
        low_reward_func, high_reward_func,
        name='algo',
    ):
        super().__init__(
            env, env_params, args,
            low_agent, high_agent, low_replay, high_replay, monitor, low_learner, high_learner,
            low_reward_func, high_reward_func,
            name=name,
        )
        self.test_env = test_env

        self.curr_subgoal = None
        self.curr_highpolicy_obs = None

        self.way_to_subgoal = 0
        self.subgoal_freq = args.subgoal_freq
        self.subgoal_scale = np.array(args.subgoal_scale)
        self.subgoal_offset = np.array(args.subgoal_offset)
        self.subgoal_dim = args.subgoal_dim
        self.low_replay = low_replay
        self.high_replay = high_replay
        
        self.waypoint_subgoal = None
        self.env_frames = []
        self.coverage = 0.0
        self.epsilon = 1.0
        self.check_positive_reward = False
        self.prev_coverage = 0
        self.episode=0
        self.epsilon_goal1 = 1/3
        self.epsilon_goal2 = 2/3

        matplotlib.use('Agg')

    def get_actions(self, ob, ag, bg, timestep, a_max=1, random_goal=False, act_randomly=False, graph=False, first = False, do_exp=False, reduce_noise = False):
        high_dict = {}
        if ((self.curr_subgoal is None) or ((self.do_next_high) and not random_goal) or (random_goal and self.way_to_subgoal == 0)):
            self.curr_highpolicy_obs = ob
            
            if random_goal:
                sub_goal = np.random.uniform(low=-1, high=1, size=self.env_params['sub_goal'])
                sub_goal = sub_goal * self.subgoal_scale + self.subgoal_offset
            else:
                if do_exp and self.graphplanner.graph is not None:
                    prob = np.random.rand()
                    if prob < self.epsilon_goal1:
                        if 'DoubleGoal' in self.args.env_name:
                            prob_temp = np.random.random()
                            if prob_temp < 0.5:
                                sub_goal = bg[:3]
                            else:
                                sub_goal = bg[3:]
                        else:
                            sub_goal = bg
                    elif prob < self.epsilon_goal2:
                        spend_time = timestep/self.env_params['max_timesteps']
                        high_ob = np.concatenate((self.curr_highpolicy_obs,[spend_time]))
                        sub_goal = self.high_agent.get_actions(high_ob, bg)
                    else:
                        if self.graphplanner.dim == 3:
                            sub_goal = self.graphplanner.select_novel_goal_3D(ag)
                        else:
                            sub_goal = self.graphplanner.select_novel_goal(ag)
                    if self.graphplanner.dim == 3:
                        column = self.args.map_size/self.args.grid_size
                        depth = self.args.map_size/self.args.grid_size
                        subgoal_s_position = (sub_goal.copy()+self.args.offset)//self.args.grid_size
                        subgoal_index = subgoal_s_position[2]*column*depth + subgoal_s_position[1]*column + subgoal_s_position[0]
                        if do_exp:
                            self.graphplanner.tryClusterPush(subgoal_index)
                    else:
                        subgoal_s_position = (sub_goal.copy()+self.args.offset)//self.args.grid_size
                        subgoal_index = subgoal_s_position[1]*(self.args.map_size/self.args.grid_size) + subgoal_s_position[0]
                        if do_exp:
                            self.graphplanner.tryClusterPush(subgoal_index)
                    
                else: 
                    prob = np.random.rand()
                    high_ob_list, high_ag_list, high_bg_list, high_a_list, high_mask_list, high_r_list = [], [], [], [], [], []
                    high_ob2_list, high_ag2_list = [], []
                    spend_time = timestep/self.env_params['max_timesteps']
                    high_ob = np.concatenate((self.curr_highpolicy_obs,[spend_time]))
                    if prob < self.epsilon:
                        while True:
                            sub_goal = np.random.uniform(low=-1, high=1, size=self.env_params['sub_goal'])
                            sub_goal = sub_goal * self.subgoal_scale + self.subgoal_offset
                            if self.graphplanner.dim == 3:
                                if self.graphplanner.check_reachability_3D(sub_goal):
                                    break
                            else:
                                if self.graphplanner.check_reachability(sub_goal):
                                    break
                            high_ob_list.append(high_ob.copy())
                            high_ob2_list.append(high_ob.copy())
                            high_ag_list.append(ag.copy())
                            high_ag2_list.append(ag.copy())
                            high_bg_list.append(self.original_bg.copy())
                            high_a_list.append(sub_goal.copy())
                            high_mask_list.append([1])
                            high_r_list.append([0])
                    else:
                        sub_goal = self.high_agent.get_actions(high_ob, bg)
                        if self.graphplanner.dim==3:
                            min_range = [-1,-1,-1] * self.subgoal_scale + self.subgoal_offset
                            max_range = [1,1,1] * self.subgoal_scale + self.subgoal_offset
                        else:
                            min_range = [-1,-1] * self.subgoal_scale + self.subgoal_offset
                            max_range = [1,1] * self.subgoal_scale + self.subgoal_offset
                        sub_goal_ori = sub_goal.copy()
                        scale = 1.0
                        while True:
                            if self.graphplanner.dim == 3:
                                if self.graphplanner.check_reachability_3D(sub_goal) or scale > 2.0:
                                    break
                            else:
                                if self.graphplanner.check_reachability(sub_goal) or scale > 2.0:
                                    break
                            high_ob_list.append(high_ob.copy())
                            high_ob2_list.append(high_ob.copy())
                            high_ag_list.append(ag.copy())
                            high_ag2_list.append(ag.copy())
                            high_bg_list.append(self.original_bg.copy())
                            high_a_list.append(sub_goal.copy())
                            high_mask_list.append([1])
                            high_r_list.append([0])
                            sub_goal = np.random.normal(sub_goal_ori,scale=scale)
                            sub_goal = np.clip(sub_goal, min_range, max_range)
                            scale += 0.1
                    high_dict['ob'] = high_ob_list
                    high_dict['ag'] = high_ag_list
                    high_dict['bg'] = high_bg_list
                    high_dict['mask'] = high_mask_list
                    high_dict['r'] = high_r_list
                    high_dict['a'] = high_a_list
                    high_dict['ag2'] = high_ag2_list
                    high_dict['o2'] = high_ob2_list
                    if reduce_noise:
                        self.epsilon -= 0.005
                        self.epsilon = max(self.args.epsilon_min,self.epsilon)
            self.curr_subgoal = sub_goal 
            self.way_to_subgoal = self.subgoal_freq
            #graph search
            if (self.graphplanner.graph is not None):
                new_sg = self.graphplanner.find_path(ob, self.curr_subgoal, ag, bg, train=True, first = first, fail_ratio= not do_exp)
                if new_sg is not None:
                    self.curr_subgoal = new_sg

        # which waypoint to chase
        self.waypoint_subgoal = self.graphplanner.get_waypoint(ob, ag, self.curr_subgoal, bg, train=True)[:self.subgoal_dim]      
        self.high_waypoint = self.waypoint_subgoal.copy()
        if self.prev_high_waypoint is not None and not np.array_equal(self.prev_high_waypoint, self.high_waypoint):
            self.prev_waypoint = self.prev_high_waypoint.copy()
            self.prev_high_ag = ag.copy()
            spend_time = timestep/self.env_params['max_timesteps']
            self.prev_high_ob = np.concatenate((ob,[spend_time])).copy()
            self.prev_high_info = self.info_high.copy()
        
        if self.high_waypoint is not None:
            self.prev_high_waypoint = self.high_waypoint.copy()

        #find low level policy action
        if act_randomly and not self.args.low_agent:
            act = np.random.uniform(low=-a_max, high=a_max, size=self.env_params['l_action_dim'])
        else:
            act = self.low_agent.get_actions(ob, self.waypoint_subgoal)
            if self.args.noise_eps > 0.0:
                act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
                act = np.clip(act, -a_max, a_max)
            if self.args.random_eps > 0.0:
                a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
                mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
                if self.num_envs > 1:
                    mask = np.expand_dims(mask, -1)
                act += mask * (a_rand - act)
        self.way_to_subgoal -= 1
        

        return act, high_dict
    


    def low_agent_optimize(self, logging=False):
        first=True
        for n_train in range(self.args.n_batches):
            batch = self.low_replay.sample(batch_size=self.args.batch_size)
            self.low_learner.update_critic(batch, train_embed=True, logging=logging and first)
            batch_g = self.low_replay.sample_g(batch_size=self.args.batch_size)
            self.low_learner.update_critic_g(batch_g, train_embed=True, logging=logging and first)
            if self.low_opt_steps % self.args.actor_update_freq == 0:
                self.low_learner.update_actor(batch, train_embed=True, logging=logging and first)
            self.low_opt_steps += 1
            if self.low_opt_steps % self.args.target_update_freq == 0:
                self.low_learner.target_update()
            first=False
        


    def high_agent_optimize(self, logging=False):
        first=True
        for n_train in range(self.args.n_batches):
            batch = self.high_replay.sample(batch_size=self.args.batch_size, graphplanner = self.graphplanner)
            self.high_learner.update_critic(batch, train_embed=True, logging=logging and first)
            if self.high_opt_steps % self.args.actor_update_freq == 0:
                self.high_learner.update_actor(batch, train_embed=True, logging=logging and first)
            self.high_opt_steps += 1
            if self.high_opt_steps % self.args.target_update_freq == 0:
                self.high_learner.target_update()
            first=False

    def collect_experience(self, epoch, n_iter, random_goal= False, act_randomly=False, train_agent=True, graph=False, video=False, do_exp=True):
        low_ob_list, low_ag_list, low_bg_list, low_a_list, low_mask_list = [], [], [], [], []
        high_ob_list, high_ag_list, high_bg_list, high_a_list, high_mask_list, high_r_list = [], [], [], [], [], []
        high_ob_list2, high_ag_list2, high_bg_list2, high_a_list2, high_mask_list2, high_r_list2 = [], [], [], [], [], []
        high_wp_list = []
        high_ag2_list, high_ob2_list = [], []
        self.monitor.update_episode()
        observation = self.env.reset()
        first = True
        visit_index = []
        
        self.curr_subgoal = None
        ob = observation['observation']
        ag = observation['achieved_goal']
        bg = observation['desired_goal']
        self.original_bg = bg.copy()
        if self.graphplanner.dim==3:
            self.prev_waypoint = np.array([0.,0.,0.])
        else:
            self.prev_waypoint = np.array([0.,0.])
        self.prev_high_ob = np.concatenate((ob,[0.])).copy()
        self.prev_high_ag = ag.copy()
        a_max = self.env_params['action_max']
        self.prev_ag = ag.copy()
        self.stay_count = 0

        temp_dist = 100
        timestep = 0
        done=False
        self.do_next_high = True
        self.high_waypoint = None
        try:
            info=observation['info']
            self.info_high=observation['info']
        except:
            info={}
            self.info_high={}
        self.prev_high_info = self.info_high.copy()
        truncated = False
        first_save=False
        first_logging=True
        first_logging_high=True
        self.prev_high_waypoint = None
        self.inside_count = 0
        while True:
            if video:            
                if "Reacher" in self.args.env_name  or "Bottle" in self.args.env_name:
                    frame = self.env.render(mode='rgb_array')
                    self.env_frames.append(frame)
                else:
                    frame = self.env.base_env.render(mode='rgb_array')
                    self.env_frames.append(frame)
            if self.curr_subgoal is not None:
                temp_dist = self.env.goal_distance(ag, self.curr_subgoal)
                if temp_dist <= self.args.lambda1_inside:
                    self.inside_count +=1
                else:
                    self.inside_count = 0
                do_not_move = self.calc_move(ag, timestep)
                truncated = do_not_move
                self.do_next_high = (temp_dist <= self.args.lambda1) or ((temp_dist<=self.args.lambda1_inside) and self.inside_count >= self.args.inside_count)
            
            if (self.do_next_high) and not act_randomly:
                spend_time = timestep/self.env_params['max_timesteps']
                high_ob = np.concatenate((ob.copy(),[spend_time]))
                high_ob_list.append(high_ob.copy())
                high_ag_list.append(ob[:self.args.subgoal_dim].copy())
                high_bg_list.append(self.original_bg.copy())
                if first_save:
                    high_a_list.append(self.curr_subgoal.copy())
                    high_r_list.append([self.high_reward_func(achieved_goal = high_ag_list[-1], goal = high_bg_list[-1], ob_old = high_ob_list[-2], ob = high_ob)])
                    high_ag2_list.append(ob[:self.args.subgoal_dim].copy())
                    high_ob2_list.append(high_ob.copy())
                    high_mask_list.append([0])
                    if high_r_list[-1][0] > 0:
                        self.check_positive_reward = True
                first_save=True
                self.info_high = info.copy()
            act, high_dict = self.get_actions(ob, ag, bg, a_max=a_max, random_goal= random_goal, act_randomly=act_randomly, graph=graph, first = first, do_exp=do_exp, reduce_noise=first, timestep=timestep)
            if self.do_next_high:
                self.do_next_high=False
                self.inside_count=0
            
            if high_dict:  # More Pythonic way to check if dict is non-empty
                for temp_ob, temp_ag, temp_ag2, temp_o2, temp_a, temp_bg, temp_r, temp_mask in zip(
                    high_dict["ob"], high_dict["ag"], high_dict["ag2"], high_dict["o2"],
                    high_dict["a"], high_dict["bg"], high_dict["r"], high_dict["mask"]
                ):
                    # Create new lists by extending with current batch elements
                    fail_experience = {
                        "ob": high_ob_list.copy(),
                        "ag": high_ag_list.copy(),
                        "bg": high_bg_list.copy(),
                        "a": high_a_list + [temp_a] if high_a_list else [temp_a],
                        "wp": [],
                        "mask": high_mask_list + [temp_mask] if high_mask_list else [temp_mask],
                        "r": high_r_list + [temp_r] if high_r_list else [temp_r],
                        "ag2": high_ag2_list + [temp_ag2] if high_ag2_list else [temp_ag2],
                        "o2": high_ob2_list + [temp_o2] if high_ob2_list else [temp_o2]
                    }

                    # Convert lists to numpy arrays for proper dimension handling
                    fail_experience = {k: np.array(v) for k, v in fail_experience.items()}

                    # Ensure correct shape
                    if fail_experience["ob"].ndim == 2:
                        fail_experience = {k: np.expand_dims(v, axis=0) for k, v in fail_experience.items()}
                    else:
                        fail_experience = {k: np.swapaxes(v, 0, 1) for k, v in fail_experience.items()}

                    # Store in appropriate replay buffer
                    self.high_replay.store0(fail_experience)
            first = False
                

            low_ob_list.append(ob.copy())
            low_ag_list.append(ag.copy())
            low_bg_list.append(self.waypoint_subgoal.copy())
            low_a_list.append(act.copy())
            low_mask_list.append([1])
            if self.graphplanner.dim==3:
                column = self.args.map_size/self.args.grid_size
                depth = self.args.map_size/self.args.grid_size
                s_position = (ag.copy()+self.args.offset)//self.args.grid_size
                low_index = s_position[2]*column*depth + s_position[1]*column + s_position[0]
            else:
                s_position = (ag.copy()+self.args.offset)//self.args.grid_size
                low_index = s_position[1]*(self.args.map_size/self.args.grid_size) + s_position[0]
            if not (low_index in visit_index) and do_exp:
                visit_index.append(low_index)
                self.graphplanner.successClusterPush(low_index)
            
            observation, _, _, info = self.env.step(act)
            ob = observation['observation']
            ag = observation['achieved_goal']
            if act_randomly == False:
                self.env_steps += 1
                self.monitor.env_steps +=1
            for every_env_step in range(self.num_envs):
                if train_agent:
                    self.low_agent_optimize(first_logging)
                    first_logging=False
                    if self.args.high_agent and epoch > self.args.start_planning_epoch and self.env_steps % self.args.high_optimize_freq == 0:
                        self.high_agent_optimize(first_logging_high)
                        first_logging_high=False
            
            self.total_timesteps += self.num_envs
            if not 'DoubleGoal' in self.args.env_name:
                Train_Dist = self.env.goal_distance(ag, self.original_bg)
            else:
                Train_Dist = self.env.goal_distance_min(ag, self.original_bg)
            temp_dist = self.env.goal_distance(ag, self.curr_subgoal.copy())
            if (Train_Dist<=self.env.distance_threshold_high) and ((temp_dist<=self.args.lambda1) or ((temp_dist<=self.args.lambda1_inside) and self.inside_count >= self.args.inside_count)):
                if 'DoubleKeyChest' in self.args.env_name:
                    if self.env.has_key1 and self.env.has_key2:
                        done = True
                elif 'KeyChest' in self.args.env_name:
                    if self.env.has_key:
                        done = True
                elif 'DoubleGoal' in self.args.env_name:
                    if self.env.goal1_achieved and self.env.goal2_achieved:
                        done = True
                else:
                    done = True
            truncated = (timestep == self.env_params['max_timesteps'] - 1) or truncated
            timestep +=1
            if done or truncated:
                if not act_randomly:
                    last_high_ob = np.concatenate((ob.copy(),[1.0]))
                    high_ob_list2 = high_ob_list.copy()
                    high_ag_list2 = high_ag_list.copy()
                    high_bg_list2 = high_bg_list.copy()
                    high_ob2_list2 = high_ob2_list.copy()
                    high_ag2_list2 = high_ag2_list.copy()
                    high_r_list2 = high_r_list.copy()
                    high_a_list2 = high_a_list.copy()
                    high_mask_list2 = high_mask_list.copy()
                    final_dist = self.env.goal_distance(ag, self.curr_subgoal.copy())
                    if final_dist < self.args.lambda1 or (final_dist < self.args.lambda1_inside) and self.inside_count >= self.args.inside_count:
                        high_r_list.append([self.high_reward_func(achieved_goal = self.prev_high_ag.copy(), goal = self.original_bg.copy(), ob_old = high_ob_list[-1], ob = self.prev_high_ob)])
                        high_a_list.append(self.prev_waypoint.copy())
                        high_a_list2.append(self.curr_subgoal.copy()) 
                        high_r_list2.append([self.high_reward_func(achieved_goal = ag.copy(), goal = self.original_bg.copy(), ob_old = high_ob_list2[-1], ob = last_high_ob)])
                    else: ### stop-on-failure and partial-success
                        high_a_list.append(self.prev_waypoint.copy()) 
                        high_a_list2.append(self.curr_subgoal.copy())
                        high_r_list.append([self.high_reward_func(achieved_goal = self.prev_high_ag.copy(), goal = self.original_bg.copy(), ob_old = high_ob_list[-1], ob =self.prev_high_ob)])
                        high_r_list2.append([0.0])
                    high_mask_list.append([1])
                    high_mask_list2.append([1])
                    if done:
                        self.graphplanner.set_success() 
                break
        
        low_ob_list.append(ob.copy())
        low_ag_list.append(ag.copy())
        if not act_randomly:
            high_ob2_list.append(self.prev_high_ob)
            high_ag2_list.append(self.prev_high_ag.copy())
            high_ob2_list2.append(last_high_ob)
            high_ag2_list2.append(ag.copy())
            high_experience = dict(ob=high_ob_list, ag=high_ag_list, bg=high_bg_list, a=high_a_list, wp=high_wp_list, mask=high_mask_list, r=high_r_list, ag2=high_ag2_list, o2= high_ob2_list)
            high_experience2 = dict(ob=high_ob_list2, ag=high_ag_list2, bg=high_bg_list2, a=high_a_list2, wp=high_wp_list, mask=high_mask_list2, r=high_r_list2, ag2=high_ag2_list2, o2=high_ob2_list2)
            high_experience = {k: np.array(v) for k, v in high_experience.items()}
            high_experience2 = {k: np.array(v) for k, v in high_experience2.items()}
            if high_experience['ob'].ndim == 2:
                high_experience = {k: np.expand_dims(v, 0) for k, v in high_experience.items()}
                high_experience2 = {k: np.expand_dims(v, 0) for k, v in high_experience2.items()}
            else:
                high_experience = {k: np.swapaxes(v, 0, 1) for k, v in high_experience.items()}
                high_experience2 = {k: np.swapaxes(v, 0, 1) for k, v in high_experience2.items()}
            high_reward = self.high_reward_func(achieved_goal = ag, goal = bg, ob_old=high_ob_list[-1], ob=high_ob2_list[-1], info=info)
            self.high_experience = high_experience.copy()
            self.high_experience2 = high_experience2.copy()
        else:
            high_reward = 0
        if truncated and not act_randomly:
            if self.graphplanner.dim==3:
                column = self.args.map_size/self.args.grid_size
                depth = self.args.map_size/self.args.grid_size
                c_position = (self.waypoint_subgoal.copy()+self.args.offset)//self.args.grid_size
                c_index =c_position[2]*depth*column+ c_position[1]*column + c_position[0]
            else:
                c_position = (self.waypoint_subgoal.copy()+self.args.offset)//self.args.grid_size
                c_index = c_position[1]*(self.args.map_size/self.args.grid_size) + c_position[0]
            if c_index in self.graphplanner.gridFailCluster and do_exp:
                self.graphplanner.failClusterPush(c_index, low_ag_list[0], self.original_bg)

        low_experience = dict(ob=low_ob_list, ag=low_ag_list, bg=low_bg_list, a=low_a_list, mask=low_mask_list)
        low_experience = {k: np.array(v) for k, v in low_experience.items()}
        if low_experience['ob'].ndim == 2:
            low_experience = {k: np.expand_dims(v, 0) for k, v in low_experience.items()}
        else:
            low_experience = {k: np.swapaxes(v, 0, 1) for k, v in low_experience.items()}
        low_reward = self.low_reward_func(ag, self.waypoint_subgoal.copy(), None)
        

        total_success_count = 0
        if not 'DoubleGoal' in self.args.env_name:
            Train_Dist = self.env.goal_distance(ag, self.original_bg)
        else:
            Train_Dist = self.env.goal_distance_min(ag, self.original_bg)
        if Train_Dist <= self.env.distance_threshold_high:
            if 'DoubleKeyChest' in self.args.env_name:
                if self.env.has_key1 and self.env.has_key2:
                    total_success_count = 1
            elif 'KeyChest' in self.args.env_name:
                if self.env.has_key:
                    total_success_count = 1
            elif 'DoubleGoal' in self.args.env_name:
                if self.env.goal1_achieved and self.env.goal2_achieved:
                    total_success_count = 1
            else:
                total_success_count = 1

        
        self.low_experience = low_experience.copy()
        self.low_replay.store(low_experience)
        if not act_randomly:
            if any(r[0] > 0 for r in high_r_list):
                self.high_replay.store1(high_experience)
            else:
                self.high_replay.store0(high_experience)
            if any(r[0] > 0 for r in high_r_list2):
                self.high_replay.store1(high_experience2)
            else:
                self.high_replay.store0(high_experience2)
        if total_success_count > 0:
            self.graphplanner.first_success = True
        return total_success_count, low_reward, high_reward, Train_Dist
    

    def run(self):
        do_exp = True
        do_high = 1.0
        total_exp = self.args.n_cycles - do_high
        for n_init_rollout in range(self.args.n_initial_rollouts // self.num_envs):
            self.collect_experience(epoch = -1, n_iter = -1, random_goal= True, act_randomly=True, train_agent=False, graph=False)

        total_rollout_time = 0
        for epoch in range(self.args.n_epochs):
            self.timer.start('epoch')
            total_success_count = 0
            print('Epoch %d: Iter (out of %d)=' % (epoch, self.args.n_cycles), end=' ')
            sys.stdout.flush()

            total_low_reward = 0
            total_high_reward = 0
            total_distance = 0
            
            total_try_exp = 0
            if self.args.env_name == 'AntMazeKeyChest':
                has_key_train_no =0
                has_key_exp_no =0
            if 'DoubleGoal' in self.args.env_name:
                first_goal = 0
                exp_first_goal = 0
                second_goal = 0
                exp_second_goal = 0
            if self.args.env_name == 'AntMazeDoubleKeyChest':
                has_key_train_no1 =0
                has_key_exp_no1 =0
                has_key_train_no2 =0
                has_key_exp_no2 =0
            
            if epoch >= self.args.start_planning_epoch :
                #goal_scheduling = True
                self.graphplanner.graph_construct(epoch)
            
            if self.graphplanner.graph is not None:
                if epoch % self.args.densify_freq == 0:
                    self.graphplanner.densify()
            total_try_count = 0
            total_success_exp = 0
            first=True
            for n_iter in range(self.args.n_cycles):
                print("%d" % n_iter, end=' ' if n_iter < self.args.n_cycles - 1 else '\n')
                sys.stdout.flush()
                
                video = False
                if (epoch % self.args.video_freq == 0) and (n_iter == 0) and self.args.save_video:
                    video = True
                self.coverage = self.graphplanner.calc_coverage()
                if n_iter < total_exp:
                    do_exp=True
                else:
                    do_exp=False

                self.timer.start('rollout')
                success_count, low_reward, high_reward, Train_Dist = self.collect_experience(epoch, n_iter, train_agent=True, graph=True, video=video, do_exp=do_exp)
                self.timer.end('rollout')
                self.monitor.store(TimePerSeqRollout=self.timer.get_time('rollout'))
                if not do_exp:
                    total_success_count += success_count
                    total_try_count += 1
                    if (self.args.env_name == 'AntMazeKeyChest') and self.env.has_key == True:
                        has_key_train_no +=1
                    if (self.args.env_name == 'AntMazeDoubleKeyChest') and self.env.has_key1 == True:
                        has_key_train_no1 +=1
                    if (self.args.env_name == 'AntMazeDoubleKeyChest') and self.env.has_key2 == True:
                        has_key_train_no2 +=1
                    if ('DoubleGoal' in self.args.env_name) and self.env.goal1_achieved:
                        first_goal += 1
                    if ('DoubleGoal' in self.args.env_name) and self.env.goal2_achieved:
                        second_goal += 1
                else:
                    total_try_exp += 1
                    total_success_exp += success_count
                    if (self.args.env_name == 'AntMazeKeyChest') and self.env.has_key == True:
                        has_key_exp_no +=1
                    if (self.args.env_name == 'AntMazeDoubleKeyChest') and self.env.has_key1 == True:
                        has_key_exp_no1 +=1
                    if (self.args.env_name == 'AntMazeDoubleKeyChest') and self.env.has_key2 == True:
                        has_key_exp_no2 +=1
                    if ('DoubleGoal' in self.args.env_name) and self.env.goal1_achieved:
                        exp_first_goal += 1
                    if ('DoubleGoal' in self.args.env_name) and self.env.goal2_achieved:
                        exp_second_goal += 1
                
                
                total_low_reward += low_reward
                total_high_reward += high_reward
                total_distance += Train_Dist
                if video:
                    self.env_render(self.env_frames, self.total_timesteps)
                if epoch >= self.args.start_planning_epoch :
                    self.graphplanner.graph_construct(epoch)
                    if n_iter % 5 != 0 or epoch %10 != 0:
                        continue
                    landmarks = self.graphplanner.landmarks
                    save_path = osp.join(self.model_path,'img')
                    os.makedirs(save_path, exist_ok=True)
                    save_path1 = osp.join(save_path,"graph"+str(epoch)+"_"+str(n_iter)+".png")
                    fig, ax=plt.subplots(2, constrained_layout=True)
                    samples = self.low_replay.sample_regular_batch(2048)
                    fps_selections = self.graphplanner.fps_selection(samples['ag'],samples['ob'], 600)
                    if not 'Reacher' in self.args.env_name:
                        map_size, wall_x, wall_y = self.get_map_info()
                        ax[0].plot(wall_x, wall_y, c ='k')
                        ax[0].set_xlim(map_size[0])
                        ax[0].set_ylim(map_size[1])
                    ax[0].plot(landmarks[:,0], landmarks[:,1], 'ro', markersize=2, label='landmarks')
                    ax[0].plot(samples['ag'][fps_selections,0], samples['ag'][fps_selections,1], 'bo', markersize=1, label='FPS selection')
                    
                    ax[0].legend(loc='upper left')

                    if not 'Reacher' in self.args.env_name:
                        ax[1].plot(wall_x, wall_y, c ='k')
                        ax[1].set_xlim(map_size[0])
                        ax[1].set_ylim(map_size[1])
                    for i in range(len(self.low_experience['ag'][0])):
                        color = (0, 1 - i /(len(self.low_experience['ag'][0])),0) 
                        low_x = self.low_experience['ag'][0][i,0]
                        low_y = self.low_experience['ag'][0][i,1]
                        ax[1].plot(low_x, low_y, ls='-', marker='.', markersize=2, color=color)
                    for i in range(len(self.high_experience['a'][0])):
                        color = (0, 0, 1) 
                        ax[1].plot(self.high_experience['a'][0][i,0], self.high_experience['a'][0][i,1], marker='o', markersize=6, color=color) #######line으로, 그러면서 색 변하게
                    if do_exp:
                        high_legend = Line2D([0], [0], color='blue', ls='--', marker='o', markersize=2, label='Exploration')
                        low_legend = Line2D([0], [0], color='green', ls='-', marker='.', markersize=2, label='Low-level agent location')
                    else:
                        high_legend = Line2D([0], [0], color='blue', ls='--', marker='o', markersize=2, label='High-level subgoal')
                        low_legend = Line2D([0], [0], color='green', ls='-', marker='.', markersize=2, label='Low-level agent location')

                    ax[1].legend(handles=[high_legend, low_legend], loc='upper right')
                    plt.savefig(save_path1)
                    if epoch %10 == 0:
                        wandb.log({"media/train_and_graph_spread_{}_{}".format(epoch,n_iter): wandb.Image(save_path1)})
                    plt.close(fig)                    
                self.episode +=1
            self.timer.end('epoch')
            if (self.args.env_name == 'AntMazeKeyChest'):
                self.monitor.store(has_key_rate_train=has_key_train_no/total_try_count)
                self.monitor.store(has_key_rate_exp=has_key_exp_no/total_try_exp)
            elif 'DoubleGoal' in self.args.env_name:
                self.monitor.store(first_goal_train=first_goal/total_try_count)
                self.monitor.store(second_goal_train=second_goal/total_try_count)
                self.monitor.store(first_goal_exp=exp_first_goal/total_try_exp)
                self.monitor.store(second_goal_exp=exp_second_goal/total_try_exp)
            elif self.args.env_name == 'AntMazeDoubleKeyChest':
                self.monitor.store(has_key_rate1_train=has_key_train_no1/total_try_count)
                self.monitor.store(has_key_rate1_exp=has_key_exp_no1/total_try_exp)
                self.monitor.store(has_key_rate2_train=has_key_train_no2/total_try_count)
                self.monitor.store(has_key_rate2_exp=has_key_exp_no2/total_try_exp)
            self.monitor.store(TimePerSeqRolloutTotal=self.timer.get_time('epoch'))
            self.monitor.store(TrainLowReward=total_low_reward/self.args.n_epochs)
            self.monitor.store(TrainHighRewardMax=total_high_reward/self.args.n_epochs)
            self.monitor.store(TrainGoalDist=total_distance/self.args.n_epochs)
            self.monitor.store(Coverage = self.coverage)
            self.monitor.store(Exploration = float(do_exp))
            self.monitor.store(Epsilon = self.epsilon)
            if total_try_count != 0:
                Train_Success_Rate=total_success_count/total_try_count
                self.monitor.store(Train_Success_Rate=Train_Success_Rate)
            self.monitor.store(env_steps=self.env_steps)
            self.monitor.store(episode=self.episode)
            self.monitor.store(low_opt_steps=self.low_opt_steps)
            self.monitor.store(high_opt_steps=self.high_opt_steps)
            self.monitor.store(low_replay_fill_ratio=float(self.low_replay.current_size / self.low_replay.size))
            self.monitor.store(high_replay_fill_ratio0=float(self.high_replay.current_size0 / self.high_replay.size0))
            self.monitor.store(high_replay_fill_ratio1=float(self.high_replay.current_size1 / self.high_replay.size1))
            self.monitor.store(exploration_number = total_exp)

            her_success = self.run_eval(epoch, use_test_env=False, render=self.args.eval_render)   
            print('Epoch %d her eval %.3f'%(epoch, her_success))
            print('Log Path:', self.log_path)
            self.monitor.store(Test_Success_Rate=her_success)
            if total_try_exp != 0:
                self.monitor.store(Exp_Success_Rate=total_success_exp/total_try_exp)
            total_exp = self.args.n_cycles - int(do_high)
            total_exp = max(int(self.args.eta * self.args.n_cycles), total_exp)
            do_high += self.args.exp_rate
            
            if self.args.store_epoch:
                if epoch > 0 and epoch % self.args.epoch_save_iter == 0:
                    self.save_all(self.model_path, epoch)

            if self.args.target_env_steps > 0 and self.env_steps >= self.args.target_env_steps:
                print('Reached target_env_steps %d at env_steps %d' % (self.args.target_env_steps, self.env_steps))
                break
            
        # if not self.args.store_epoch:
        self.save_all(self.model_path)


    def run_eval(self, epoch, use_test_env=False, render=False, printer = False):
        env = self.env
        if use_test_env and hasattr(self, 'test_env'):
            print("use test env")
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        total_success_timestep = 0
        success_first = False
        haskey_count=  0
        haskey_count1 = 0
        haskey_count2 = 0
        first_goal = 0
        second_goal = 0
        total_test_dist = 0
        for n_test in range(self.args.n_test_rollouts):
            eval_frame = []
            success_timestep = self.env_params['max_timesteps']
            success_first = False
            self.curr_subgoal = None
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            self.prev_ag = ag.copy()
            self.stay_count = 0
            first = True
            self.do_next_high = True
            self.new_command_count = 0
            info=None
            timestep = 0
            done = False
            truncated = False
            high_ob_list, high_ag_list, high_bg_list, high_a_list = [], [], [], []
            low_ob_list, low_ag_list, low_bg_list, low_a_list = [], [], [], []
            first_save=False
            self.inside_count = 0
            while True:
                if self.args.save_video:
                    if "Reacher" in self.args.env_name or "Bottle" in self.args.env_name:
                        frame = env.render(mode='rgb_array')
                        eval_frame.append(frame)
                    else:
                        frame = env.base_env.render(mode='rgb_array')
                        eval_frame.append(frame)
                if self.curr_subgoal is not None:
                    temp_dist = env.goal_distance(ag, self.curr_subgoal)
                    if temp_dist <=self.args.lambda1_inside:
                        self.inside_count +=1
                    else:
                        self.inside_count = 0
                    
                    do_not_move = self.calc_move(ag, timestep)
                    truncated = do_not_move
                    self.do_next_high = (temp_dist <= self.args.lambda1) or ((temp_dist<= self.args.lambda1_inside) and self.inside_count >= self.args.inside_count) or (self.stay_count>300 and self.new_command_count > 100)
                if (self.do_next_high):
                    spend_time = timestep/self.env_params['max_timesteps']
                    high_ob = np.concatenate((ob.copy(),[spend_time]))
                    high_ob_list.append(high_ob.copy())
                    high_ag_list.append(ob[:self.args.subgoal_dim].copy())
                    high_bg_list.append(bg.copy())
                    if first_save:
                        high_a_list.append(self.curr_subgoal.copy())
                    first_save=True
                act = self.eval_get_actions(ob, ag, bg, first = first, timestep=timestep)
                self.new_command_count +=1
                if self.do_next_high:
                    self.do_next_high=False  
                    self.inside_count=0 
                if render:
                    env.render()
                first = False
                
                
                           
                low_ob_list.append(ob.copy())
                low_ag_list.append(ag.copy())
                low_bg_list.append(bg.copy())
                low_a_list.append(act.copy())
                observation, _, _, info = env.step(act)
                ob = observation['observation']
                ag = observation['achieved_goal']
                if info['is_success']:
                    if success_first == False:
                        success_first = True
                        success_timestep = timestep
                if not 'DoubleGoal' in self.args.env_name:
                    Train_Dist = env.goal_distance(ag, bg)
                else:
                    Train_Dist = env.goal_distance_min(ag, bg)
                if Train_Dist <= env.distance_threshold_high:
                    if 'DoubleKeyChest' in self.args.env_name:
                        if env.has_key1 and env.has_key2:
                            done = True
                    elif 'KeyChest' in self.args.env_name:
                        if env.has_key:
                            done = True
                    elif 'DoubleGoal' in self.args.env_name:
                        if env.goal1_achieved and env.goal2_achieved:
                            done = True
                    else:
                        done = True
                truncated = (timestep == self.env_params['max_timesteps'] - 1) or truncated
                timestep +=1
                if done or truncated:
                    last_high_ob = np.concatenate((ob.copy(),[1.0]))
                    high_a_list.append(self.curr_subgoal.copy())
                    break
            if printer:
                print("Final location : ", ag)
                print("Distance : ", np.linalg.norm(ag - self.curr_subgoal))
            low_ob_list.append(ob.copy())
            low_ag_list.append(ag.copy())
            high_ob_list.append(last_high_ob)
            high_ag_list.append(ag.copy())
            low_ag_list = np.array(low_ag_list)
            high_a_list = np.array(high_a_list)


            if not 'DoubleGoal' in self.args.env_name:
                TestEvn_Dist = env.goal_distance(ag, bg)
            else:
                TestEvn_Dist = env.goal_distance_min(ag, bg)
            total_test_dist += TestEvn_Dist
            total_success_timestep += success_timestep
            total_trial_count += 1
            prev_total_success_count = total_success_count
            if TestEvn_Dist <= env.distance_threshold_high:
                if 'DoubleKeyChest' in self.args.env_name:
                    if env.has_key1 and env.has_key2:
                        total_success_count +=1
                elif 'KeyChest' in self.args.env_name:
                    if env.has_key:
                        total_success_count +=1
                elif 'DoubleGoal' in self.args.env_name:
                    if env.goal1_achieved and env.goal2_achieved:
                        total_success_count += 1
                else:
                    total_success_count +=1
            if (self.args.env_name == 'AntMazeKeyChest') and env.has_key:
               haskey_count +=1  
            if (self.args.env_name == 'AntMazeDoubleKeyChest') and env.has_key1:
               haskey_count1 +=1   
            if (self.args.env_name == 'AntMazeDoubleKeyChest') and env.has_key2:
               haskey_count2 +=1       
            if ('DoubleGoal' in self.args.env_name) and env.goal1_achieved:
                first_goal +=1
            if ('DoubleGoal' in self.args.env_name) and env.goal2_achieved:
                second_goal +=1
            if total_success_count > prev_total_success_count and self.args.save_video:
                self.env_render_eval(frames=eval_frame, epoch=epoch, n_test = n_test)
            save_path = osp.join(self.model_path,'img')
            os.makedirs(save_path, exist_ok=True)
            if n_test % 10 != 0:
                continue
            save_path3 = osp.join(save_path,"eval_trajectory"+str(epoch)+"_"+str(n_test)+".png")
            fig = plt.figure()
            if not 'Reacher' in self.args.env_name:
                map_size, wall_x, wall_y = self.get_map_info()
                plt.plot(wall_x, wall_y, c ='k')
                plt.xlim(map_size[0])
                plt.ylim(map_size[1])
            for i in range(len(low_ag_list)):
                color = (0, 1 - i /(len(low_ag_list)),0) 
                low_x = low_ag_list[i:i+1,0]
                low_y = low_ag_list[i:i+1,1]
                plt.plot(low_x, low_y, ls='-', marker='.', markersize=2, color=color)
            
            for i in range(len(high_a_list)):
                color = (0, 0, 1) 
                plt.plot(high_a_list[i:i+1,0], high_a_list[i:i+1,1], marker='o', markersize=6, color=color) #######line으로, 그러면서 색 변하게

            self.high_experience_high = high_a_list.copy()
            self.low_experience_high = low_ag_list.copy()

            high_legend = Line2D([0], [0], color='blue', ls='--', marker='o', markersize=2, label='High-level subgoal')
            low_legend = Line2D([0], [0], color='green', ls='-', marker='.', markersize=2, label='Low-level agent location')
            plt.legend(handles=[high_legend, low_legend], loc='upper right')
            plt.savefig(save_path3)
            if epoch %10 == 0:
                wandb.log({"media/eval_trajectory_{}_{}".format(epoch,n_test): wandb.Image(save_path3)})
            plt.close(fig)
        self.monitor.store(TestEvn_Dist=total_test_dist/self.args.n_test_rollouts)
        self.monitor.store(success_timestep=total_success_timestep/self.args.n_test_rollouts)
        if (self.args.env_name == 'AntMazeKeyChest'):
            self.monitor.store(has_key_rate_eval=haskey_count/total_trial_count)
        if 'DoubleGoal' in self.args.env_name:
            self.monitor.store(first_goal_eval=first_goal/total_trial_count)
            self.monitor.store(second_goal_eval=second_goal/total_trial_count)
        if self.args.env_name == 'AntMazeDoubleKeyChest':
            self.monitor.store(has_key_rate1_eval=haskey_count1/total_trial_count)
            self.monitor.store(has_key_rate2_eval=haskey_count2/total_trial_count)
        success_rate = total_success_count / total_trial_count
        return success_rate


    def calc_move(self, ag, timestep):
        dist = self.env.goal_distance(self.prev_ag, ag)
        if dist > 0.5:
            self.prev_ag = ag.copy()
            self.stay_count = 0
            return False
        else:
            self.stay_count +=1
        if self.stay_count > self.args.move_count:
            self.stay_count = 0
            return True
        else:
            return False



    def get_map_info(self):
        if self.env.env_name == 'AntMazeShortcutDetour':
            map_size = [[-4, 24], [-4, 16]]
            wall_x = [
                -2.2, 21.8, 21.8, -2.2, -2.2, np.nan,
                -2.2, 1.9, 1.9, -2.2, -2.2, np.nan,
                4.1, 15.2, 15.2, 4.1, 4.1
            ]
            wall_y = [
                -2.6, -2.6, 15.4, 15.4, -2.6, np.nan,
                2.4, 2.4, 10.4, 10.4, 2.4, np.nan,
                2.4, 2.4, 10.4, 10.4, 2.4
            ]
            size = 2
        elif self.env.env_name == 'AntMazeBottleneck':
            map_size = [[-8, 20],[-8, 20]]
            wall_x = [-4, 20, 20, 17, 17, 20, 20, -4, -4, 12, 12, 15, 15, 12, 12, -4, -4]
            wall_y = [-4, -4,  7,  7,  9,  9, 20, 20, 12, 12,  9,    9,    7,  7,  4,  4, -4]
            size = 4
        elif self.env.env_name == 'AntMazeDoubleBottleneck':
            map_size = [[-8, 40],[-8, 40]]
            wall_x = [-4, 20, 20, 17, 17, 20, 20,  4,  4,  1,  1,  4,  4, 20, 20, -4, -4, -1, -1, -4, -4, 12, 12, 15, 15, 12, 12, -4, -4]
            wall_y = [-4, -4,  7,  7,  9,  9, 20, 20, 23, 23, 25, 25, 28, 28, 36, 36, 25, 25, 23, 23, 12, 12,  9,  9,  7,  7,  4,  4, -4]
            size=4
        elif self.env.env_name == 'AntMaze':
            map_size = [[-4, 20],[-4, 20]]
            wall_x = [-4, 20, 20, -4, -4, 12, 12, -4, -4]
            wall_y = [-4, -4, 20, 20, 12, 12,  4,  4, -4]
            size = 4
        elif self.env.env_name == 'AntMazeP':
            map_size = [[-14, 30],[-6, 38]]
            wall_x = [-12,  4,  4, -4, -4,  4,  4, 12, 12, 20, 20, 12, 12, 28, 28, 20, 20, 28, 28, -12, -12, -4, -4, -12, -12]
            wall_y = [ -4, -4,  4,  4, 12, 12, 28, 28, 12, 12,  4,  4, -4, -4, 20, 20, 28, 28, 36,  36,  28, 28, 20,  20,  -4]
            size = 4
        elif self.env.env_name == 'AntMazeComplex-v0':
            # -4 ~ 52
            map_size = [[-4, 52],[-4, 52]]
            wall_x = [-4, -4, 12, 12, -4, -4,  4,  4, 12, 12, 20, 20, 28, 28, 52, 52, 44, 44, 36, 36, 44, 44, 52, 52, 28, 28, 36, 36, 28, 28, 20, 20, 12, 12,  4,  4, 20, 20, -4]
            wall_y = [-4,  4,  4, 12, 12, 52, 52, 44, 44, 52, 52, 44, 44, 52, 52, 36, 36, 44, 44, 28, 28, 12, 12, -4, -4, 12, 12, 20, 20, 36, 36, 28, 28, 36, 36, 20, 20, -4, -4]
            size = 4    
        elif self.env.env_name =='AntMazeKeyChest':
            map_size = [[-8,40],[-8,40]]
            wall_x = [-4, 36, 36, -4, -4, 28, 28, -4, -4, 28, 28, -4, -4]
            wall_y = [-4, -4, 36, 36, 28, 28, 20, 20, 12, 12,  4,  4, -4]
        elif self.env.env_name == 'AntMazeDoubleKeyChest':
            map_size = [[-8,40],[-8,40]]
            wall_x = [-4, -4, 20, 20,  4,  4, 36, 36, 28, 28, 20, 20, 12, 12,  4,  4, -4]
            wall_y = [-4, 36, 36, 28, 28, 20, 20, -4, -4, 12, 12, -4, -4, 12, 12, -4, -4]
        return map_size, wall_x, wall_y



    def eval_get_actions(self, ob, ag, bg, timestep, a_max=1, random_goal=False, act_randomly=False, graph=False, first = False):
        if ((self.curr_subgoal is None) or (self.do_next_high)):
            self.curr_highpolicy_obs = ob
            spend_time = timestep/self.env_params['max_timesteps']
            high_ob = np.concatenate((self.curr_highpolicy_obs,[spend_time]))
            sub_goal = self.high_agent.get_actions(high_ob, bg)
            self.curr_subgoal = sub_goal
            if (self.graphplanner.graph is not None):
                new_sg = self.graphplanner.find_path(ob, self.curr_subgoal, ag, bg, train = False, first = first, fail_ratio=True)
                if new_sg is not None:
                    self.curr_subgoal = new_sg
            self.new_command_count = 0
        self.waypoint_subgoal = self.graphplanner.get_waypoint(ob, ag, self.curr_subgoal, bg)
            
        act = self.low_agent.get_actions(ob, self.waypoint_subgoal)
        return act

    
    def state_dict(self):
        return dict(total_timesteps=self.total_timesteps)
    
    def load_state_dict(self, state_dict):
        self.total_timesteps = state_dict['total_timesteps']

    
    def env_render(self, frames, global_step):

        imageio.mimsave("train_video_{}.mp4".format(global_step), frames, fps=0.5 / 0.05)
        self.env_frames = []
    
    def env_render_eval(self, frames, epoch, n_test):

        imageio.mimsave("eval_video_{}_{}.mp4".format(epoch,n_test), frames, fps=0.5 / 0.05)
        self.env_frames = []
    
