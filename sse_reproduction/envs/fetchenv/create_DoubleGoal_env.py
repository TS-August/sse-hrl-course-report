from .reacher_fix_DoubleGoal import Reacher3DEnv_DoubleGoal
from collections import OrderedDict
import gym
import numpy as np
from gym import Wrapper


class GoalWrapper(Wrapper):
    def __init__(self, env, env_name, reward_shaping='sparse', seed=0, subgoal_repr='subspace', mask_goal_in_obs=False):
        super(GoalWrapper, self).__init__(env)
        self.env_name = env_name
        ob_space = env.observation_space
        high = np.array([np.inf, np.inf, np.inf])
        low = -high
        goal_space = gym.spaces.Box(low=low, high=high)

        if subgoal_repr == 'subspace':
            achieved_goal_space = goal_space
        elif subgoal_repr == 'whole':
            achieved_goal_space = ob_space
        else:
            raise NotImplementedError
        self.subgoal_repr = subgoal_repr

        self.observation_space = gym.spaces.Dict(OrderedDict({
            'observation': ob_space,
            'desired_goal': goal_space,
            'achieved_goal': achieved_goal_space,
        }))

        self.distance_threshold = 0.25
        self.distance_threshold_high = 0.25
        self.reward_shaping = reward_shaping
        self.mask_goal_in_obs = mask_goal_in_obs
        self.goal1 = np.array([0.4, 0.4, -0.1])
        self.goal2 = np.array([0.4, -0.8, -0.1])
        self.goal = np.concatenate([self.goal1, self.goal2])
        self.goal1_achieved = 0.
        self.goal2_achieved = 0.

    def step(self, action):
        obs, sparse_reward, done, info = self.env.step(action)
        obs[-3:] = self.goal1.copy()
        if "Reacher3D" in self.env_name:
            achieved_goal = self.env.get_EE_pos(obs[None]).squeeze()
        elif self.env_name == "Pusher-v0":
            achieved_goal = self.env.ac_goal_pos
        else:
            raise NotImplementedError
        goalDist1 = self.goal_distance(achieved_goal, self.goal1)
        goalDist2 = self.goal_distance(achieved_goal, self.goal2)
        if np.array(goalDist1) < self.distance_threshold_high:
            self.goal1_achieved = 1.
        if np.array(goalDist2) < self.distance_threshold_high:
            self.goal2_achieved = 1.
        new_obs = np.concatenate([obs, self.goal2, [self.goal1_achieved], [self.goal2_achieved]])
        
        
        if self.mask_goal_in_obs:
            obs[7:10] = 0.
        
        out = {
            'observation': new_obs,
            'desired_goal': self.goal,
            'achieved_goal': achieved_goal}
        reward = self.high_reward_func(achieved_goal,  self.goal, ob_old=self.prev_obs.copy(), ob=new_obs)
        self.prev_obs = new_obs.copy()
        info['success'] = self.goal1_achieved and self.goal2_achieved and self.goal_distance_min(achieved_goal, self.goal) <= self.distance_threshold_high
        return out, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "Reacher3D" in self.env_name:
            achieved_goal = self.env.get_EE_pos(obs[None]).squeeze()
        elif self.env_name == "Pusher-v0":
            achieved_goal = self.env.ac_goal_pos
        else:
            raise NotImplementedError
        self.goal1_achieved = 0.
        self.goal2_achieved = 0.
        if self.mask_goal_in_obs:
            obs[7:10] = 0.
        new_obs = np.concatenate([obs, self.goal2, [self.goal1_achieved], [self.goal2_achieved]])

        out = {
            'observation': new_obs,
            'desired_goal': self.goal,
            'achieved_goal': achieved_goal}
        self.prev_obs = new_obs.copy()
        return out

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist
        
    def compute_reward_high(self, achieved_goal, goal, sparse=True, ob_old=None, ob=None):
        goal1Dist = self.goal_distance(achieved_goal, goal[:3])
        goal2Dist = self.goal_distance(achieved_goal, goal[3:])
        if sparse:
            if (np.array(goal1Dist) < self.distance_threshold_high) and ob_old[-2] == 0.0:
                return 1.
            if (np.array(goal2Dist) < self.distance_threshold_high) and ob_old[-1] == 0.0:
                return 1.
            if ob[-2] == 1. and ob[-1] == 1.:
                return 5.
            return 0.0
        else: ##Now only sparse reward is designed.
            if (np.array(goal1Dist) < self.distance_threshold_high) and ob_old[-2] == 0.0:
                return 1.
            if (np.array(goal2Dist) < self.distance_threshold_high) and ob_old[-1] == 0.0:
                return 1.
            if ob[-2] == 1. and ob[-1] == 1.:
                return 5.
            return 0.0

    def low_reward_func(self, achieved_goal, goal, info, ob=None,ob_old=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None,ob_old=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info=None, ob_old=None, ob=None):
        return self.compute_reward_high(achieved_goal, goal, sparse=True, ob_old=ob_old, ob=ob)

    def high_dense_reward_func(self, achieved_goal, goal, info=None, ob_old=None, ob=None):
        return self.compute_reward_high(achieved_goal, goal, sparse=False, ob_old=ob_old, ob=ob) * 0.5

    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist
    def goal_distance_min(self, achieved_goal, bg):
        if(achieved_goal.ndim == 1):
            dist1 = np.linalg.norm(bg[:3] - achieved_goal)
            dist2 = np.linalg.norm(bg[3:] - achieved_goal)
        else:
            dist1 = np.linalg.norm(bg[:3] - achieved_goal, axis=1)
            dist1 = np.expand_dims(dist1, axis=1)
            dist2 = np.linalg.norm(bg[3:] - achieved_goal, axis=1)
            dist2 = np.expand_dims(dist2, axis=1)
        return min(dist1, dist2)
    def get_image(self, goal=None, subgoal=None, waypoint=None):
        return self.base_env.render(mode='rgb_array', width=500, height=500)



def create_DoubleGoal_env(env_name=None, seed=0, reward_shaping='sparse', subgoal_repr='subspace', mask_goal_in_obs=False):
    if env_name == "Reacher3D_Doublegoal":
        cls = Reacher3DEnv_DoubleGoal
    else:
        raise NotImplementedError

    gym_env = cls()
    gym_env.reset()
    return GoalWrapper(gym_env, env_name, reward_shaping=reward_shaping, seed=seed, subgoal_repr=subgoal_repr,
                       mask_goal_in_obs=mask_goal_in_obs)