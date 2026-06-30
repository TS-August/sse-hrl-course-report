import os
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy

import mujoco_py
import numpy as np
from gym import spaces, utils
from gym.envs.mujoco import mujoco_env


class AntMazeShortcutDetourEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Compact Ant maze with a risky narrow shortcut and an easy long detour."""

    goal_xy = np.array([0.0, 12.8])
    init_xy = np.array([0.0, 0.0])
    reward_type = 'sparse'
    distance_threshold = 0.5
    distance_threshold_high = 2.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    objects_nqpos = [0]
    objects_nqvel = [0]
    setting = 'FIFG'
    evaluate = False

    def __init__(self, file_path=None, expose_all_qpos=True,
                 expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 500
        self.nb_step = 0
        self.env_name = 'AntMazeShortcutDetour'
        self.goal = self.goal_xy.copy()
        self._temp_xml = self._build_xml()
        mujoco_env.MujocoEnv.__init__(self, self._temp_xml, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _build_xml(self):
        xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'ant.xml')
        tree = ET.parse(xml_path)
        worldbody = tree.find('.//worldbody')
        floor = worldbody.find(".//geom[@name='floor']")
        if floor is not None:
            floor.set('size', '30 25 40')

        goal_body = worldbody.find("./body[@name='goal_point']")
        goal_body.set('pos', '0.0 12.8 0.12')
        goal_site = goal_body.find(".//site[@name='goal_point:box']")
        goal_site.set('pos', '0.0 0.0 0.0')
        goal_site.set('rgba', '0 1 0 1')
        for name in ['subgoal_point', 'way_point']:
            body = worldbody.find(f"./body[@name='{name}']")
            site = body.find(f".//site[@name='{name}:box']")
            body.set('pos', '0.0 0.0 0.12')
            site.set('pos', '100.0 100.0 100.0')

        wall_body = ET.SubElement(worldbody, 'body', name='shortcut_detour_walls', pos='0.0 0.0 0.0')
        walls = [
            ('outer_left', -2.7, 6.4, 0.5, 9.5),
            ('outer_right', 22.3, 6.4, 0.5, 9.5),
            ('outer_bottom', 9.8, -3.1, 12.5, 0.5),
            ('outer_top', 9.8, 15.9, 12.5, 0.5),
            ('left_block', -0.15, 6.4, 2.05, 4.0),
            ('center_block', 9.65, 6.4, 5.55, 4.0),
        ]
        for name, x, y, sx, sy in walls:
            ET.SubElement(
                wall_body, 'geom',
                name=name, type='box', pos=f'{x} {y} 0',
                size=f'{sx} {sy} 4',
                rgba='0.4 0.4 0.4 1',
                contype='1', conaffinity='1', friction='2 0.1 0.002')

        fd, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        os.close(fd)
        tree.write(file_path)
        return file_path

    def _check_model_parameter_dimensions(self):
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def observation_space(self):
        obs_space = spaces.Box(-np.inf, np.inf, shape=(29,), dtype=np.float64)
        goal_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
        return spaces.Dict({
            'observation': obs_space,
            'achieved_goal': goal_space,
            'desired_goal': goal_space,
        })

    @property
    def physics(self):
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        return self.model

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        obs, _ = mujoco_env.MujocoEnv.reset(self, seed=seed, options=options)
        return obs

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = self.goal_distance(ob['achieved_goal'], self.goal) <= self.distance_threshold_high
        self.nb_step += 1
        info = {
            'is_success': success,
            'success': success,
            'dist': dist,
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info=None, sparse=False, ob_old=None):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = np.array(dist) > self.distance_threshold
            return -rs.astype(np.float32)
        return -dist

    def compute_reward_high(self, achieved_goal, goal, info=None, sparse=False, threshold=None, ob_old=None):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            threshold = self.distance_threshold_high if threshold is None else threshold
            rs = np.array(dist) < threshold
            return rs.astype(np.float32)
        return max(0, 1 - dist / self.distance_threshold_high)

    def low_reward_func(self, achieved_goal, goal, info=None, ob=None, ob_old=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info=None, ob=None, ob_old=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info=None, ob=None, ob_old=None):
        return self.compute_reward_high(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15],
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }

    def rand_goal(self):
        while True:
            self.goal = np.random.uniform(low=[-2.0, -2.0], high=[22.0, 14.0], size=2)
            x, y = self.goal
            in_left_block = (-2.2 < x < 1.9) and (2.4 < y < 10.4)
            in_center_block = (4.1 < x < 15.2) and (2.4 < y < 10.4)
            if not (in_left_block or in_center_block):
                break

    def reset_model(self):
        if self.setting == 'FIFG' or self.evaluate:
            self.goal = self.goal_xy.copy()
        elif self.setting == 'FIRG':
            self.rand_goal()
        else:
            raise NotImplementedError

        self.set_goal('goal_point')
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        self.init_qpos[:2] = self.init_xy
        qpos[:2] = self.init_xy
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        self.nb_step = 0
        return self._get_obs()

    def change_goal(self, x=0, y=12.8, size=2):
        self.goal = np.array([x, y]) + np.random.uniform(low=-size, high=size, size=2)
        self.set_goal('goal_point')
        return self._get_obs()

    def _body_name2id(self, name):
        if hasattr(self.model, 'body_name2id'):
            return self.model.body_name2id(name)
        return self.model.body(name).id

    def _site_name2id(self, name):
        if hasattr(self.model, 'site_name2id'):
            return self.model.site_name2id(name)
        return self.model.site(name).id

    def set_goal(self, name):
        body_id = self._body_name2id(name)
        self.model.body_pos[body_id][:2] = self.goal
        self.model.body_quat[body_id] = [1., 0., 0., 0.]

    def goal_distance(self, achieved_goal, goal):
        if achieved_goal.ndim == 1:
            return np.linalg.norm(goal - achieved_goal)
        dist = np.linalg.norm(goal - achieved_goal, axis=1)
        return np.expand_dims(dist, axis=1)

    def get_image(self, goal=None, subgoal=None, waypoint=None):
        if goal is not None:
            self.sim.data.site_xpos[self._site_name2id('goal_point:box')] = np.array([goal[0], goal[1], 0])
        if subgoal is not None:
            self.sim.data.site_xpos[self._site_name2id('subgoal_point:box')] = np.array([subgoal[0], subgoal[1], 0])
        if waypoint is not None:
            self.sim.data.site_xpos[self._site_name2id('way_point:box')] = np.array([waypoint[0], waypoint[1], 0])
        return self.render(mode='rgb_array', width=500, height=500)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 35
        self.viewer.cam.elevation = -90


class AntMazeShortcutDetourEvalEnv(AntMazeShortcutDetourEnv):
    def __init__(self, file_path=None, expose_all_qpos=True,
                 expose_body_coms=None, expose_body_comvels=None, seed=0):
        super().__init__(
            file_path=file_path,
            expose_all_qpos=expose_all_qpos,
            expose_body_coms=expose_body_coms,
            expose_body_comvels=expose_body_comvels,
            seed=seed)
        self.evaluate = True
