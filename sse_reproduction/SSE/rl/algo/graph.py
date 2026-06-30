import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import io
import networkx as nx
import torch
import random
from PIL import Image
import pickle
import os

class GraphPlanner:
    def __init__(self, args, low_replay, low_agent, high_agent, env):
        self.low_replay = low_replay
        self.low_agent = low_agent
        self.high_agent = high_agent
        self.env = env
        self.dim = args.subgoal_dim
        self.args = args
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.graph = None
        self.expanded_graph = None
        self.deleted_node = []
        self.init_dist = 0
        self.n_graph_node = 0
        self.cutoff = args.cutoff
        self.wp_candi = None
        self.landmarks = None
        self.states = None
        self.waypoint_vec = None
        self.waypoint_idx = 0
        self.waypoint_chase_step = 0
        self.edge_lengths = None
        self.edge_visit_counts = None
        self.initial_sample = args.initial_sample
        self.disconnected = []
        self.current = None
        self.n_succeeded_node = 0
        self.realDistWeight = self.args.real_dist_weight
        self.agentDistWeight = self.args.agent_dist_weight
        self.gridFailCluster = {} #[index] = success count, fail count
        self.tryCluster = {}
        self.first_success = False
        random.seed(self.args.seed)

    def check_reachability(self, sub_goal):
        s_position = (sub_goal+self.args.offset)//self.args.grid_size
        index = s_position[1]*(self.args.map_size/self.args.grid_size) + s_position[0]
        if index in self.tryCluster and self.tryCluster[index] > 0:
            if index in self.gridFailCluster:
                return True
            else:
                return False
        else:
            return True
    def check_reachability_3D(self, sub_goal):
        offset = self.args.offset
        column = self.args.map_size/self.args.grid_size
        depth = self.args.map_size/self.args.grid_size
        s_position = (sub_goal+offset)//self.args.grid_size
        index = s_position[2]*column*depth + s_position[1]*column + s_position[0]
        if index in self.tryCluster and self.tryCluster[index] > 0:
            if index in self.gridFailCluster:
                return True
            else:
                return False
        else:
            return True
        
    def failClusterPush(self, index, initial_point, bg):
        if index in self.gridFailCluster:
            if self.args.detour:
                if self.detour_check(index, initial_point, bg) and self.first_success:
                    self.gridFailCluster[index][1] +=1
            else:
                self.gridFailCluster[index][1] +=1
    
    def successClusterPush(self,index):
        if index in self.gridFailCluster:
            self.gridFailCluster[index][0] +=1
        else:
            self.gridFailCluster[index] = [1, 0]
        self.tryClusterPush(index)
    def tryClusterPush(self, index):
        if index in self.tryCluster:
            self.tryCluster[index] +=1
        else:
            self.tryCluster[index] = 1

    def fps_selection(
            self,
            landmarks,
            states,
            n_select: int,
            inf_value=1e6,
            low_bound_epsilon=10, early_stop=True, oracle=False
    ):
        n_states = landmarks.shape[0]
        dists = np.zeros(n_states) + inf_value
        chosen = []
        while len(chosen) < n_select:
            if (np.max(dists) < low_bound_epsilon) and early_stop and (len(chosen) > self.args.n_graph_node/10):
                break
            idx = np.argmax(dists)  # farthest point idx
            farthest_state = states[idx]
            chosen.append(idx)
            # distance from the chosen point to all other pts
            if self.args.use_oracle_G or oracle:
                new_dists = self._get_dist_from_start_oracle(farthest_state, landmarks)
            else:
                new_dists = self.low_agent._get_dist_from_start(farthest_state, landmarks)
            new_dists[idx] = 0
            dists = np.minimum(dists, new_dists)
        return chosen
        
    def graph_construct(self, iter):
        self.current = None
        self.init_dist = self.args.init_dist
        if self.graph is None:
            if 'Reacher3D' in self.env.env_name:
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-1.0, 1.1, 0.4)
                y = np.arange(-1.0, 1.1, 0.4)
                z = np.arange(-1.0, 1.1, 0.4)
                X,Y,Z = np.meshgrid(x, y, z)
                self.landmarks = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 0.5:
                                self.graph.add_edge(i, j, weight = 1.)
                nx.set_node_attributes(self.graph, 0.4, 'distance')                 
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')
                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.05:
                            self.graph.nodes[j]['success_count'] += 1       
                self.generate_novelty_grid_3D() ### need to check
                return self.landmarks, self.states
            elif self.env.env_name == 'AntMaze':
                self.xmin = -8
                self.xmax = 24
                self.ymin = -8
                self.ymax = 24
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-5, 22, 2.0)
                y = np.arange(-5, 22, 2.0)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                
                # random
                self.states = np.zeros((self.landmarks.shape[0], 29))
                random_state = replay_data['ob'][0,2:29]
                self.states[:,2:29] = random_state
                self.states[:,:2] = self.landmarks
                
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    cnt = 0
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                self.graph.add_edge(i, j, weight = 2.)
                                cnt += 1
                    if cnt == 0:
                        self.graph.add_node(i)
                        
                nx.set_node_attributes(self.graph, 2., 'distance')
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')
                
                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1         
                for i in range(self.n_graph_node):
                    if self.graph.nodes[i]['success_count'] > 0:
                        self.n_succeeded_node += 1
                self.generate_novelty_grid()        
                return self.landmarks, self.states
            
            elif self.env.env_name == 'AntMazeShortcutDetour':
                self.xmin = -4
                self.xmax = 24
                self.ymin = -4
                self.ymax = 16
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-2, 22.1, 2.0)
                y = np.arange(-2, 14.1, 2.0)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))

                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                self.graph.add_edge(i, j, weight = 2.)

                nx.set_node_attributes(self.graph, 2., 'distance')
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')

                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1
                self.generate_novelty_grid()
                return self.landmarks, self.states

            elif self.env.env_name == 'AntMazeBottleneck':
                self.xmin = -8
                self.xmax = 24
                self.ymin = -8
                self.ymax = 24
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-5, 22, 2.0)
                y = np.arange(-5, 22, 2.0)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                self.graph.add_edge(i, j, weight = 2.)
                                
                nx.set_node_attributes(self.graph, 2., 'distance')
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')
                
                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1     
                self.generate_novelty_grid()            
                return self.landmarks, self.states
            
            elif self.env.env_name == 'AntMazeDoubleBottleneck':
                self.xmin = -8
                self.xmax = 24
                self.ymin = -8
                self.ymax = 40
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-5, 22, 2.0)
                y = np.arange(-5, 38, 2.0)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                self.graph.add_edge(i, j, weight = 2.)
                                
                nx.set_node_attributes(self.graph, 2., 'distance')
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')
                
                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1  
                self.generate_novelty_grid()            
                return self.landmarks, self.states
            
            elif self.env.env_name == 'AntMazeP':
                self.xmin = -13.5
                self.xmax = 29.5
                self.ymin = -5.5
                self.ymax = 37.5
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-13, 30, 2.0)
                y = np.arange(-5, 38, 2.0)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                self.graph.add_edge(i, j, weight = 2.)
                                
                nx.set_node_attributes(self.graph, 2., 'distance')
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')
                
                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1     
                self.generate_novelty_grid()
                return self.landmarks, self.states
            elif self.env.env_name == 'AntMazeComplex-v0':
                self.xmin = -5.5
                self.xmax = 54.5
                self.ymin = -5.5
                self.ymax = 54.5
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-5, 54, 2.0)
                y = np.arange(-5, 54, 2.0)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                self.graph.add_edge(i, j, weight = 2.)
                                
                nx.set_node_attributes(self.graph, 2., 'distance')
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')
                
                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1      
                self.generate_novelty_grid()
                return self.landmarks, self.states
            elif self.env.env_name == 'AntPush':
                self.xmin = -16
                self.xmax = 16
                self.ymin = -8
                self.ymax = 24
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-13, 13, 2.0)
                y = np.arange(-5, 21, 2.0)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                self.graph.add_edge(i, j, weight = 2.)
                                
                nx.set_node_attributes(self.graph, 2., 'distance')
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')
                
                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1     
                self.generate_novelty_grid()            
                return self.landmarks, self.states
            elif self.env.env_name == 'AntMazeKeyChest':
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-5.0, 38.0, 2.0)
                y = np.arange(-5.0, 38.0, 2.0)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                self.graph.add_edge(i, j, weight = 2.)

                nx.set_node_attributes(self.graph, 2., 'distance')                
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')
                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1        
                self.generate_novelty_grid()
                return self.landmarks, self.states
            elif self.env.env_name == 'AntMazeDoubleKeyChest':
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                x = np.arange(-5.0, 38.0, 2.0)
                y = np.arange(-5.0, 38.0, 2.0)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    for j in range(self.n_graph_node):
                        if i != j:
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                self.graph.add_edge(i, j, weight = 2.)

                nx.set_node_attributes(self.graph, 2., 'distance')                
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')

                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1       
                self.generate_novelty_grid()
                return self.landmarks, self.states
            return self.landmarks, self.states

    def generate_novelty_grid(self):
        for i in self.landmarks:
            s_position = (i+self.args.offset)//self.args.grid_size
            index = s_position[1]*(self.args.map_size/self.args.grid_size) + s_position[0]
            self.tryCluster[index] = 0
    def generate_novelty_grid_3D(self):
        offset = 1
        column = self.args.map_size/self.args.grid_size
        depth = self.args.map_size/self.args.grid_size
        for i in self.landmarks:
            s_position = (i+offset)//self.args.grid_size
            index = s_position[2]*column*depth + s_position[1]*column + s_position[0]
            self.tryCluster[index] = 0


    def expand_node(self, anchor, dist):
        candi_list = []
        
        if self.dim == 2:
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if i % 2 == 0 and j % 2 == 0:
                        continue
                    else:
                        if anchor[0]+i*dist <= self.xmin:
                            continue
                        if anchor[0]+i*dist >= self.xmax:
                            continue
                        if anchor[1]+j*dist <= self.ymin:
                            continue
                        if anchor[1]+j*dist >= self.ymax:
                            continue
                        candi_list.append([anchor[0]+i*dist, anchor[1]+j*dist, dist])
        elif self.dim == 3:
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        if i % 2 == 0 and j % 2 == 0 and k % 2 == 0:
                            continue
                        else:
                            candi_list.append([anchor[0]+i*dist, anchor[1]+j*dist, anchor[2]+k*dist, dist])
    
        return np.array(candi_list)
            
    def expand(self):
        edges = self.graph.edges(data=True)
        node_type = np.zeros(self.landmarks.shape[0])
        # 0 : untried, 1 : success, 2 : failed
        for i in range(self.landmarks.shape[0]):
            if self.graph.nodes[i]['attempt_count'] == 0:
                node_type[i] = 0
            elif self.graph.nodes[i]['success_count'] != 0:
                node_type[i] = 1
            else:
                node_type[i] = 2
        for edge in edges:
            if (node_type[edge[0]] + node_type[edge[1]] == 1):
                return False
        
        distance_reduce = np.zeros(self.landmarks.shape[0])
        removed_edges = []
        
        for edge in edges:
            if (node_type[edge[0]] == 2) or (node_type[edge[1]] == 2):
                distance_reduce[edge[0]] = 1
                distance_reduce[edge[1]] = 1
                
        for edge in edges:
            if (distance_reduce[edge[0]]) and (distance_reduce[edge[1]]):
                removed_edges.append((edge[0], edge[1]))
                
        for edge in removed_edges:
            self.graph.remove_edge(edge[0], edge[1])
        landmark_candi = []
        for i in range(self.landmarks.shape[0]):
            if distance_reduce[i] == 1:
                self.graph.nodes[i]['distance'] = self.graph.nodes[i]['distance'] / 2.
            if node_type[i] == 2:
                anchor = self.landmarks[i]
                dist = self.graph.nodes[i]['distance']
                self.graph.nodes[i]['attempt_count'] = 0
                candi_list = self.expand_node(anchor, dist)
                if len(landmark_candi) == 0:
                    landmark_candi = candi_list
                else:
                    landmark_candi = np.concatenate((landmark_candi, candi_list))
        
        if len(landmark_candi) == 0:
            return False

        landmark_candi = np.unique(landmark_candi, axis = 0)
        n = self.n_graph_node
        self.landmarks = np.concatenate((self.landmarks, landmark_candi[:,:self.args.subgoal_dim]))
        self.n_graph_node = self.landmarks.shape[0]
        
        for i in range(n, self.n_graph_node):
            dist_ = landmark_candi[i-n, 2]
            self.graph.add_node(i)
            self.graph.nodes[i]['attempt_count'] = 0
            self.graph.nodes[i]['success_count'] = 0
            self.graph.nodes[i]['distance'] = dist_
            for j in range(i):
                dist = np.linalg.norm(self.landmarks[i]-self.landmarks[j])
                threshold = np.max([dist_, self.graph.nodes[j]['distance']])
                if (dist <= threshold * 1.01):
                    self.graph.add_edge(i, j, weight = dist)#threshold)
                    self.graph.add_edge(j, i, weight = dist)#threshold)
        
        self.disconnected = []
        
        return True
    
    def dense(self, dg):
        edges = self.graph.edges(data=True)
        
        remove_edges = []
        for edge in edges:
            if edge[0] == dg or edge[1] == dg:
                remove_edges.append(edge)
                
        for edge in remove_edges:
            self.graph.remove_edge(*edge[:2])
        
        self.graph.nodes[dg]['attempt_count'] = 0
        self.graph.nodes[dg]['success_count'] = 0
        dist = self.graph.nodes[dg]['distance'] / 2.
        self.graph.nodes[dg]['distance'] = dist
        
        for i in range(-2, 3):
            for j in range(-2, 3):
                exist = False
                candi = np.array([0., 0.])
                candi[0] = self.landmarks[dg][0] + dist * i
                candi[1] = self.landmarks[dg][1] + dist * j
                for k in range(self.n_graph_node):
                    if np.linalg.norm(self.landmarks[k] - candi) < 0.01:
                        exist = True
                        self.graph.nodes[k]['attempt_count'] = 0
                        self.graph.nodes[k]['success_count'] = 0
                        for l in range(self.n_graph_node):
                            if l != k:
                                d = np.max([dist, self.graph.nodes[l]['distance']])
                                if np.linalg.norm(self.landmarks[k] - self.landmarks[l]) < 1.01 * d:
                                   
                                    if self.graph.has_edge(k, l):
                                        pdist = self.get_pdist(self.landmarks[k], self.landmarks[l])
                                        self.graph[k][l]['weight'] = pdist[0]
                                        self.graph[l][k]['weight'] = pdist[1]
                                        self.graph.nodes[k]['distance'] = d
                        if k in self.disconnected:
                            self.disconnected.remove(k)
                        
                if not exist:
                    candi = np.expand_dims(candi, axis = 0)
                    self.landmarks = np.concatenate((self.landmarks, candi))
                    self.graph.add_node(self.n_graph_node)
                    self.graph.nodes[self.n_graph_node]['attempt_count'] = 0
                    self.graph.nodes[self.n_graph_node]['success_count'] = 0
                    self.graph.nodes[self.n_graph_node]['distance'] = dist
                    for m in range(self.n_graph_node):
                        d = np.max([dist,self.graph.nodes[m]['distance']])
                        if np.linalg.norm(self.landmarks[self.n_graph_node] - self.landmarks[m]) < 1.01 * d:
                            if((self.landmarks[self.n_graph_node][0] == self.landmarks[m][0]) or (self.landmarks[self.n_graph_node][1] == self.landmarks[m][1])):
                                pdist = self.get_pdist(self.landmarks[m], self.landmarks[self.n_graph_node])
                                self.graph.add_edge(m, self.n_graph_node, weight = pdist[0])
                                self.graph.add_edge(self.n_graph_node, m, weight = pdist[1])
                    
                    self.n_graph_node += 1

    def dense_3D(self, dg):
        edges = self.graph.edges(data=True)
        
        remove_edges = []
        for edge in edges:
            if edge[0] == dg or edge[1] == dg:
                remove_edges.append(edge)
                
        for edge in remove_edges:
            self.graph.remove_edge(*edge[:2])
        
        self.graph.nodes[dg]['attempt_count'] = 0
        self.graph.nodes[dg]['success_count'] = 0
        dist = self.graph.nodes[dg]['distance'] / 2.
        self.graph.nodes[dg]['distance'] = dist
        
        for i in range(-2, 3):
            for j in range(-2, 3):
                for m in range(-2, 3):
                    exist = False
                    candi = np.array([0., 0., 0.])
                    candi[0] = self.landmarks[dg][0] + dist * i
                    candi[1] = self.landmarks[dg][1] + dist * j
                    candi[2] = self.landmarks[dg][2] + dist * m
                    for k in range(self.n_graph_node):
                        if np.linalg.norm(self.landmarks[k] - candi) < 0.01:
                            exist = True
                            self.graph.nodes[k]['attempt_count'] = 0
                            self.graph.nodes[k]['success_count'] = 0
                            for l in range(self.n_graph_node):
                                if l != k:
                                    d = np.max([dist, self.graph.nodes[l]['distance']])
                                    if np.linalg.norm(self.landmarks[k] - self.landmarks[l]) < 1.01 * d:
                                    
                                        if self.graph.has_edge(k, l):
                                            pdist = self.get_pdist_3D(self.landmarks[k], self.landmarks[l])
                                            self.graph[k][l]['weight'] = pdist[0]
                                            self.graph[l][k]['weight'] = pdist[1]
                                            self.graph.nodes[k]['distance'] = d
                            if k in self.disconnected:
                                self.disconnected.remove(k)
                            
                    if not exist:
                        candi = np.expand_dims(candi, axis = 0)
                        self.landmarks = np.concatenate((self.landmarks, candi))
                        self.graph.add_node(self.n_graph_node)
                        self.graph.nodes[self.n_graph_node]['attempt_count'] = 0
                        self.graph.nodes[self.n_graph_node]['success_count'] = 0
                        self.graph.nodes[self.n_graph_node]['distance'] = dist
                        for m in range(self.n_graph_node):
                            d = np.max([dist,self.graph.nodes[m]['distance']])
                            if np.linalg.norm(self.landmarks[self.n_graph_node] - self.landmarks[m]) < 1.01 * d:
                                if((self.landmarks[self.n_graph_node][0] == self.landmarks[m][0]) or (self.landmarks[self.n_graph_node][1] == self.landmarks[m][1])):
                                    pdist = self.get_pdist_3D(self.landmarks[m], self.landmarks[self.n_graph_node])
                                    self.graph.add_edge(m, self.n_graph_node, weight = pdist[0])
                                    self.graph.add_edge(self.n_graph_node, m, weight = pdist[1])
                        self.n_graph_node += 1
                    
    
    def densify(self):
        failed = []
        cnt = 0
        for i in range(self.n_graph_node):
            if(self.graph.nodes[i]['success_count'] > 0):
                cnt += 1
        for i in range(self.n_graph_node):
            if((self.graph.nodes[i]['attempt_count'] > self.args.fail_count) and (self.graph.nodes[i]['success_count'] == 0)):
                failed.append(self.graph.nodes[i]['distance'])
            else:
                failed.append(0)
                
        failed = np.array(failed)
        max_dist = np.max(failed)
        
        if cnt > self.n_succeeded_node:
            self.n_succeeded_node = cnt
            return
        if max_dist == 0:
            return
        candi = np.where(failed == max_dist)
        
        if self.dim == 3:
            self.dense_3D(candi[0][random.choices(range(len(candi[0])))][0])
        else:
            self.dense(candi[0][random.choices(range(len(candi[0])))][0])
        
        return 
    
    def get_pdist(self, ag,bg):
        real_dist = np.linalg.norm(ag - bg)
        sample_ob1 = self.low_replay.sample_search_batch(ag)
        agent_dist = self.low_agent._get_point_to_point(sample_ob1, bg)
        weight_sum_dist1 = (self.agentDistWeight * agent_dist + self.realDistWeight * real_dist)
        sample_ob2 = self.low_replay.sample_search_batch(bg)
        agent_dist = self.low_agent._get_point_to_point(sample_ob2, ag)
        weight_sum_dist2 = (self.agentDistWeight * agent_dist + self.realDistWeight * real_dist)
        return [weight_sum_dist1,weight_sum_dist2]

    def get_pdist_3D(self, ag,bg):
        real_dist = np.linalg.norm(ag - bg)
        sample_ob1 = self.low_replay.sample_search_batch(ag)
        agent_dist = self.low_agent._get_point_to_point(sample_ob1, bg)
        weight_sum_dist1 = (self.agentDistWeight * agent_dist + self.realDistWeight * real_dist)
        sample_ob2 = self.low_replay.sample_search_batch(bg)
        agent_dist = self.low_agent._get_point_to_point(sample_ob2, ag)
        weight_sum_dist2 = (self.agentDistWeight * agent_dist + self.realDistWeight * real_dist)
        return [weight_sum_dist1,weight_sum_dist2]

    def get_fail_percentage(self, bg):
        s_position = (bg+self.args.offset)//self.args.grid_size
        index = s_position[1]*(self.args.map_size/self.args.grid_size) + s_position[0]
        if index in self.gridFailCluster:
            return self.gridFailCluster[index][1]/(self.gridFailCluster[index][0]+self.gridFailCluster[index][1])
        else:
            return 0.0
        
    def get_fail_percentage_3D(self, bg):
        offset = 1
        column = self.args.map_size/self.args.grid_size
        depth = self.args.map_size/self.args.grid_size
        s_position = (bg+offset)//self.args.grid_size
        index = s_position[2]*column*depth + s_position[1]*column + s_position[0]
        if index in self.gridFailCluster:
            return self.gridFailCluster[index][1]/(self.gridFailCluster[index][0]+self.gridFailCluster[index][1])
        else:
            return 0.0
    def find_path(self, ob, subgoal, ag, bg, inf_value=1e6, train = False, first = False, fail_ratio=False, freeze=False):
        expanded_graph = self.graph.copy()
        self.edge_lengths = []
        subgoal = subgoal[:self.dim]    
        self.wp_candi = None
        if first:
            self.deleted_node = []
        if self.graph is not None and not freeze:
            if self.expand():
                expanded_graph = self.graph.copy()
        if self.deleted_node:
            for i in self.deleted_node:
                for j in range(self.n_graph_node):
                    if i != j:
                        threshold = np.max([expanded_graph.nodes[i]['distance'], expanded_graph.nodes[j]['distance']])
                        if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) < threshold * 1.01:
                            if expanded_graph.has_edge(i, j):
                                expanded_graph[i][j]['weight'] = 1000.
                                expanded_graph[j][i]['weight'] = 1000.
        
        start_to_goal_length = np.linalg.norm(ag - subgoal)
        if start_to_goal_length < self.args.graph_threshold:
            expanded_graph.add_edge('start', 'goal', weight = 1.)
            
        start_edge_length = self.dist_to_graph(ag, self.landmarks)
        goal_edge_length = self.dist_to_graph(subgoal, self.landmarks)
        
        self.edge_lengths = [] 
        
        for i in range(self.n_graph_node):
            if start_edge_length[i] < self.args.graph_threshold:
                if i not in self.disconnected:
                    expanded_graph.add_edge('start', i, weight = 1.)
                else:
                    expanded_graph.add_edge('start', i, weight = 1000.)
            if goal_edge_length[i] < self.args.graph_threshold:
                if i not in self.disconnected:
                    expanded_graph.add_edge(i, 'goal', weight = 1.)
                else:
                    expanded_graph.add_edge(i, 'goal', weight = 1000.)
        if (not expanded_graph.has_node('start')):
            added = False
            adjusted = 1.5
            while True:
                adjusted_cutoff = 2.0 * adjusted
                for i in range(self.n_graph_node):
                    if(start_edge_length[i] < adjusted_cutoff):
                        if i not in self.disconnected:
                            expanded_graph.add_edge('start', i, weight = 1.)
                            added = True
                if added:
                    break
                adjusted += 0.5           
        
        if(not expanded_graph.has_node('goal')):
            adjusted_cutoff = 2.0 * 2.0
            for i in range(self.n_graph_node):
                if(goal_edge_length[i] < adjusted_cutoff):
                    if i not in self.disconnected:
                        expanded_graph.add_edge(i, 'goal', weight = 1.)
        
        if(not expanded_graph.has_node('goal')) or (not nx.has_path(expanded_graph, 'start', 'goal')):
            while True:
                nearestnode = np.argmin(goal_edge_length) #nearest point from the goal
                if goal_edge_length[nearestnode] > start_to_goal_length:
                    expanded_graph.add_edge('start', 'goal', weight = 1.)
                    break
                if(expanded_graph.has_node(nearestnode)) and (nx.has_path(expanded_graph, 'start', nearestnode)):
                    expanded_graph.add_edge(nearestnode, 'goal', weight = goal_edge_length[nearestnode])
                    break
                else:
                    goal_edge_length[nearestnode] = inf_value
        if fail_ratio:
            expanded_graph = self.apply_fail_percentage(expanded_graph)   
        path = nx.shortest_path(expanded_graph, 'start', 'goal', weight='weight')
        for (i, j) in zip(path[:-1], path[1:]):
            self.edge_lengths.append(expanded_graph[i][j]['weight'])
            
        self.waypoint_vec = list(path)[1:-1]
        self.waypoint_vec_origin = self.waypoint_vec.copy()
        self.waypoint_idx = 0
        self.waypoint_chase_step = 0
        self.wp_candi = subgoal
        self.expanded_graph = expanded_graph.copy()
        
        return self.wp_candi


    def apply_fail_percentage(self, graph):
        for i in range(self.n_graph_node):
            for j in range(self.n_graph_node):
                if graph.has_edge(i, j):
                    weight = graph[i][j]['weight']
                    if self.dim == 3:
                        graph[i][j]['weight'] = weight * (max(self.get_fail_percentage(self.landmarks[j])*self.args.fail_weight, 1))
                    else:
                        graph[i][j]['weight'] = weight * (max(self.get_fail_percentage(self.landmarks[j])*self.args.fail_weight, 1))
        return graph
    
    def detour_check(self, dindex,ag,subgoal):
        offset = self.args.offset
        column = self.args.map_size/self.args.grid_size
        depth = self.args.map_size/self.args.grid_size

        expanded_graph = self.graph.copy()
        if self.deleted_node:
            for i in self.deleted_node:
                for j in range(self.n_graph_node):
                    if i != j:
                        threshold = np.max([expanded_graph.nodes[i]['distance'], expanded_graph.nodes[j]['distance']])
                        if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) < threshold * 1.01:
                            if expanded_graph.has_edge(i, j):
                                expanded_graph[i][j]['weight'] = 1000.
                                expanded_graph[j][i]['weight'] = 1000.
        
        start_to_goal_length = np.linalg.norm(ag - subgoal)
        if start_to_goal_length < self.args.graph_threshold:
            expanded_graph.add_edge('start', 'goal', weight = 1.)
            
        start_edge_length = self.dist_to_graph(ag, self.landmarks)
        goal_edge_length = self.dist_to_graph(subgoal, self.landmarks)
        for i in range(self.n_graph_node):
            if start_edge_length[i] < self.args.graph_threshold:
                if i not in self.disconnected:
                    expanded_graph.add_edge('start', i, weight = 1.)
                else:
                    expanded_graph.add_edge('start', i, weight = 1000.)
            if goal_edge_length[i] < self.args.graph_threshold:
                if i not in self.disconnected:
                    expanded_graph.add_edge(i, 'goal', weight = 1.)
                else:
                    expanded_graph.add_edge(i, 'goal', weight = 1000.)
        if (not expanded_graph.has_node('start')):
            added = False
            adjusted = 1.5
            while True:
                adjusted_cutoff = 2.0 * adjusted
                for i in range(self.n_graph_node):
                    if(start_edge_length[i] < adjusted_cutoff):
                        if i not in self.disconnected:
                            expanded_graph.add_edge('start', i, weight = 1.)
                            added = True
                if added:
                    break
                adjusted += 0.5           
        
        if(not expanded_graph.has_node('goal')):
            adjusted_cutoff = 2.0 * 2.0
            for i in range(self.n_graph_node):
                if(goal_edge_length[i] < adjusted_cutoff):
                    if i not in self.disconnected:
                        expanded_graph.add_edge(i, 'goal', weight = 1.)
        
        for i in range(self.n_graph_node):
            if self.dim==3:
                s_position = (self.landmarks[i]+offset)//self.args.grid_size
                index = s_position[2]*column*depth + s_position[1]*column + s_position[0]
            else:
                s_position = (self.landmarks[i]+self.args.offset)//self.args.grid_size
                index = s_position[1]*(self.args.map_size/self.args.grid_size) + s_position[0]
            if dindex ==index:
                expanded_graph.remove_node(i)
        if nx.has_path(expanded_graph, 'start', 'goal') and nx.shortest_path_length(expanded_graph, 'start', 'goal') < 1000:
            return True
        else:
            return False

    def set_success(self):
        offset = 1
        column = self.args.map_size/self.args.grid_size
        depth = self.args.map_size/self.args.grid_size
        if self.graph is not None:
            for i in self.waypoint_vec[self.waypoint_idx:]:
                self.graph.nodes[i]['attempt_count'] += 1
                self.graph.nodes[i]['success_count'] += 1
                if self.dim==3:
                    s_position = (self.landmarks[i]+offset)//self.args.grid_size
                    index = s_position[2]*column*depth + s_position[1]*column + s_position[0]
                else:
                    s_position = (self.landmarks[i]+self.args.offset)//self.args.grid_size
                    index = s_position[1]*(self.args.map_size/self.args.grid_size) + s_position[0]
                self.successClusterPush(index=index)

    

    def dist_to_graph(self, node, landmarks):
        return np.linalg.norm(node[:self.dim]-landmarks, axis = 1)
            
    def select_novel_goal(self, ag):
        unique_counts = sorted(set(self.tryCluster.values()))
        count_index = 0 
        while count_index < len(unique_counts):
            min_try_count = unique_counts[count_index]
            candidate_grids = np.array([node for node, count in self.tryCluster.items() if count == min_try_count])

            landmarks_array = np.array(self.landmarks)

            s_positions = (landmarks_array + self.args.offset) // self.args.grid_size
            indices = (s_positions[:, 1] * (self.args.map_size // self.args.grid_size) + s_positions[:, 0]).astype(int)

            mask = np.isin(indices, candidate_grids)
            candidate_landmarks = landmarks_array[mask]
            candidate_indices = indices[mask]

            if len(candidate_landmarks) > 0:  
                break

            count_index += 1

        if len(candidate_landmarks) == 0:  
            return None
        unique_grids = np.unique(candidate_indices)
        mean_positions = np.array([candidate_landmarks[candidate_indices == grid].mean(axis=0) for grid in unique_grids])

        closest_grid = unique_grids[np.argmin(np.linalg.norm(mean_positions - ag, axis=1))]

        selected_node = random.choice(candidate_landmarks[candidate_indices == closest_grid])
        return selected_node  

    def select_novel_goal_3D(self, ag):
        offset = self.args.offset
        column = self.args.map_size/self.args.grid_size
        depth = self.args.map_size/self.args.grid_size
        unique_counts = sorted(set(self.tryCluster.values()))
        count_index = 0  
        while count_index < len(unique_counts):
            min_try_count = unique_counts[count_index]
            candidate_grids = np.array([node for node, count in self.tryCluster.items() if count == min_try_count])

            landmarks_array = np.array(self.landmarks)

            s_positions = (landmarks_array + offset) // self.args.grid_size
            indices = (s_positions[:,2]*depth*column+ s_positions[:, 1] * column + s_positions[:, 0]).astype(int)

            mask = np.isin(indices, candidate_grids)
            candidate_landmarks = landmarks_array[mask]
            candidate_indices = indices[mask]

            if len(candidate_landmarks) > 0:  
                break

            count_index += 1

        if len(candidate_landmarks) == 0:  
            return None
        unique_grids = np.unique(candidate_indices)
        mean_positions = np.array([candidate_landmarks[candidate_indices == grid].mean(axis=0) for grid in unique_grids])

        closest_grid = unique_grids[np.argmin(np.linalg.norm(mean_positions - ag, axis=1))]

        selected_node = random.choice(candidate_landmarks[candidate_indices == closest_grid])
        return selected_node 

    def get_waypoint(self, ob, ag, subgoal, bg, train=False, freeze=False):
        if self.graph is not None:
            self.waypoint_chase_step += 1
            if(self.waypoint_idx >= len(self.waypoint_vec)):
                waypoint_subgoal = subgoal
            else:
                i = self.waypoint_vec[self.waypoint_idx]
            
                if((np.linalg.norm(ag[:self.dim]-self.landmarks[i][:self.dim]) < 0.5)):

                    if train:
                        self.graph.nodes[i]['attempt_count'] += 1
                        self.graph.nodes[i]['success_count'] += 1
                    
                    self.waypoint_idx += 1
                    self.waypoint_chase_step = 0
                    
                elif((self.waypoint_chase_step > 100.)):
                    if train:
                        self.graph.nodes[i]['attempt_count'] += 1
                    if self.graph.nodes[i]['success_count'] == 0:
                        if train:
                            if self.graph.nodes[i]['attempt_count'] > self.args.fail_count and not freeze:
                                self.disconnected.append(i)
                                for j in range(self.n_graph_node):
                                    if i != j:
                                        if self.graph.has_edge(i, j):
                                            self.graph[i][j]['weight'] = 1000.
                                            self.graph[j][i]['weight'] = 1000.
                                                
                                self.find_path(ob, subgoal, ag, bg, freeze=freeze)
                            else:
                                if not freeze:
                                    self.deleted_node.append(i)
                                self.find_path(ob, subgoal, ag, bg, freeze=freeze)
                        else:
                            if not freeze:
                                self.deleted_node.append(i)
                            self.find_path(ob, subgoal, ag, bg, freeze=freeze)
                    else:
                        if not freeze:
                            self.deleted_node.append(i)
                        self.find_path(ob, subgoal, ag, bg, freeze=freeze)
                        
                if(self.waypoint_idx >= len(self.waypoint_vec)):
                    waypoint_subgoal = subgoal
                else:
                    waypoint_subgoal = self.landmarks[self.waypoint_vec[self.waypoint_idx]][:self.dim]
        else:
            waypoint_subgoal = subgoal
        return waypoint_subgoal


    def calc_coverage(self):
        if self.graph is None:
            return 0.0
        success_count=0
        total_count=0
        if len(self.tryCluster) == 0:
            return 0.0
        for i in self.tryCluster.keys():
            if i in self.gridFailCluster and self.gridFailCluster[i][0]>0:
                success_count+=1
            total_count+=1
                
        return success_count/total_count


    #####################oracle graph#########################
    def _get_dist_to_goal_oracle(self, obs_tensor, goal):
        goal_repeat = np.ones_like(obs_tensor[:, :self.args.subgoal_dim]) \
            * np.expand_dims(goal[:self.args.subgoal_dim], axis=0)
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - goal_repeat, axis=1)
        return dist

    def _get_dist_from_start_oracle(self, start, obs_tensor):
        start_repeat = np.ones((obs_tensor.shape[0], np.squeeze(start).shape[0])) * np.expand_dims(start, axis=0)
        # start_repeat = np.ones_like(obs_tensor) * np.expand_dims(start, axis=0)
        start_repeat = start_repeat[:, :self.args.subgoal_dim]
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - start_repeat, axis=1)
        return dist

    def _get_point_to_point_oracle(self, point1, point2):
        point1 = point1[:self.args.subgoal_dim]
        point2 = point2[:self.args.subgoal_dim]
        dist = np.linalg.norm(point1-point2)
        return dist

    def _get_pairwise_dist_oracle(self, obs_tensor):
        goal_tensor = obs_tensor
        dist_matrix = []
        for obs_index in range(obs_tensor.shape[0]):
            obs = obs_tensor[obs_index]
            obs_repeat_tensor = np.ones_like(goal_tensor) * np.expand_dims(obs, axis=0)
            dist = np.linalg.norm(obs_repeat_tensor[:, :self.args.subgoal_dim] - goal_tensor[:, :self.args.subgoal_dim], axis=1)
            dist_matrix.append(np.squeeze(dist))
        pairwise_dist = np.array(dist_matrix) #pairwise_dist[i][j] is dist from i to j
        return pairwise_dist
    

    ######## Save and load graph component
    def generate_dict(self):
        saveDict = {}
        saveDict['landmarks'] = self.landmarks
        saveDict['edge_visit_counts'] = self.edge_visit_counts
        saveDict['n_graph_node'] = self.n_graph_node
        saveDict['n_succeeded_node'] = self.n_succeeded_node
        saveDict['xmin'] = self.xmin
        saveDict['xmax'] = self.xmax
        saveDict['ymin'] = self.ymin
        saveDict['ymax'] = self.ymax
        return saveDict
    def load_dict(self, loadDict):
        self.landmarks = loadDict['landmarks']
        self.edge_visit_counts = loadDict['edge_visit_counts']
        self.n_graph_node = loadDict['n_graph_node']
        self.n_succeeded_node = loadDict['n_succeeded_node']
        self.xmin = loadDict['xmin']
        self.xmax = loadDict['xmax']
        self.ymin = loadDict['ymin']
        self.ymax = loadDict['ymax']


    def save(self, path):
        save_path = os.path.join(path,'graph.pkl')
        save_path2 = os.path.join(path,'expanded_graph.pkl')
        save_path_dict = os.path.join(path, 'dict.pkl')
        save_path_cluster = os.path.join(path, 'failCluster.pkl')
        save_path_try = os.path.join(path, 'tryCluster.pkl')
        if self.graph is not None:
            with open(save_path, 'wb') as handle:
                pickle.dump(self.graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.expanded_graph is not None:
            with open(save_path2, 'wb') as handle:
                pickle.dump(self.expanded_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)    
            with open(save_path_dict, 'wb') as handle:
                save_dict = self.generate_dict()
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.gridFailCluster:
            with open(save_path_cluster, 'wb') as handle:
                pickle.dump(self.gridFailCluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.tryCluster:
            with open(save_path_try, 'wb') as handle:
                pickle.dump(self.tryCluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self, path):
        load_path = os.path.join(path,'graph.pkl')
        load_path2 = os.path.join(path,'expanded_graph.pkl')
        load_path_dict = os.path.join(path, 'dict.pkl')
        load_path_cluster = os.path.join(path, 'failCluster.pkl')
        load_path_try = os.path.join(path, 'tryCluster.pkl')
        with open(load_path, 'rb') as handle:
            self.graph = pickle.load(handle)
        with open(load_path2, 'rb') as handle:
            self.expanded_graph = pickle.load(handle)
        with open(load_path_dict, 'rb') as handle:
            loadDict = pickle.load(handle)
            self.load_dict(loadDict=loadDict)
        with open(load_path_cluster, 'rb') as handle:
            self.gridFailCluster = pickle.load(handle)
        with open(load_path_try, 'rb') as handle:
            self.tryCluster = pickle.load(handle)
