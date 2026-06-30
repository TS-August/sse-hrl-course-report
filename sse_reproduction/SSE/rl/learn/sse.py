import torch
from torch.optim import Adam

import os.path as osp
from rl.utils import net_utils
import numpy as np
from rl.learn.core import dict_to_numpy

class HighLearner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        low_agent = None,
        name='high_learner',
    ):
        self.agent = agent
        self.low_agent = low_agent
        self.monitor = monitor
        self.args = args
        
        self.q_optim1 = Adam(agent.critic1.parameters(), lr=args.lr_critic_high)
        self.q_optim2 = Adam(agent.critic2.parameters(), lr=args.lr_critic_high)
        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor_high)
        
        self._save_file = str(name) + '.pt'
    
    def critic_loss(self, batch):
        o, a, o2, r, bg, mask = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg'], batch['mask']
        reward_high = self.agent.to_tensor(r.flatten())
        
        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, bg, pi_target=True) + noise
            q_next = self.agent.get_qs(o2, bg, n_a, q_target=True)
            q_targ = reward_high + self.args.gamma_high * q_next *torch.FloatTensor((1-mask).flatten()).cuda(device=self.args.cuda_num)
            q_targ = torch.clamp(q_targ, -self.args.clip_return, self.args.clip_return)
        q_bg1 = self.agent.get_qs(o, bg, a, net = 1)
        q_bg2 = self.agent.get_qs(o, bg, a, net = 2)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()
        
        loss_critic = {'critic_1' : loss_q1, 'critic_2' : loss_q2}
        if torch.isnan(loss_q1).any() or torch.isnan(loss_q2).any() or torch.isinf(loss_q1).any() or torch.isinf(loss_q2).any():
            raise ValueError
        return loss_critic, q_targ
    
    def actor_loss(self, batch, logging=False):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2= batch['ag'], batch['ag2']

        a = self.agent.to_tensor(a)
        
        q_pi, pi = self.agent.forward1(o, bg) ###pi : action choosen from actor q_pi: Q Value using choosen action 
        loss_actor = (- q_pi).mean()

        if torch.isnan(loss_actor).any() or torch.isinf(loss_actor).any():
            raise ValueError

        return loss_actor, q_pi
    
    
    def update_critic(self, batch, train_embed=True, logging=False):
        loss, q_targ =self.critic_loss(batch)
        q_targ_mean = q_targ.mean()
        loss_critic1 = loss['critic_1']
        loss_critic2 = loss['critic_2']
        loss_1 = loss_critic1.item()
        loss_2 = loss_critic2.item()
        self.q_optim1.zero_grad()
        self.q_optim2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        if self.args.grad_norm_clipping_high > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), self.args.grad_norm_clipping_high).item()
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), self.args.grad_norm_clipping_high).item()  
        if self.args.grad_value_clipping_high > 0.:
            gradnorm_mean_critic1_total =net_utils.mean_grad_norm(self.agent.critic1.parameters()).item()
            gradnorm_mean_critic2_total =net_utils.mean_grad_norm(self.agent.critic2.parameters()).item()
            torch.nn.utils.clip_grad_value_(self.agent.critic1.parameters(), self.args.grad_value_clipping_high)
            torch.nn.utils.clip_grad_value_(self.agent.critic2.parameters(), self.args.grad_value_clipping_high)
        self.q_optim1.step()
        self.q_optim2.step()
        if logging:
            self.monitor.store(critic_return_high_mean = batch['normal_return'].mean())
            self.monitor.store(
                Loss_critic_1_high=loss_1,
                Loss_critic_2_high=loss_2,
            )
            if self.args.grad_norm_clipping_high > 0.:
                self.monitor.store(gradnorm_critic1_high=c_norm1)
                self.monitor.store(gradnorm_critic2_high=c_norm2)
            if self.args.grad_value_clipping_high > 0.:
                self.monitor.store(gradnorm_mean_critic1_high=gradnorm_mean_critic1_total)
                self.monitor.store(gradnorm_mean_critic2_high=gradnorm_mean_critic2_total)
            monitor_log = dict(
                q_targ_high=q_targ_mean,
            )
            self.monitor.store(**dict_to_numpy(monitor_log))
        
        

    def update_actor(self, batch, train_embed=True, logging=False):
            
        loss_actor, q_pi = self.actor_loss(batch, logging=logging)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        if self.args.grad_norm_clipping_high > 0.:
            a_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.grad_norm_clipping_high).item()
        if self.args.grad_value_clipping_high > 0.:
            gradnorm_mean = net_utils.mean_grad_norm(self.agent.actor.parameters()).item()
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.args.grad_value_clipping_high)
        self.pi_optim.step()
        if logging:
            self.monitor.store(actor_return_high_mean = batch['normal_return'].mean())
            self.monitor.store(
                Loss_actor_high=loss_actor.item(),
            )
            if self.args.grad_norm_clipping_high > 0.:
                self.monitor.store(gradnorm_actor_high=a_norm)
            if self.args.grad_value_clipping_high > 0.:
                self.monitor.store(gradnorm_mean_actor_high=gradnorm_mean)
            monitor_log = dict(q_pi_high=q_pi.mean())
            self.monitor.store(**dict_to_numpy(monitor_log))
    def target_update(self):
        self.agent.target_update()
    
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def state_dict(self):
        return dict(
            q_optim1=self.q_optim1.state_dict(),
            q_optim2=self.q_optim2.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim1.load_state_dict(state_dict['q_optim1'])
        self.q_optim2.load_state_dict(state_dict['q_optim2'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
    
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


class LowLearner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        name='low_learner',
    ):
        self.agent = agent
        self.monitor = monitor
        self.args = args
        
        self.q_optim1 = Adam(agent.critic1.parameters(), lr=args.lr_critic_low)
        self.q_optim2 = Adam(agent.critic2.parameters(), lr=args.lr_critic_low)
        self.q_optim1_g = Adam(agent.critic1_g.parameters(), lr=args.lr_critic_low)
        self.q_optim2_g = Adam(agent.critic2_g.parameters(), lr=args.lr_critic_low)
        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor_low)
        
        self._save_file = str(name) + '.pt'
    
    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05 
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, bg, pi_target=True) + noise
            q_next = self.agent.get_qs(o2, bg, n_a, q_target=True)
            q_targ = r + self.args.gamma_low * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)
        
        q_bg1 = self.agent.get_qs(o, bg, a, net = 1)
        q_bg2 = self.agent.get_qs(o, bg, a, net = 2)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()
        
    
        loss_critic = {'critic_1' : loss_q1, 'critic_2' : loss_q2}
        
        return loss_critic, q_targ

    def critic_loss_g(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())

        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05 
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, bg, pi_target=True) + noise
            q_next = self.agent.get_qs_g(o2, bg, n_a, q_target=True)
            q_targ = r + self.args.gamma_low * q_next
            q_targ = torch.clamp(q_targ, max = 0.0)

        q_bg1 = self.agent.get_qs_g(o, bg, a, net = 1)
        q_bg2 = self.agent.get_qs_g(o, bg, a, net = 2)
        q_bg = self.agent.get_qs_g(o, bg, a)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()


        loss_critic = {'critic_1_g' : loss_q1, 'critic_2_g' : loss_q2}
        return loss_critic, q_targ
    
    def actor_loss(self, batch, logging=False):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']
        
        a = self.agent.to_tensor(a)
        
        q_pi, pi = self.agent.forward1(o, bg)
        action_l2 = (pi / self.agent.actor.act_limit).pow(2).mean()
        loss_actor = (- q_pi).mean() + self.args.action_l2 * action_l2
        
        pi_future = self.agent.get_pis(o, future_ag)
        loss_bc = (pi_future - a).pow(2).mean()
        if logging:
            self.monitor.store(
                Loss_actor=loss_actor.item(),
                Loss_action_l2=action_l2.item(),
                Loss_bc=loss_bc.item(),
            )
            monitor_log = dict(q_pi=q_pi)
            self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_actor
    
    
    def update_critic(self, batch, train_embed=True, logging=False):
        loss, q_targ = self.critic_loss(batch)
        loss_critic1 = loss['critic_1']
        loss_critic2 = loss['critic_2']
        self.q_optim1.zero_grad()
        self.q_optim2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), self.args.grad_norm_clipping).item()
            if logging:
                self.monitor.store(gradnorm_critic1=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), self.args.grad_norm_clipping).item()
            if logging:
                self.monitor.store(gradnorm_critic2=c_norm2)
        if self.args.grad_value_clipping > 0.:
            if logging:
                self.monitor.store(gradnorm_mean_critic1=net_utils.mean_grad_norm(self.agent.critic1.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1.parameters(), self.args.grad_value_clipping)
            if logging:
                self.monitor.store(gradnorm_mean_critic2=net_utils.mean_grad_norm(self.agent.critic2.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2.parameters(), self.args.grad_value_clipping)
        self.q_optim1.step()
        self.q_optim2.step()
        if logging:
            self.monitor.store(
                Loss_critic_1=loss_critic1.item(),
                Loss_critic_2=loss_critic2.item(),
            )
            monitor_log = dict(
                q_targ=q_targ
            )
            self.monitor.store(**dict_to_numpy(monitor_log))

    def update_critic_g(self, batch, train_embed=True, logging=False):
        loss, q_targ = self.critic_loss_g(batch)
        loss_critic1 = loss['critic_1_g']
        loss_critic2 = loss['critic_2_g']
        self.q_optim1_g.zero_grad()
        self.q_optim2_g.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1_g.parameters(), self.args.grad_norm_clipping).item()
            if logging:
                self.monitor.store(gradnorm_critic1_g=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2_g.parameters(), self.args.grad_norm_clipping).item()
            if logging:
                self.monitor.store(gradnorm_critic2_g=c_norm2)
        if self.args.grad_value_clipping > 0.:
            if logging:
                self.monitor.store(gradnorm_mean_critic1_g=net_utils.mean_grad_norm(self.agent.critic1_g.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1_g.parameters(), self.args.grad_value_clipping)
            if logging:
                self.monitor.store(gradnorm_mean_critic2_g=net_utils.mean_grad_norm(self.agent.critic2_g.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2_g.parameters(), self.args.grad_value_clipping)
        self.q_optim1_g.step()
        self.q_optim2_g.step()
        if logging:
            self.monitor.store(
                G_Loss_critic_1=loss_critic1.item(),
                G_Loss_critic_2=loss_critic2.item(),
            )
            monitor_log = dict(
                G_q_targ=q_targ
            )
            self.monitor.store(**dict_to_numpy(monitor_log))
            
    def update_actor(self, batch, train_embed=True, logging=False):
        loss_actor = self.actor_loss(batch, logging=logging)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        
        if self.args.grad_norm_clipping > 0.:
            a_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.grad_norm_clipping).item()
            if logging:
                self.monitor.store(gradnorm_actor=a_norm)
        if self.args.grad_value_clipping > 0.:
            if logging:
                self.monitor.store(gradnorm_mean_actor=net_utils.mean_grad_norm(self.agent.actor.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.args.grad_value_clipping)
        self.pi_optim.step()
    
    def target_update(self):
        self.agent.target_update()
    
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def state_dict(self):
        return dict(
            q_optim1=self.q_optim1.state_dict(),
            q_optim2=self.q_optim2.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
            q_optim1_g=self.q_optim1_g.state_dict(),
            q_optim2_g=self.q_optim2_g.state_dict()
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim1.load_state_dict(state_dict['q_optim1'])
        self.q_optim2.load_state_dict(state_dict['q_optim2'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
        self.q_optim1_g.load_state_dict(state_dict['q_optim1_g'])
        self.q_optim2_g.load_state_dict(state_dict['q_optim2_g'])
    
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

