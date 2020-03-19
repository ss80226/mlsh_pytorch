import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import wandb
import os
import collections
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'


class PPO():
    def __init__(self, policy, val_net, args, optimizer='adam'):
        '''
        Proximal Policy Optimization
        
        input:
        - policy: the policy that PPO intend to optimize
        - val_net: the value network that PPO intend to optimize
        - args: a dictionary that contains the all the parameters of PPO
        '''

        super(PPO, self).__init__()
        self._policy, self._value_net = policy, val_net
        self._clip_im = args['clip_th']
        self._ent_coef = args['entropy_coef']
        self._grad_norm_clip = args['grad_norm_clip']
        self._epoch = args['epoch']
        self._mini_bs = args['mini_bs']
        self._td_lambda = args['td_lambda']
        self._discount = args['discount']
        self._save_ckpt_period = args['save_ckpt_period']
        self._output_ckpt = args['output_ckpt']
        self._toggle_period = args['toggle_period']
        self._toggle_th = args['toggle_th']
        self._primi_energy = args['primi_energy']
        self._num_update, self._timestep = 0, 0
        self._used_energy = 0
        self._v_err_queue = collections.deque(maxlen=args['q_len'])

        self._mse = nn.MSELoss()
        if optimizer == 'adam':
            self._policy_optim = optim.Adam(self._policy.parameters(),
                                            lr=args['policy_lr'])
            self._val_net_optim = optim.Adam(self._value_net.parameters(),
                                             lr=args['val_net_lr'])
        elif optimizer == 'sgd':
            self._policy_optim = optim.SGD(
                self._policy.parameters(),
                lr=args['policy_lr'],
                momentum=args['sgd_momentum'],
                weight_decay=args['pol_weight_decay'])
            self._val_net_optim = optim.SGD(
                self._value_net.parameters(),
                lr=args['val_net_lr'],
                momentum=args['sgd_momentum'],
                weight_decay=args['val_net_weight_decay'])
        else:
            raise ValueError('invalid optimizer')
        self._pol_optim_decay = optim.lr_scheduler.StepLR(
            self._policy_optim, 500, args['pol_lr_decay'])
        self._val_net_optim_decay = optim.lr_scheduler.StepLR(
            self._val_net_optim, 500, args['val_net_lr_decay'])
        return

    def update_master(self, rollout):
        
        return
    def update_subpolicy(self, rollout):
        return
    
    def update(self, rollout):
        '''
        Given the replay buffer, we update the policy and value network 
        with Proximal Policy Optimization (PPO).

        input:
        - replay_buf: the replay buffer that store the experience of 
                      interaction
        '''

        log = []
        for i in range(self._epoch):
            for s, g, a, ret, adv, logp in rollout.generator(
                    self._mini_bs, self._discount, self._td_lambda):

                state = torch.tensor(s).float().to(DEVICE)
                goal = torch.tensor(g).float().to(DEVICE)
                action = torch.tensor(a).float().to(DEVICE)
                logp = torch.tensor(logp).to(DEVICE)
                mb_returns = torch.tensor(ret).float().to(DEVICE)
                mb_adv = torch.tensor(adv).float().to(DEVICE)

                critic_loss = self._update_critic(state, goal, mb_returns)
                pg_loss, ent = self._update_actor(state, goal, action, mb_adv,
                                                  logp)
                if self._policy.update_phase == 'update_primitive_policies':
                    self._used_energy += self._policy.calc_primitive_grad_norm()
                    # print('used energy:', self._used_energy)
                log.append((critic_loss, pg_loss, ent))
        wandb.log(
            {
                'gate_fn_grad_norm': self._policy.calc_gate_fn_grad_norm(),
                'primitive_gradnorm': self._policy.calc_primitive_grad_norm()
            },
            step=self._num_update)
        log = np.array(log)
        self._v_err_queue.append(log[:, 0].mean())
        self._timestep += len(rollout)
        self._val_net_optim_decay.step()
        self._pol_optim_decay.step()
        if self._num_update % self._save_ckpt_period == 0:
            self._save_ckpt()
        if self._num_update % 5 == 0:
            self._wandb_log(log)

        # print('toggle condition vp:', self._toggle_condition_vp(log))
        # if self._toggle_condition_vp(log):
        # if self._toggle_condition_queue_vp():
        # if self._toggle_symmetric_vp():
        #     self._policy.toggle_training()
        self._num_update += 1
        return

    # def _toggle_condition_period(self):
    #     return (self._num_update + 1) % self._toggle_period == 0

    # def _toggle_symmetric_vp(self):
    #     if self._policy.update_phase == 'update_gate_fn':
    #         if np.mean(self._v_err_queue) < self._toggle_th:
    #             self._v_err_queue.clear()
    #             return True
    #     else:
    #         if np.mean(self._v_err_queue) < self._toggle_th:
    #             self._used_energy = 0
    #             self._v_err_queue.clear()
    #             return True
    #     return False

    # def _toggle_condition_queue_vp(self):
    #     if self._policy.update_phase == 'update_gate_fn':
    #         if np.mean(self._v_err_queue) < self._toggle_th:
    #             self._v_err_queue.clear()
    #             return True
    #     else:
    #         if self._used_energy >= self._primi_energy:
    #             self._used_energy = 0
    #             return True
    #     return False
    # def _toggle_condition_vp(self, log):
    #     # print('current phase:', self._policy.update_phase)
    #     # print('gating fn update -> primitive update:',
    #     #       log[:, 0].mean() < self._toggle_th)
    #     # print('primitive update -> gating fn update:',
    #     #       self._used_energy >= self._primi_energy)

    #     if self._policy.update_phase == 'update_gate_fn':
    #         if log[:, 0].mean() < self._toggle_th:
    #             return True
    #     else:
    #         if self._used_energy >= self._primi_energy:
    #             self._used_energy = 0
    #             return True
    #     return False

    def _update_actor(self, state, goal, action, advantage, old_logp):
        logp, ent = self._policy.evaluate_action(state, goal, action)
        ratio = torch.exp(logp - old_logp.detach())
        p1 = advantage * ratio
        p2 = torch.clamp(ratio, min=1 - self._clip_im,
                         max=1 + self._clip_im) * advantage
        pg_loss = -torch.mean(torch.min(p1, p2))
        entropy_penality = torch.mean(ent)
        self._policy_optim.zero_grad()
        actor_loss = pg_loss - entropy_penality * self._ent_coef
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self._policy.parameters(),
                                 self._grad_norm_clip)
        self._policy_optim.step()
        return pg_loss.item(), entropy_penality.item()

    def _update_critic(self, state, goal, returns):
        value = self._value_net(state, goal)
        loss = self._mse(value.squeeze(), returns) * 0.5
        self._val_net_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._value_net.parameters(),
                                 self._grad_norm_clip)
        self._val_net_optim.step()
        return loss.item()

    def _save_ckpt(self):
        torch.save(
            {
                'mcp': self._policy.state_dict(),
                'val_net': self._value_net.state_dict()
            }, self._output_ckpt.replace('.',
                                         '_' + str(self._num_update) + '.'))
        return

    def _wandb_log(self, log):
        wandb.log(
            {
                'critic_loss': log[:, 0].mean(),
                'pg_loss': log[:, 1].mean(),
                'entropy_penality': log[:, 2].mean(),
                'actor_loss':
                log[:, 1].mean() + log[:, 2].mean() * self._ent_coef,
                'actor_lr': self._pol_optim_decay.get_lr()[0],
                'critic_lr': self._val_net_optim_decay.get_lr()[0],
                'timestep': self._timestep
            },
            step=self._num_update)
        return