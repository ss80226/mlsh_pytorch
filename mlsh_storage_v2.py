import numpy as np
import torch
import pdb
import math


class Storage():
    def __init__(self,
                 num_steps,
                 num_proc,
                 obs_dim,
                 action_dim,
                 subpolicies_num,
                 macrolen,
                 latent_dim=None,
                 goal_dim=None,
                 p_recur_dim=None,
                 v_recur_dim=None):
        self._n, self._step_limit = num_proc, num_steps
        self._s_dim, self._a_dim = obs_dim, action_dim
        
        self._macrolen = macrolen
        self._subpolicies_num = subpolicies_num

        self._use_goal = goal_dim is not None
        if self._use_goal:
            self._g_dim = goal_dim
            self._goal = torch.zeros((num_steps + 1, num_proc, goal_dim))

        self._use_latent = latent_dim is not None
        if self._use_latent:
            self._z_dim = latent_dim
            self._eps = torch.zeros((num_steps, num_proc, latent_dim))

        self._use_recurr = not (p_recur_dim is None or v_recur_dim is None)
        if self._use_recurr:
            self._p_recurr_dim, self._v_recurr_dim = p_recur_dim, v_recur_dim
            self._p_rec_hxt = torch.zeros(
                (num_steps + 1, num_proc, p_recur_dim))
            self._v_rec_hxt = torch.zeros(
                (num_steps + 1, num_proc, v_recur_dim))

        self._obs = torch.zeros((num_steps + 1, num_proc, obs_dim))
        self._action = torch.zeros((num_steps, num_proc, action_dim))
        self._done = torch.zeros((num_steps + 1, num_proc, 1))
        self._reward = torch.zeros((num_steps, num_proc, 1))
        self._value = torch.zeros((num_steps + 1, num_proc, 1))
        self._logp = torch.zeros((num_steps, num_proc, 1))
        self._subId = torch.zeros((num_steps, num_proc, 1))
        self._num_steps, self._step, self._dev = num_steps, 0, 'cpu'
        self._create_log_counter()
        return

    def __len__(self):
        return self._step * self._n

    def to(self, device):
        if self._use_goal:
            self._goal = self._goal.to(device)
        if self._use_latent:
            self._eps = self._eps.to(device)
        if self._use_recurr:
            self._p_rec_hxt = self._p_rec_hxt.to(device)
            self._v_rec_hxt = self._v_rec_hxt.to(device)

        self._obs, self._action = self._obs.to(device), self._action.to(device)
        self._done, self._logp = self._done.to(device), self._logp.to(device)
        self._reward = self._reward.to(device)
        self._value, self._dev = self._value.to(device), device
        return

    def is_full(self):
        return self._step >= self._step_limit

    def put(self,
            obs,
            action,
            reward,
            done,
            value,
            logp,
            sub_id,
            goal=None,
            eps=None,
            p_rec_hxt=None,
            v_rec_hxt=None):
        if self._use_goal:
            self._goal[self._step + 1] = goal
        if self._use_latent:
            self._eps[self._step] = eps
        if self._use_recurr:
            self._p_rec_hxt[self._step + 1].copy_(p_rec_hxt.detach())
            self._v_rec_hxt[self._step + 1].copy_(v_rec_hxt.detach())

        subId = torch.tensor(sub_id).unsqueeze(-1)
        rwd = torch.tensor(reward).unsqueeze(-1)
        d = torch.tensor(done).unsqueeze(-1).float()

        self._obs[self._step + 1].copy_(obs)
        self._action[self._step].copy_(action.detach())
        self._value[self._step].copy_(value.detach())
        self._reward[self._step].copy_(rwd)
        self._done[self._step + 1].copy_(d)
        self._logp[self._step].copy_(logp.unsqueeze(-1).detach())
        self._subId[self._step].copy(subId.detach())
        self._step += 1

        self._inc_episode_counter(d.squeeze().bool(), rwd.squeeze())
        return

    def insert_init(self, obs, mask, goal=None, p_rec_hxt=None,
                    v_rec_hxt=None):
        if self._use_goal:
            self._goal[0].copy_(goal.detach())
        if self._use_recurr:
            self._p_rec_hxt[0].copy_(p_rec_hxt.detach())
            self._v_rec_hxt[0].copy_(v_rec_hxt.detach())

        d = 1 - mask.unsqueeze(-1).float()
        self._done[0].copy_(d.detach())
        self._obs[0].copy_(obs.detach())
        self._inc_episode_counter(start=True)
        return

    def insert_last_val(self, value):
        self._value[-1].copy_(value)
        return

    def after_update(self):
        self._obs[0], self._done[0] = self._obs[-1], self._done[-1]
        if self._use_goal:
            self._goal[0].copy_(self._goal[-1])
        if self._use_recurr:
            self._p_rec_hxt[0].copy_(self._p_rec_hxt[-1])
            self._v_rec_hxt[0].copy_(self._v_rec_hxt[-1])
        self._step = 0
        return

    def reset_episode_log(self):
        self._reset_episode_counter()
        return

    # def flat_generator(self, gamma, td, gae=True):
    #     indices = np.random.choice(len(self), len(self))
    #     s, g, eps, a, p_rec_hxt, v_rec_hxt, m, ret, adv, logp = self._pack(
    #         gamma, td, gae)
    #     return [
    #         s[indices], g[indices], eps[indices], a[indices],
    #         p_rec_hxt[indices], v_rec_hxt[indices], m[indices], ret[indices],
    #         adv[indices], logp[indices]
    #     ]

    def generator(self, mini_bs, gamma, td, gae=True):
        '''
        Return the generator of reply buffer.
        
        input:
        - mini_bs: the batch size of the sampled batch
        - gamma: discount factor
        - td: lambda of generalized advantage estimation
        - gae: use gae or not

        output (yield):
        s, g, eps, a, p_rec_hxt, v_rec_hxt, mask, ret, adv, logp, sub_id
        '''

        indices = np.random.choice(len(self), len(self))
        s, g, eps, a, p_rec_hxt, v_rec_hxt, m, ret, adv, logp = self._pack(
            gamma, td, gae)
        for i in range(len(self) // mini_bs):
            mb_idx = indices[i * mini_bs:(i + 1) * mini_bs]
            for j in range(self._subpolicies_num):

            yield [
                s[mb_idx], g[mb_idx], eps[mb_idx], a[mb_idx],
                p_rec_hxt[mb_idx], v_rec_hxt[mb_idx], m[mb_idx], ret[mb_idx],
                adv[mb_idx], logp[mb_idx]
            ]

    def recurrent_generator(self, mini_bs, gamma, td, gae=True):
        num_batch = self._n // math.ceil(mini_bs / self._step_limit)
        env_per_batch = self._n // num_batch
        s, g, eps, a, p_rec_hxt, v_rec_hxt, m, ret, adv, logp = self._traj_pack(
            gamma, td, gae)
        indice = np.random.choice(self._n, self._n)
        # indice = np.arange(self._n)
        for i in range(num_batch):
            mb_idx = indice[i * env_per_batch:(i + 1) * env_per_batch]
            N, T = env_per_batch, self._step_limit
            # pdb.set_trace()
            s_, a_ = s[mb_idx].view(N * T, -1), a[mb_idx].view(N * T, -1)
            g_, m_ = g[mb_idx].view(N * T, -1), m[mb_idx].view(N, T)
            eps_ = eps[mb_idx].view(N * T, -1)
            ret_ = ret[mb_idx].view(N * T, -1)
            adv_ = adv[mb_idx].view(N * T, -1)
            logp_ = logp[mb_idx].view(N * T)
            yield [
                s_, g_, eps_, a_, p_rec_hxt[mb_idx, 0], v_rec_hxt[mb_idx, 0],
                m_, ret_, adv_, logp_
            ]

    def ave_step_reward(self):
        return torch.mean(self._reward).item()

    def ave_episode_reward(self):
        n_traj = max(1, self._ep_cnt.sum())
        return (self._ep_rd_cnt.sum() / n_traj).item()

    def ave_episode_len(self):
        n_traj = max(1, self._ep_cnt.sum())
        return (self._ep_len_cnt.sum() / n_traj).item()

    def num_episode(self):
        return self._ep_cnt.sum()

    def _cal_gae(self, gamma, td):
        ret = torch.zeros((self._step_limit, self._n, 1)).to(self._dev)
        adv = torch.zeros((self._step_limit, self._n, 1)).to(self._dev)
        mask, gae = 1 - self._done, 0

        for t in reversed(range(self._step_limit)):
            delta = self._reward[t] + gamma * self._value[t + 1] * mask[
                t + 1] - self._value[t]
            gae = delta + gamma * td * mask[t + 1] * gae
            ret[t] = gae + self._value[t]
        adv = ret - self._value[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return ret, adv

    def _pack(self, gamma, td, gae):
        with torch.no_grad():
            ret, adv = self._cal_gae(
                gamma, td) if gae else self._cal_normal_adv(gamma)
        state = self._obs.view(-1, self._s_dim)
        action = self._action.view(-1, self._a_dim)
        logp = self._logp.view(-1)
        ret, adv = ret.view(-1), adv.view(-1)
        mask = 1 - self._done.view(-1)
        default = np.array([None] * len(self))

        goal = self._goal.view(-1, self._g_dim) if self._use_goal else default
        eps = self._eps.view(-1, self._z_dim) if self._use_latent else default
        p_rec_hxt = self._p_rec_hxt.view(
            -1, self._p_recurr_dim) if self._use_recurr else default
        v_rec_hxt = self._v_rec_hxt.view(
            -1, self._v_recurr_dim) if self._use_recurr else default
        subpol_counts = []
        for i in range(self._subpolicies_num):
            subpol_counts.append(0)
        for i in range(len(state) // self._macrolen):
            subId = self._subId[i*self._macrolen]
            subpol_counts[subId] += self._macrolen
        states = goals = eps = actoins = p_rec_hxt = v_rec_hxt
        for i in range(self._subpolicies_num):

        return [
            state, goal, eps, action, p_rec_hxt, v_rec_hxt, mask, ret, adv,
            logp
        ]

    def _traj_pack(self, gamma, td, gae):
        with torch.no_grad():
            ret, adv = self._cal_gae(
                gamma, td) if gae else self._cal_normal_adv(gamma)
        m = (1 - self._done[:-1].squeeze(-1)).permute(1, 0)
        default = torch.zeros((self._n, self._step_limit)).to(self._dev)
        s = self._obs[:-1].permute(1, 0, 2)
        a = self._action.permute(1, 0, 2)
        logp = self._logp.permute(1, 0, 2).squeeze(-1)
        ret = ret.permute(1, 0, 2).squeeze(-1)
        adv = adv.permute(1, 0, 2).squeeze(-1)

        g = self._goal[:-1].permute(1, 0, 2) if self._use_goal else default
        eps = self._eps.permute(1, 0, 2) if self._use_latent else default
        p_rec_hxt = self._p_rec_hxt[:-1].permute(
            1, 0, 2) if self._use_recurr else default
        v_rec_hxt = self._v_rec_hxt[:-1].permute(
            1, 0, 2) if self._use_recurr else default
        return [s, g, eps, a, p_rec_hxt, v_rec_hxt, m, ret, adv, logp]

    def _cal_normal_adv(self, gamma):
        ret = np.zeros((self._step_limit + 1, self._n, 1)).to(self._dev)
        adv = np.zeros((self._step_limit, self._n, 1)).to(self._dev)
        mask = 1 - self._done
        ret[-1] = self._value[-1]

        for t in reversed(range(self._step_limit)):
            ret[t] = ret[t + 1] * gamma * mask[t + 1] + self._reward[t]
        adv = ret[:-1] - self._value[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return ret, adv

    def _create_log_counter(self):
        self._ep_len_cnt = torch.zeros(self._n)
        self._ep_cnt = torch.zeros(self._n)
        self._ep_rd_cnt = torch.zeros(self._n)
        self._cur_ep_cnt = torch.zeros(self._n)
        self._cur_rd_cnt = torch.zeros(self._n)
        self._ave_len = 0
        return

    def _inc_episode_counter(self, done=None, reward=None, start=False):
        if isinstance(done, list):
            done = torch.tensor(done)
        if isinstance(reward, list):
            reward = torch.tensor(reward)
        self._cur_ep_cnt += 1
        if start:
            return
        # pdb.set_trace()
        self._cur_rd_cnt += reward
        self._ep_len_cnt[done] += self._cur_ep_cnt[done]
        self._ep_rd_cnt[done] += self._cur_rd_cnt[done]
        self._ep_cnt[done] += 1
        self._cur_ep_cnt[done], self._cur_rd_cnt[done] = 0, 0
        return

    def _reset_episode_counter(self):
        self._ep_len_cnt = torch.zeros(self._n)
        self._ep_cnt = torch.zeros(self._n)
        self._ep_rd_cnt = torch.zeros(self._n)
        return
