import numpy as np
import pdb


class Storage():
    def __init__(self,
                 num_steps,
                 num_processes,
                 obs_dim,
                 action_dim,
                 subpolicies_num,
                 macrolen,
                 goal_dim=None):
        self._n, self._step_limit = num_processes, num_steps
        self._obs_dim, self._action_dim = obs_dim, action_dim
        self._use_goal = goal_dim is not None
        if goal_dim is not None:
            self._goal_dim = goal_dim
            self._goal = np.zeros((num_steps + 1, num_processes, goal_dim))
        # pdb.set_trace()
        self._obs = np.zeros((num_steps + 1, num_processes, obs_dim))

        self._macrolen = macrolen
        self._subpolicies_num = subpolicies_num

        self._action = np.zeros((num_steps, num_processes, action_dim))
        self._done = np.zeros((num_steps, num_processes, 1))
        self._reward = np.zeros((num_steps, num_processes, 1))
        self._value = np.zeros((num_steps + 1, num_processes, 1))
        self._logp = np.zeros((num_steps, num_processes, 1))
        self._subId = torch.zeros((num_steps, num_processes, 1))

        self._num_steps, self._step = num_steps, 0

        self._ep_len_cnt, self._ep_cnt = np.zeros(self._n), np.zeros(self._n)
        self._ep_rd_cnt = np.zeros(self._n)
        self._cur_ep_cnt = np.zeros(self._n)
        self._cur_rd_cnt = np.zeros(self._n)
        self._ave_len = 0
        return

    def __len__(self):
        return self._step * self._n

    def is_full(self):
        return self._step >= self._step_limit

    def put(self, obs, goal, action, reward, done, value, logp, subId):
        if self._use_goal:
            self._goal[self._step + 1] = goal
        self._obs[self._step + 1] = obs
        self._action[self._step] = action
        self._value[self._step] = value
        self._reward[self._step] = np.expand_dims(reward, axis=-1)
        self._subId[self._step] = np.expand_dims(subId, axis=-1)
        self._done[self._step] = np.expand_dims(done, -1).astype(float)
        self._logp[self._step] = np.expand_dims(logp, axis=-1)
        self._step += 1

        # update episode counter 
        # self._inc_episode_counter(done)
        return

    def insert_init(self, obs, goal=None):
        if goal is not None and self._use_goal:
            self._goal[0] = goal
        self._obs[0] = obs
        return

    def insert_last_val(self, value):
        self._value[-1] = value
        return

    def after_update(self):
        self._obs[0] = self._obs[-1]
        if self._use_goal:
            self._goal[0] = self._goal[-1]
        self._step = 0
        return

    def _cal_gae(self, gamma, td):
        returns = np.zeros((self._step_limit, self._n, 1))
        advantages = np.zeros((self._step_limit, self._n, 1))
        mask = 1 - self._done
        gae = 0
        for t in reversed(range(self._step_limit)):
            delta = self._reward[t] + gamma * self._value[
                t + 1] * mask[t] - self._value[t]
            gae = delta + gamma * td * mask[t] * gae
            returns[t] = gae + self._value[t]
        advantages = returns - self._value[:-1]
        # some bug here
        advantages = (advantages - advantages.mean()) / (advantages.std() +
                                                         1e-8)
        return returns, advantages

    def _inc_episode_counter(self, done):
        self._cur_ep_cnt[not done] += 1
        self._ep_len_cnt[done] += self._cur_ep_cnt[done]
        self._ep_cnt[done] += 1
        self._cur_ep_cnt[done] = 0
        return

    def _reset_episode_counter(self, done):
        self._ep_len_cnt, self._ep_cnt = np.zeros(self._n), np.zeros(self._n)
        return

    def generator(self, mini_bs, gamma, td):
        ret, adv = self._cal_gae(gamma, td)
        indices = np.random.choice(len(self), len(self))
        state = self._obs.reshape(-1, self._obs_dim)
        action = self._action.reshape(-1, self._action_dim)
        logp = self._logp.reshape(-1)
        ret, adv = ret.reshape(-1), adv.reshape(-1)
        subId = self._subId.reshape(-1)
        subpol_counts = []
        states = actions = logps = rets = advs = subIds = []
        for i in range(self._subpolicies_num):
            subpol_counts.append(0)
            states.append(np.zeros(shape=(self._n*self._step_limit, self._obs_dim)))
            actions.append(np.zeros(shape=(self._n * self._step_limit, self._action_dim)))
            logps.append(np.zeros(shape=(self._n * self._step_limit, 1)))
            rets.append(np.zeros(shape=(self._n * self._step_limit, 1)))
            advs.append(np.zeros(shape=(self._n * self._step_limit, 1)))
            subIds.append(np.zeros(shape=(self._n * self._step_limit, 1)))

        for i in range(len(subId)):
            x = self._subId[i]
            subpol_counts[x] += 1
            states[x][i] = state[i]
            actions[x][i] = action[i]
            logps[x][i] = logp[i]
            ret[x][i] = ret[i]
            subIds[x][i] = subIds[i]

        for i in range(self._subpolicies_num):
            x = self._n*self._step_limit // subpol_counts[i]
            for j in range(x):
                states[i][j*subpol_counts[i]:(j+1)*subpol_counts[i]] = state[i][:subpol_counts[i]]
                actions[i][j*subpol_counts[i]:(j+1)*subpol_counts[i]] = actions[i][:subpol_counts[i]]
                logps[i][j*subpol_counts[i]:(j+1)*subpol_counts[i]] = logps[i][:subpol_counts[i]]
                rets[i][j*subpol_counts[i]:(j+1)*subpol_counts[i]] = rets[i][:subpol_counts[i]]
                advs[i][j*subpol_counts[i]:(j+1)*subpol_counts[i]] = advs[i][:subpol_counts[i]]
                subIds[i][j*subpol_counts[i]:(j+1)*subpol_counts[i]] = subIds[i][:subpol_counts[i]]
            
        if self._use_goal:
            goal = self._goal.reshape(-1, self._goal_dim)

        for i in range(len(self) // mini_bs):
            mb_idx = indices[i * mini_bs:(i + 1) * mini_bs]

            if self._use_goal:
                yield (state[mb_idx], goal[mb_idx], action[mb_idx],
                       ret[mb_idx], adv[mb_idx], logp[mb_idx])
            else:
                yield (state[mb_idx], action[mb_idx], ret[mb_idx], adv[mb_idx],
                       logp[mb_idx])

    def ave_step_reward(self):
        return np.mean(self._reward)

    def ave_episode_reward(self):
        n_traj = np.count_nonzero(self._done)
        if n_traj <=0:
            n_traj = 1
        return np.sum(self._reward) / n_traj

    def ave_episode_len(self):
        n_traj = np.count_nonzero(self._done)
        if n_traj <=0:
            n_traj = 1
        return self._n * self._step_limit / n_traj
