from algorithm.prioritized_experience_replay_buffer.replay_buffer import ReplayBuffer
import numpy as np
import math
class FixPolicyPrey2():
    def __init__(self, agent_idx, action_dim):
        self.agent_idx = agent_idx
        self.action_dim = action_dim
        self._is_deterministic = True
        self.pool = ReplayBuffer(1e6)
        print("FixPolcyPrey action_dim: ", action_dim)

    def get_actions(self, observations, single=True, step_explore=0, explore_direction=0,gama_gussian=0):
        # [0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]
        # print("-observations-------",observations,observations[0])
        agent1_v = observations[0][-2:]
        d_v1 =  np.linalg.norm(agent1_v)
        agent2_v = observations[0][-4:-2]
        d_v2 =  np.linalg.norm(agent2_v)
        agent3_v = observations[0][-6:-4]
        d_v3 =  np.linalg.norm(agent3_v)
        pos = observations[0][2:4]
        dist_=np.sqrt(np.sum(np.square(pos)))
        if dist_ > 0.4:
            e_force = -pos * (dist_- 0.4) * 30
        else :
            e_force = 0.0
        a_force = np.array(agent3_v)/d_v3*np.clip((1-d_v3),0,100) + np.array(agent2_v)/d_v2*np.clip((1-d_v2),0,100)  + np.array(agent1_v)/d_v1*np.clip((1-d_v1),0,100) 
        noise = np.random.rand(2)-0.5
        # noise = 0
        force = e_force + noise - a_force
        force_v = np.max(np.fabs(force))
        # force = force *0.7 + force * 0.3 * (np.random.rand()-0.5 * 2)#add noise jone
        if force_v == 0 :
            return [0.0, 0.0]
        return np.clip(force/force_v,-1.0,1.0)  # regulation

    def toggle_deterministic(self):
        print("hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        pass
    # def target_action(self, observations):
    #     return [[np.random.rand() * 2 - 1 for _ in range(self.action_dim)]] * observations.shape[0]

    def experience(self, obs, action, reward, obs_nxt, done):
        self.pool.add(obs, action, reward, obs_nxt, float(done), None, None)

    def _do_training(self, agents, iteration, batch_size, epoch_len):
        pass

    def is_exploration_enough(self, min_pool_size):
        return False