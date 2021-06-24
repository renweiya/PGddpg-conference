from algorithm.trainer import AgentTrainer

class FixedPrey(AgentTrainer):
    def __init__(self, agent_idx, action_dim):
        self.agent_idx = agent_idx
        self.action_dim = action_dim
        print("random action_dim: ", action_dim)

    def get_actions(self, observations, single=True, step_explore=False,explore_direction=1):
        return [0 for _ in range(self.action_dim)]

    # def experience(self, obs, action, reward, obs_nxt, done):
    #     self.pool.add(obs, action, reward, obs_nxt, float(done), None, None)

    # def do_training(self, agents, end):
    #     pass

    # def is_exploration_enough(self, min_pool_size):
    #     return False