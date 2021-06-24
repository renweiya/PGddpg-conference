class AgentTrainer(object):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    def get_actions(self, obs):
        raise NotImplemented()

    def experience(self, obs, act, rew, new_obs, done):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def do_training(self, agents, train_step, end):
        raise NotImplemented()


from algorithm.trainer.pgddpg import AgentTrainer_PGDDPG
from algorithm.trainer.ddpg import AgentTrainer_DDPG
from algorithm.trainer.fixed import FixedPrey
from algorithm.trainer.random_agent import RandomAgent
from algorithm.prioritized_experience_replay_buffer.replay_buffer import ReplayBuffer as NormalReplayBuffer
from algorithm.prioritized_experience_replay_buffer.replay_buffer import PrioritizedReplayBuffer as PrioritizedReplayBuffer
from algorithm.prioritized_experience_replay_buffer.priority_queue_buffer import PriorityTrajectoryReplayBuffer
from algorithm.prioritized_experience_replay_buffer.trajectory_replay_buffer import TrajectoryReplayBuffer
from algorithm.common.network_utils import mlp_model, plan_model, plan_model_force, plan_model_force2
from algorithm.trainer.fix_policy_prey import FixPolicyPrey


class SimpleAgentFactory(object):

    @staticmethod #python staticmethod 返回函数的静态方法，该方法不强制要求传递参数
    def createAgent(agent_idx, policy_name, obs_shape, action_dim, hyper_parameters):

        if policy_name == 'pgddpg':
            if hyper_parameters.prioritized_er:
                pool = PrioritizedReplayBuffer(hyper_parameters.buffer_size, alpha=hyper_parameters.alpha)
            else:
                pool = NormalReplayBuffer(hyper_parameters.buffer_size, save_return=hyper_parameters.save_return)
            agent = AgentTrainer_PGDDPG(
                "agent_%d" % agent_idx, mlp_model, plan_model_force2 ,obs_shape, action_dim, agent_idx,
                buffer=pool, local_q_func=True)
            print("Agent {} is pgddpg...".format(agent_idx))
        
        elif policy_name == 'ddpg':
            if hyper_parameters.prioritized_er:
                pool = PrioritizedReplayBuffer(hyper_parameters.buffer_size, alpha=hyper_parameters.alpha)
            else:
                pool = NormalReplayBuffer(hyper_parameters.buffer_size, save_return=hyper_parameters.save_return)
            agent = AgentTrainer_DDPG(
                "agent_%d" % agent_idx, mlp_model, plan_model_force2 ,obs_shape, action_dim, agent_idx,
                buffer=pool, local_q_func=True)
            print("Agent {} is ddpg...".format(agent_idx))

        elif policy_name == 'random':
            agent = RandomAgent(agent_idx, action_dim)
            print("Agent {} is random...".format(agent_idx))
        
        elif policy_name == 'fixed':
            agent = FixedPrey(agent_idx, action_dim)
            print("Agent {} is fixed...".format(agent_idx))
        
        elif policy_name == 'designed':
            agent = FixPolicyPrey(agent_idx, action_dim)
            print("Agent {} is designed...".format(agent_idx))

        return agent