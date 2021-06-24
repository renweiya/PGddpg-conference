# -*- coding: utf-8 -*-
import os

import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 使用 GPU 0
import tensorflow as tf
import numpy as np
import algorithm.common.tf_utils as tf_utils
import time
import random

# read input cmd from standard input device
flags = tf.app.flags

# Game parameter
# flags.DEFINE_string('env_name', 'predator_prey', 'env used')
flags.DEFINE_string('env_name', 'round_up_base', 'env used')
flags.DEFINE_bool('render', False, 'whether to render the scenario')
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_integer('num_adversaries', 3, 'num_adversaries')#3
flags.DEFINE_integer('num_good_agents', 1, 'num_good_agents')#1
flags.DEFINE_integer('max_step_before_punishment', 8, 'max_step_before_punishment')#modify 8
flags.DEFINE_bool('reload_prey', True, 'whether to reload the pre-trained prey model')
flags.DEFINE_bool('train_prey', False, 'whether to reload the pre-trained prey model')
# flags.DEFINE_string('prey_model_path','test_model/prey/model-prey-01',#'./exp_result/3v1_dual/saved_models/round_up_reward_shaping/seed_ddpg/20200905143204/model-38000',  
                    # 'path of the pre-trained prey model')
flags.DEFINE_bool('reload_predator', True, 'whether to reload the pre-trained predator model')
# test for all model 
predator_model_path = [ 
                        'test_model/predator/ddpg_rs_1/model-40000', 
                        'test_model/predator/pg_exp_obstacle/model-40000',  
                        'test_model/predator/pgddpg/model-40000',  
                        'test_model/predator/ddpg/model-40000', 
                        ]
prey_model_path = [ 
                    'test_model/prey/model-prey-s', 
                    'test_model/prey/model-prey-s', 
                    'test_model/prey/model-prey-s', 
                    'test_model/prey/model-prey-s',
                                            ]
# test policy
# predator_policy = ['ddpg','maddpg','ddpg',
#                                         'ddpg','maddpg','ddpg',
#                                         'ddpg','maddpg','ddpg',
#                                         'ddpg','maddpg','ddpg',
#                                         'ddpg','maddpg','ddpg',
#                                         'ddpg','maddpg','ddpg',
#                                         'ddpg','maddpg','ddpg',
#                                         'ddpg','maddpg','ddpg',
#                                         'ddpg','maddpg','ddpg',
#                                         'ddpg','maddpg','ddpg',
#                                     ]
predator_policy = [
                    'ddpg',#'ddpg','ddpg','ddpg','ddpg','ddpg','ddpg','ddpg','ddpg','ddpg'
                    'pgddpg_exp',#'pgddpg_exp','pgddpg_exp','pgddpg_exp','pgddpg_exp','pgddpg_exp','pgddpg_exp','pgddpg_exp','pgddpg_exp','pgddpg_exp'
                    'pgddpg',#'pgddpg','pgddpg','pgddpg','pgddpg','pgddpg','pgddpg','pgddpg','pgddpg','pgddpg'
                    'ddpg',#'ddpg','ddpg','ddpg','ddpg','ddpg','ddpg','ddpg','ddpg','ddpg'
                                    ]
prey_policy = [ ]

# Training parameters
flags.DEFINE_bool('learning', False, 'train the agents')
flags.DEFINE_string('exp_name', 'rounding', 'exp_name')
# flags.DEFINE_string('predator_policy', 'pgddpg', 'predator_policy')
# flags.DEFINE_float('rlpl_beta', 0.5, 'rlpl_beta') #jone   # 0 < rlpl_beta < 1 , if rlpl_beta < 0 pgddpg rlpl_beta_dec

flags.DEFINE_string('prey_policy', 'ddpg', 'prey_policy: [random, fixed, ddpg, fixpolicy]')

flags.DEFINE_integer('episodes', 1000, 'maximum training episode')

flags.DEFINE_integer('max_episode_len', 200, 'maximum step of each episode')#modify 60
flags.DEFINE_float('ddpg_plr', 0.01, 'policy learning rate')
flags.DEFINE_float('ddpg_qlr', 0.001, 'critic learning rate')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_float('tau', 0.01, 'target network update frequency')
flags.DEFINE_integer('target_update_interval', 1, 'target network update frequency')
flags.DEFINE_float('return_confidence_factor', 0.7, 'return_confidence_factor')
flags.DEFINE_integer('batch_size', 1024, 'batch size')
flags.DEFINE_integer('n_train_repeat', 1, 'repeated sample times at each training time')
flags.DEFINE_integer('save_checkpoint_every_epoch', 2000, 'save_checkpoint_every_epoch')#5000
flags.DEFINE_integer('plot_reward_recent_mean', 1000, 'show the avg reward of recent 200 episode')
flags.DEFINE_bool('save_return', True, 'save trajectory Return by default')
flags.DEFINE_float('lambda1', 0., 'n-step return')
flags.DEFINE_float('lambda1_max', 1., 'n-step return')
flags.DEFINE_float('lambda2', 1e-6, 'coefficient of regularization')

# experience replay
flags.DEFINE_integer('buffer_size', 3000, 'buffer size')
flags.DEFINE_integer('min_buffer_size', 300, 'minimum buffer size before training')#30000
flags.DEFINE_integer('positive_buffer_size', 32, 'buffer size')
flags.DEFINE_integer('min_positive_buffer_size', 32, 'min buffer size before training')
# prioritized
flags.DEFINE_bool('prioritized_er', False, 'whether to use prioritized ER')
flags.DEFINE_float('alpha', 0.6, 'how much prioritization is used (0 - no prioritization, 1 - full prioritization)')
flags.DEFINE_float('beta', 0.4, 'To what degree to use importance weights (0 - no corrections, 1 - full correction)')

# Net structure
flags.DEFINE_integer('num_units', 128, 'layer neuron number')#laker 32
flags.DEFINE_integer('num_units_ma', 256, 'layer neuron number for multiagent alg')#laker:64
flags.DEFINE_integer('h_layer_num', 2, 'hidden layer num')

# Model saving dir
flags.DEFINE_string('model_save_dir', './exp_result/{}/{}/saved_models/seed_{}/model',
                    'Model saving dir')
flags.DEFINE_string('learning_curve_dir', './exp_result/{}/{}/learning_curves/seed_{}',
                    'learning_curve_dir')
FLAGS = flags.FLAGS  # alias


def make_env(scenario_name, max_step_before_punishment,predator_policy):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from env.multiagent.environment import MultiAgentEnv
    import env.multiagent.scenarios as scenarios

    # load scenario from script
    # scenario = scenarios.load(scenario_name + ".py").Scenario() # modify
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    scenario.max_step_before_punishment = max_step_before_punishment
    scenario.adv_policies = predator_policy
    
    #add by laker
    scenario.num_adversaries=FLAGS.num_adversaries
    scenario.num_good_agents=FLAGS.num_good_agents
    scenario.successed_round_up=FLAGS.num_adversaries
    scenario.good_colors = [np.array([0x00, 0x99, 0xff]) / 255] * FLAGS.num_good_agents
    #add end
    print('==============================================================')
    print('max_step_before_punishment: ', scenario.max_step_before_punishment)
    print('==============================================================')

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        info_callback=scenario.collision_number,
                        done_callback=scenario.done,
                        shared_viewer=True,#modify share view or not
                        other_callbacks=[scenario.set_arrested_pressed_watched])
    return env


def build_agents(action_dim_n, observation_dim_n, policies_name):
    '''
    build agents
    :param action_dim_n:
    :param observation_dim_n:
    :param policies_name:
    :return:
    '''
    from algorithm.trainer import SimpleAgentFactory
    agents = []
    obs_shape_n = [[dim] for dim in observation_dim_n]
    
    
    for agent_idx, policy_name in enumerate(policies_name):
        agents.append(SimpleAgentFactory.createAgent(agent_idx, policy_name, obs_shape_n, action_dim_n, FLAGS))
    return agents


def reload_previous_models(session, env, pa_model_path, pe_model_path):
    import gc
    # 加载提前训练好的 prey 策略
    if FLAGS.reload_prey:
        prey_vars = []
        for idx in range(FLAGS.num_adversaries, env.n):#range(2,5)
            var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_{}'.format(idx))
            prey_vars += var

        saver_prey = tf.train.Saver(var_list=prey_vars)
        saver_prey.restore(session, pe_model_path)

        print('[prey] successfully reload previously saved ddpg model({})...'.format(pe_model_path))
        del saver_prey
        gc.collect()
        # all the predator using the same policy
        # best_agent = agents[base_kwargs['num_adversaries']]
        # for i in range(base_kwargs['num_adversaries'], env.n):
        #     agents[i] = best_agent

    if FLAGS.reload_predator:
        predator_vars = []
        for idx in range(FLAGS.num_adversaries):
            var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_{}'.format(idx))
            predator_vars += var
        saver_predator = tf.train.Saver(var_list=predator_vars)
        saver_predator.restore(session, pa_model_path)
        print('[predator] successfully reload previously saved RL model({})...'.format(
            pa_model_path
        ))
        del saver_predator
        gc.collect()


def train():
    for i in range(len(predator_model_path)):
        adv_policies = [predator_policy[i]] * FLAGS.num_adversaries
        good_policies = [ FLAGS.prey_policy ]* FLAGS.num_good_agents
        print(adv_policies + good_policies)
        # reload previous prey and predator model
        pa_model_path = predator_model_path[i]
        pe_model_path = prey_model_path[i]


        # init env
        env = make_env(FLAGS.env_name, FLAGS.max_step_before_punishment, predator_policy[i])
        env = env.unwrapped
        # set env seed
        np.random.seed(FLAGS.seed)
        random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)
        print("Using seed {} ...".format(FLAGS.seed))

        print('There are total {} agents.'.format(env.n))
        obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
        action_shpe_n = [2] * env.n
        print('obs_shape_n: ', obs_shape_n)  # [16, 16, 16, 14]
        print(action_shpe_n)  # [5, 5, 5, 5]

        tf.reset_default_graph()
        with tf_utils.make_session().as_default() as sess:
            # init agents
            agents = build_agents(action_shpe_n, obs_shape_n, adv_policies + good_policies)

            # init tf summaries
            summary_path = FLAGS.learning_curve_dir.format(FLAGS.env_name, FLAGS.exp_name, FLAGS.seed)
            print('summary_path', summary_path)
            # summary_writer = tf.summary.FileWriter(summary_path)
        
            reload_previous_models(session=sess, env=env , pa_model_path = pa_model_path , pe_model_path = pe_model_path)

            # Initialize uninitialized variables.
            tf_utils.initialize(sess=sess)
            # assert using same session
            same_session(sess, agents)
            #  make the tensor graph unchangeable
            sess.graph.finalize()

            # collect some statistical data
            episode_rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
            agent_episode_rewards = [[0.0] for _ in range(env.n)]
            losses_transformation = [[] for _ in range(FLAGS.num_adversaries)]
            coordination_reach_times = [[0] for _ in range(FLAGS.num_good_agents)]

            obs_n = env.reset() #observations 18,18,16,16,16
            episode_step = 0  # step for each episode
            episode_step_r = []  # step for each episode record
            train_step = 0  # total training step
            t_start = time.time()
            #deterministic action .no noise.
            for agent in agents:
                agent.toggle_deterministic()
                print("*********************agent._is_deterministic*********************:",agent._is_deterministic)
            print('Starting iterations...')
            sucess_rate=[]
            while len(episode_rewards) <= FLAGS.episodes:#100
                # increment global step counter
                train_step += 1
                
                action_2_dim_n = [agent.get_actions(observations=[obs], single=True) for agent, obs in zip(agents, obs_n)]

                action_n = [[0, a[0], 0, a[1], 0] for a in action_2_dim_n]

                new_obs_n, rew_n, done_n, info_n = env.step(action_n, restrict_move=True)

                info_n = info_n['n']
                episode_step += 1

                done = all(done_n)  # 达到任务
                terminal = (episode_step >= FLAGS.max_episode_len)  # 最大步数
                ended = done or terminal
                #add by laker
                if ended:
                    sucess_rate.append(done)
                    
                # modify: add by laker: render after the step. important
                show_fre=1
                if FLAGS.render and len(episode_rewards) % 1 == 0: # modify
                    time.sleep(0.3)
                    env.render()
                
                if  len(episode_rewards) % show_fre == 0: # modify
                    if ended==True:
                        print("done?",done,terminal)
                        length_k=show_fre
                        if len(sucess_rate)>length_k:
                            # print("sucess rate",sum(sucess_rate[-length_k:])/len(sucess_rate[-length_k:]))
                            print("sucess rate",sum(sucess_rate)/len(sucess_rate))
                    if episode_step >= FLAGS.max_episode_len:
                        print("max steps",episode_step)

                # collect experience

                # step forward observations
                obs_n = new_obs_n

                # TODO: 这里记录每一轮最大reward
                # record some analysis information
                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew #lastest reward sum for each step of all agent
                    agent_episode_rewards[i].append(rew)
                    agent_rewards[i][-1] += rew #lastest reward sum for each step of each agent

                for discrete_action in range(FLAGS.num_good_agents):
                    # print("######")
                    coordination_reach_times[discrete_action][-1] += info_n[0][discrete_action]
                # add some log and records...
                if ended:
                    # print log for debugging......
                    if len(episode_rewards) % show_fre == 0:
                        print('process {}, episode {}: '.format(os.getpid(), len(episode_rewards)))
                        print("")
                        print("")
                    #laker
                    # time.sleep(3)
                    # reset environment
                    
                    obs_n = env.reset()
                    # reset episode tags
                    episode_step_r.append(episode_step)
                    episode_step = 0
                    episode_rewards.append(0)  # reset sum rewards
                    for idx, a in enumerate(agent_rewards):  # reset each agent's reward
                        a.append(0)
                    agent_episode_rewards = [[0.0] for _ in range(env.n)]
                    for coord_count in coordination_reach_times:  # reset coordination times
                        coord_count.append(0)
            # close sess
            # sess.close()    
            for a in agents: del a
        # do record
        log_path = './exp_result/evaluate_record'
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
        losses_transformation_file_name = log_path + '/' + pa_model_path.split("/")[2] + '_success_rate_mean_step.txt'
        with open(losses_transformation_file_name, 'a') as fp:
            fp.write(pa_model_path.split("/")[3] +"\t\t"+ pe_model_path.split("/")[2]  +"\t\t"+ str(sum(sucess_rate)/len(sucess_rate)) +"\t\t"+ str(np.mean(episode_step_r))+"\n")#predator prey success_rate episode_step
        print("-----------***sucess_rate**********mean(episode_step******-------------")
        print(str(sum(sucess_rate)/len(sucess_rate)) +"\t\t"+ str(np.mean(episode_step_r))+"\n")
        print("-----------******-*****-*****-*****-*****-*****-*****-*****-------------")
# for debug below ..........................................................
def same_session(sess, agents):
    for agent in agents[:FLAGS.num_adversaries]:
        if sess != agent.get_session():
            print("Session error (diff tf session)")
    print("The same session.........................")


if __name__ == '__main__':
    train()
