# -*- coding: utf-8 -*-
import os

import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 使用 GPU 0 or -1 #laker denote
import tensorflow as tf
import numpy as np
import algorithm.common.tf_utils as tf_utils
import time
import random

# read input cmd from standard input device
flags = tf.app.flags

import datetime
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
# Game parameter
# flags.DEFINE_string('env_name', 'predator_prey', 'env used')
flags.DEFINE_string('env_name', 'round_up_base', 'env used') #round_up_base ,round_up_reward_shaping, round_up_reward_shaping_5 ,round_up_reward_shaping_10
flags.DEFINE_bool('render', False, 'whether to render the scenario')
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_integer('num_adversaries', 3, 'num_adversaries')#3
flags.DEFINE_integer('num_good_agents', 1, 'num_good_agents')#1
flags.DEFINE_integer('max_step_before_punishment', 8, 'max_step_before_punishment')#modify 8
flags.DEFINE_bool('reload_prey', True, 'whether to reload the pre-trained prey model')
flags.DEFINE_bool('train_prey',  False,'whether to reload the pre-trained prey model')
flags.DEFINE_string('prey_model_path', './exp_result/prey/model-prey-s',
                    'path of the pre-trained prey model')
flags.DEFINE_bool('reload_predator', False, 'whether to reload the pre-trained predator model')
flags.DEFINE_string('predator_model_path', './exp_result/round_up/pre_train_prey/saved_models/seed_ddpg_81/model-87000',
                    'path of the pre-trained predator model')

# Training parameters
flags.DEFINE_bool('learning', True, 'train the agents')
flags.DEFINE_string('exp_name', '3v1', 'exp_name')
flags.DEFINE_string('predator_policy', 'pgmaddpg', 'predator_policy') #laker
flags.DEFINE_float('rlpl_beta', 11, 'rlpl_beta') #jone   # 0 < rlpl_beta < 1 , if rlpl_beta < 0,eg:-1 pgddpg rlpl_beta dec
# flags.DEFINE_string('predator_policy', 'gasil', 'predator_policy') 
flags.DEFINE_string('prey_policy', 'ddpg', 'prey_policy: [random, fixed, ddpg]')
flags.DEFINE_integer('episodes', 500000, 'maximum training episode')#140000
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

# GASIL
flags.DEFINE_bool('consider_state_action_confidence', True, 'The closer (state, action) to the end state the important')
flags.DEFINE_float('state_action_confidence', 0.8, 'discount factor of (state, action)')
flags.DEFINE_float('state_action_confidence_max', 1., 'discount factor of (state, action)')
flags.DEFINE_integer('gradually_inc_start_episode', 0,
                     'increase parameters start at ${gradually_inc_start_episode} episode')
flags.DEFINE_integer('gradually_inc_within_episode', 12000,
                     'increase parameters in ${gradually_inc_within_episode} episode')
flags.DEFINE_integer('inc_or_dec_step', 1000, 'natural_exp_inc parameter: inc_step')
flags.DEFINE_float('d_lr', 0.001, 'discriminator learning rate')
flags.DEFINE_float('imitation_lambda', 0., 'coefficient of imitation learning')
flags.DEFINE_float('imitation_lambda_max', 1., 'maximum coefficient of imitation learning')
flags.DEFINE_integer('train_discriminator_k', 1, 'train discriminator net k times at each update')
flags.DEFINE_integer('gan_batch_size', 8, 'batch_size of training GAN')#laker 8

# experience replay
flags.DEFINE_integer('buffer_size', 300000, 'buffer size')
flags.DEFINE_integer('min_buffer_size', 30000, 'minimum buffer size before training')#laker 30000
flags.DEFINE_integer('positive_buffer_size', 32, 'buffer size')#laker 32
flags.DEFINE_integer('min_positive_buffer_size', 32, 'min buffer size before training')#laker 32
# prioritized
flags.DEFINE_bool('prioritized_er', False, 'whether to use prioritized ER')
flags.DEFINE_float('alpha', 0.6, 'how much prioritization is used (0 - no prioritization, 1 - full prioritization)')
flags.DEFINE_float('beta', 0.4, 'To what degree to use importance weights (0 - no corrections, 1 - full correction)')

# Net structure
flags.DEFINE_integer('num_units', 128, 'layer neuron number')#laker 32
flags.DEFINE_integer('num_units_ma', 128, 'layer neuron number for multiagent alg')#laker 64
flags.DEFINE_integer('h_layer_num', 2, 'hidden layer num')

FLAGS = flags.FLAGS  # alias
# Model saving dir
if FLAGS.predator_policy[0:6] =="pgddpg" or FLAGS.predator_policy[0:8] =="pgmaddpg":
    if FLAGS.rlpl_beta>=0 and FLAGS.rlpl_beta<=1:
        suffix = "_{}".format(FLAGS.rlpl_beta) 
    elif FLAGS.rlpl_beta<0:
        suffix = "_dec"
    else :
        print("*-*-*-*--rlpl_beta error*-*-*-*-*-*-*-*-")
else:
    suffix = ''

flags.DEFINE_string('model_save_dir', './exp_result/{}/saved_models/{}/seed_{}/{}/model'.format(FLAGS.exp_name,FLAGS.env_name, FLAGS.predator_policy+suffix,DATETIME),
                    'Model saving dir')
flags.DEFINE_string('learning_curve_dir', './exp_result/{}/learning_curves/{}/seed_{}/{}'.format(FLAGS.exp_name,FLAGS.env_name, FLAGS.predator_policy+suffix,DATETIME),
                    'learning_curve_dir')
flags.DEFINE_string('tensorboard_dir', './exp_result/{}/tensorboard_dir/{}/seed_{}/{}'.format(FLAGS.exp_name,FLAGS.env_name, FLAGS.predator_policy+suffix,DATETIME),
                    'tensorboard_dir')
FLAGS = flags.FLAGS  # alias
# init tf summaries
summary_path = FLAGS.learning_curve_dir
# init tf summaries
tensorboard_dir = FLAGS.tensorboard_dir
# init tf model_path
model_path = FLAGS.model_save_dir
print('||||||||||||||||||--------Log Path------||||||||||||||||||')
print('summary_path', summary_path)
print('tensorboard_dir', tensorboard_dir)
print('model_path', model_path)
print('||||||||||||||||||--------Log Path------||||||||||||||||||')
# print('||||||',FLAGS.flag_values_dict())
# dic = FLAGS.flag_values_dict()

# print(type(FLAGS.flag_values_dict()))
# for attr, value in FLAGS.flag_values_dict().items():
#     print("{}={}".format(attr, value))

if __name__ == '__main__':
    rew_file_name = 'presetting_parameters_log.txt'
    with open(rew_file_name, 'w') as fp:
        # pickle.dump(kwargs['agent_rewards'], fp)
        for attr, value in FLAGS.flag_values_dict().items():
            # print("{}={}".format(attr, value))
            fp.write("{}\t{}\n".format(attr, value))