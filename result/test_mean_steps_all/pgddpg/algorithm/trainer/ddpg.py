import numpy as np
import tensorflow as tf

import algorithm.common.tf_utils as U
from algorithm.trainer import AgentTrainer
from algorithm.common.distributions2 import make_pdtype
from algorithm.common.reinforce_utils import make_update_exp
from algorithm.prioritized_experience_replay_buffer.utils import add_episode
from algorithm.misc import *

#modified by laker, 2020.06

FLAGS = tf.app.flags.FLAGS  # alias

def p_train(make_obs_ph, act_space, p_index, p_func, q_func,plan_func, optimizer, grad_norm_clipping=None, local_q_func=True,
            scope="trainer", reuse=None, layer_norm=True):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype = make_pdtype(act_space)
        # set up placeholders
        obs_ph = make_obs_ph
        act_ph = act_pdtype.sample_placeholder([None], name="action") 
        
        weight_ph = tf.placeholder(tf.float32, [None], name="important_weight")
        n_steps_annealings = tf.placeholder(tf.float32, shape=[], name='n_steps_annealing')#
        step_explore = tf.placeholder(tf.float32, shape=[], name='step_explore')#
        explore_direction = tf.placeholder(tf.float32, shape=[], name='explore_direction')#
        gama_gussian = tf.placeholder(tf.float32, shape=[], name='gama_gussian')#
        
        p_input = obs_ph
        p = p_func(p_input, int(act_pdtype.param_shape()[0]), scope="p_func",
                   num_units=FLAGS.num_units, layer_norm=layer_norm)
        

        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        reg_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.lambda2), p_func_vars)

        # wrap parameters in distribution
        act_pd = act_pdtype.pdfromflat(p)

        # action_sample
        act_no_noise,act_noise = act_pd.sample(deterministic=True,step_explore=step_explore,explore_direction=explore_direction,n_steps_annealings=n_steps_annealings,gama_gussian=gama_gussian)
        
        # p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input = [act_ph] + []
        act_input = act_pd.sample()  # No explore here.# act_pd.mode() sample action from current policy

        train_obs_input = obs_ph
        train_action_input = act_ph

        q_input = tf.concat([obs_ph, act_input], 1)
        q_num_units = FLAGS.num_units  # cell number for ddpg
        
        #critic1 by reward
        q_critic1=q_func(q_input, 1, scope="q_func", reuse=True, num_units=q_num_units, layer_norm=layer_norm)[:, 0] #[:, 0]:from column to line
        
        #critic2 by potential field
        q_critic2=plan_func(q_input, scope="plan_func", reuse=True)
 
        q = q_critic1#
        # q = q_critic1+q_critic2#q_func(q_input, 1, scope="q_func", reuse=True, num_units=q_num_units, layer_norm=layer_norm)[:, 0]
        
        # soft pgddpg
        # mean, variance = tf.nn.moments(act_input, axes=0)
        # variance_loss = -0.0*tf.reduce_mean(variance)

        #for debug
        xxx=[]
        yyy=[]
        zzz=[]

        # maximax q and minimise pg_loss
        pg_loss = -tf.reduce_mean(q)   # pg_loss = -tf.reduce_mean(q * weight_ph)


        loss = pg_loss + reg_loss #+ variance_loss   # loss = pg_loss

        # return
        act = U.function(inputs=[obs_ph]+[n_steps_annealings,step_explore,explore_direction,gama_gussian,], outputs=[act_no_noise,act_noise,n_steps_annealings,step_explore,explore_direction])
        # act = U.function(inputs=[obs_ph], outputs=[act_sample, determin_act_sample])
        p_values = U.function([obs_ph], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype.param_shape()[0]), scope="target_p_func",
                          num_units=FLAGS.num_units, layer_norm=layer_norm)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype.pdfromflat(target_p).sample()# No explore here.
        target_act = U.function(inputs=[obs_ph], outputs=target_act_sample)

        # build optimizer
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[train_obs_input] + [train_action_input] + [weight_ph,n_steps_annealings,],
                           outputs=[pg_loss,xxx,yyy,zzz],
                           updates=[optimize_expr])

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act,
                                             'act_pdtype': act_pdtype}


def q_train(make_obs_ph, act_space, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=True,
            scope="trainer", reuse=None, layer_norm=True):
    with tf.variable_scope(scope, reuse=reuse):
        # create distributions
        # act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        act_pdtype = make_pdtype(act_space)
        # set up placeholders
        obs_ph = make_obs_ph
        # act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        act_ph = act_pdtype.sample_placeholder([None], name="action")
        
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        return_ph = tf.placeholder(tf.float32, [None], name="return")
        dis_2_end_ph = tf.placeholder(tf.float32, [None], name="dis_2_end")
        lambda1_ph = tf.placeholder(tf.float32, shape=[], name='lambda1')
        weight_ph = tf.placeholder(tf.float32, [None], name="important_weight")
        # build q-function input
        train_obs_input = obs_ph
        train_action_input = act_ph

        # q_num_units = FLAGS.num_units_ma  # cell number for maddpg
        if local_q_func:
            q_input = tf.concat([obs_ph, act_ph], 1)
            q_num_units = FLAGS.num_units  # cell number for ddpg

        q = q_func(q_input, 1, scope="q_func", num_units=q_num_units, layer_norm=layer_norm)[:, 0]

        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        reg_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.lambda2), q_func_vars)

        # TODO: for using prioritized replay buffer, adding weight
        td_0 = target_ph - q
        q_loss_td_0 = -tf.reduce_mean(weight_ph * tf.stop_gradient(td_0) * q)
        q_td_0_loss = tf.reduce_mean(weight_ph * tf.square(td_0))

        # TODO: 这里对正向差异 (R-Q) > 0 做截断
        # mask = tf.where(return_ph - tf.squeeze(q) > 0.0,
        #                 tf.ones_like(return_ph), tf.zeros_like(return_ph))
        # TODO: add dis_2_end: return_confidence_factor
        confidence = tf.pow(FLAGS.return_confidence_factor, dis_2_end_ph)
        # td_n = (return_ph * confidence - q) * mask
        # TODO: add clip here...
        # td_n = tf.clip_by_value(return_ph * confidence - q, 0., 4.) * mask
        td_n = tf.clip_by_value(return_ph * confidence - q, 0., 4.)
        q_loss_monte_carlo = -tf.reduce_mean(weight_ph * tf.stop_gradient(td_n) * q)
        # q_td_n_loss = tf.reduce_mean(weight_ph * tf.square((return_ph * confidence - q) * mask))
        q_td_n_loss = tf.reduce_mean(weight_ph * tf.square(td_n))

        loss = q_loss_td_0 + lambda1_ph * q_loss_monte_carlo + reg_loss #lambda1_ph=0
        # loss = q_td_0_loss + lambda1_ph * q_td_n_loss + lambda2_ph * margin_classification_loss + reg_loss

        q_values = U.function([train_obs_input] + [train_action_input], q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=q_num_units, layer_norm=layer_norm)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function([train_obs_input] + [train_action_input], target_q)

        # build optimizer
        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[train_obs_input] + [train_action_input] + [target_ph] + [weight_ph,
                                                                                        lambda1_ph, dis_2_end_ph,
                                                                                        return_ph,
                                                                                        ],
                           outputs=q_loss_td_0,
                           # outputs=[loss, q_loss_td_0, q_loss_monte_carlo, margin_classification_loss, reg_loss,
                           #          q_td_0_loss, q_td_n_loss],
                           updates=[optimize_expr])

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class AgentTrainer_DDPG(AgentTrainer):

    def __init__(self, name, learn_model,plan_model, obs_shape, act_space, agent_index, buffer,
                 local_q_func=True):
        self.name = name
        # self.n = len(obs_shape_n)
        # print('AgentTrainer_DDPG_APF n: ', self.n)
        self._is_deterministic = False #noise action to explore
        self.agent_index = agent_index
        # control the important sampling weight
        self.beta = FLAGS.beta

        obs_ph = []
        self.learn_model = learn_model
        self.plan_model= plan_model
        self.act_space = act_space
        self.obs_shape = obs_shape
        print('AgentTrainer_DDPG_APF obs_shape: ', self.obs_shape)
        self.show_start_traing_updating=True
        self.n_steps_annealing=FLAGS.n_steps_annealing
        self.delta_n_steps_annealing=FLAGS.delta_n_steps_annealing
        obs_ph=tf.placeholder(tf.float32, shape=(None,self.obs_shape))
        # obs_ph.append(U.BatchInput(obs_shape, name="observation").get())
        print('obs_ph: ', obs_ph)
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph=obs_ph,
            act_space=act_space,
            q_index=agent_index,
            q_func=learn_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.ddpg_qlr, beta1=0.5, beta2=0.9),
            grad_norm_clipping=0.5,  # oroginal 0.5,
            local_q_func=local_q_func,
            layer_norm=False # 
        )

        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph=obs_ph,
            act_space=act_space,
            p_index=agent_index,
            p_func=learn_model,
            q_func=learn_model,
            plan_func=plan_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.ddpg_plr, beta1=0.5, beta2=0.9),
            grad_norm_clipping=0.5, # original 0.5,
            local_q_func=local_q_func,
            layer_norm=False # 
        )

        # Create experience buffer
        self.replay_buffer = buffer
        # Coefficient of n-step return
        self.lambda1 = FLAGS.lambda1
        #
        self.running_episode = []
        self.replay_sample_index = None
        print('AgentTrainer_DDPG_APF {} built success...'.format(self.agent_index))

    def toggle_deterministic(self):
        self._is_deterministic = not self._is_deterministic

    def get_actions(self, observations, single=False, step_explore=0.0, explore_direction=1.0,gama_gussian=0.0):
        if step_explore<0.5:#no explore, step_explore=0.0
            return self.act((*([observations] + [self.n_steps_annealing,step_explore,explore_direction,gama_gussian,])))[0][0]
        else:#explore, step_explore=1.0
            return self.act((*([observations] + [self.n_steps_annealing,step_explore,explore_direction,gama_gussian,])))[1][0]
          
    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.running_episode.append([obs, act, rew, new_obs, done])
        if done:
            add_episode(self.replay_buffer, self.running_episode, gamma=FLAGS.gamma)
            self.running_episode = [] #temp
            # print("So far, the length of the relpay buffer is :",len(self.replay_buffer))

    def preupdate(self):
        self.replay_sample_index = None

    @property
    def pool(self):
        return self.replay_buffer

    def decay_parameters(self):
        pass

    def do_training(self, agent, train_step, end):
        self.decay_parameters()
        if not self.is_exploration_enough(FLAGS.min_buffer_size):
            return
        if train_step % FLAGS.train_every!=0: #or if (1-end): #train after one episode
            return
        if self.show_start_traing_updating:
            print("                               ##########---------------------------------------############")
            print("                               ##########Warm up end. Start traing and updating.############")
            print("                               ##########---------------------------------------############")
            self.show_start_traing_updating=False
        return self.update(agent)

    def is_exploration_enough(self, min_pool_size):
        return len(self.pool) >= min_pool_size

    def update(self, agent):
        '''
        For update using uniform experience replay (using normal tuple data)
        :param agents:
        :return:
        '''
        self.replay_sample_index = self.replay_buffer.make_index(FLAGS.batch_size)
        # collect replay sample from all agents
        obs = []
        obs_next = []
        act = []
        idxes = self.replay_sample_index

        # obs, act, rew, obs_next, done, dis_2_end, R = agent.replay_buffer.sample_index(idxes)
        # print("agent.obs:",obs)
        # same as below

        obs, act, rew, obs_next, done, dis_2_end, R = self.replay_buffer.sample_index(idxes)
        # print("self.obs:",obs)
        # train q network
        num_sample = 1
        target_q = 0.0

        for i in range(num_sample):
            target_act_next = agent.p_debug['target_act'](obs_next)
            # target_act_next = agent.p_debug['target_act'](*([obs_next] + [self.n_steps_annealing]))

            target_q_next = self.q_debug['target_q_values'](*([obs_next] + [target_act_next]))
            target_q += rew + FLAGS.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        # TODO: checking: W_placeholder=1 when PER is not used.
        q_loss = self.q_train(*([obs] + [act] + [target_q] + [np.ones_like(target_q),0,dis_2_end,R,]))


        self.n_steps_annealing=self.n_steps_annealing+self.delta_n_steps_annealing
        
        p_loss,xxx,yyy,zzz= self.p_train(*([obs] + [act] + [np.ones_like(target_q),self.n_steps_annealing,]))
        # print('xxx:',xxx)
        # print('yyy:',yyy)
        # print('zzz:',zzz)  
        if 1:#iteration % FLAGS.target_update_interval == 0:
            self.p_update()
            self.q_update()

        # return [q_loss, p_loss], None
        return [q_loss, p_loss]


    def get_session(self):
        return tf.get_default_session()
