import numpy as np
from env.multiagent.core import World, Agent, Landmark
from env.multiagent.scenario import BaseScenario

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS  # 

class Scenario(BaseScenario):
    rewards_base = 10
    adv_policies = "ddpg"

    def make_world(self):
        print("now",self.max_step_before_punishment)
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = self.num_good_agents

        num_adversaries = self.num_adversaries
        # 0 ~ num_adversaries: are adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        # add agents
        world.agents = [Agent(i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.pushable=True if i < num_adversaries else True #laker
            agent.spread_rewards = True if i < num_adversaries else False
            agent.size = 0.035 if agent.adversary else 0.035 #modify by laker
        
            # agent.u_range = 1.0 if agent.adversary else 0.5 #modify by laker: the maximum speed of prey
            ######success#agent.size = 0.035 if agent.adversary else 0.075 #modify by laker
            agent.accel = 1.0 if agent.adversary else 1.0
            agent.max_speed = 0.5 if agent.adversary else 0.7
            #######success# agent.accel = 2.0 if agent.adversary else 2.0
            #######success# agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # for reward calculating...
        self.adversary_episode_max_rewards = [0] * self.num_adversaries
        self.end_without_supports = [False] * self.num_adversaries

        # print("reset world....")
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = self.good_colors[i - self.num_adversaries] if not agent.adversary else np.array(
                [0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        init_pos = []
        for agent in world.agents:
            if agent.adversary:
                agent.reset_predator()
            else:
                agent.reset_prey()

            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

            agent.state.c = np.zeros(world.dim_c)
            # print('agent state: ', agent.state)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                while 1: 
                    n = len(init_pos)
                    pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                    for init in init_pos:
                        if init[0]-0.035 < pos[0] <init[0]+0.035 and init[1]-0.035 < pos[1] <init[1]+0.035: 
                            pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                        else:
                            n = n-1
                    if n == 0:
                        break
                # pos = []
                init_pos.append(pos)
                landmark.state.p_pos = init_pos[-1]
                landmark.state.p_vel = np.zeros(world.dim_p)
        

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, predator, prey, collision_level=Agent.distance_spread[1],printing=False):
        delta_pos = predator.state.p_pos - prey.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = prey.size + predator.size * collision_level

        if printing:
            print("delta_pos:",delta_pos,dist,dist_min)
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    # define the reward (coordination)
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        # doesn't change (rewarded only when there is a real collision)
        step_penalize = 0.1
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))

        # if agent.collide:
        #     for a in adversaries:
        #         if self.is_collision(a, agent):
        #             # print('collision...')
        #             rew -= 10 #laker ori:10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                # print('agent is out...')
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        #laker
        # TODO: if bounded, then hidden this code.
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        return rew+step_penalize
        # return rew 

    def return_collision_good_agent_idx(self, predator, good_agents, distance_range):
        for idx, good in enumerate(good_agents):
            if self.is_collision(predator, good, distance_range):
                return idx
        return -1

    def set_arrested(self, prey):
        prey.arrested = True

    def set_unarrested(self, prey):
        prey.arrested = False

    def set_watched(self, prey):
        prey.watched = True

    def set_unwatched(self, prey):
        prey.watched = False

    def set_pressed(self, prey,collision_num):
        #laker:
        prey.collision_num=collision_num
        prey.pressed = True

    def set_unpressed(self, prey):
        prey.pressed = False
        #laker
        prey.collision_num=0

    def set_predator_pressed(self, predator, prey_idx):
        if predator.press_prey_idx == -1:  # 没抓过
            predator.press_prey_idx = prey_idx
            predator.press_down_step += 1
            # print("predator ", predator.idx, ": ", predator.press_down_step, ' prey: ', prey_idx)
        elif predator.press_prey_idx == prey_idx:  # 抓了同一个
            predator.press_down_step += 1
            # print("predator ", predator.idx, ": ", predator.press_down_step, ' prey: ', prey_idx)
        else:  # 没抓同一个
            predator.press_prey_idx = prey_idx
            predator.press_down_step = 1

    def release_predator_pressed(self, predator, prey_idx):
        if predator.press_prey_idx == prey_idx:
            predator.reset_predator()

    def set_arrested_pressed_watched(self, world):#judge each step
        # print('set_arrested_pressed_watched')
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # Agent.distance_spread[1:]:  [2.25, 6]
        for dis_idx, distance_range in enumerate(Agent.distance_spread[1:]):
            for prey_idx, prey in enumerate(good_agents):
                collision_num = 0
                for predator in adversaries:
                    if self.is_collision(predator, prey, collision_level=distance_range):
                        collision_num += 1
                        # 处理 predator 状态
                        if dis_idx == 0:
                            self.set_predator_pressed(predator, prey_idx)
                        elif dis_idx == 1:
                            pass
                    else: # 没抓当前这个
                        if dis_idx == 0:
                            self.release_predator_pressed(predator, prey_idx)

                if dis_idx == 0:
                    if collision_num >= self.successed_round_up:#modify by laker
                        self.set_arrested(prey)
                        # print("Setting arrested....")
                    elif collision_num >= 1:#modify by laker: ==1
                        self.set_pressed(prey,collision_num)
                        # print("Setting pressed....")
                    elif collision_num == 0:
                        self.set_unarrested(prey)
                        self.set_unpressed(prey)
                #modify by laker
                elif dis_idx == 1:
                    if collision_num >= 1:
                        self.set_watched(prey)
                    else:
                        self.set_unwatched(prey)

    # define the reward (coordination) for agent
    def adversary_reward(self, agent, world):
        step_penalize = 0
        # step_penalize = 0
        # Adversaries are rewarded for collisions with agents
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        #
        shape_rew=0 #total and same
        shape = False
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:

                # shape_rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in good_agents])
                #modify by laker
                for a in good_agents:
                    dist_=np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    if dist_<0.5:
                        shape_rew += 5*(0.5-dist_)
           
        #        
        # collision avoidance between praydator
        # for predator_ in adversaries:
        #     partners = [a for a in adversaries if a != predator_]
        #     for partners_ in partners:
        #         if self.is_collision(predator_, partners_, collision_level=1):
        #             shape_rew=shape_rew-0.1

        if agent.collide:
            # Agent.distance_spread[1:]:  [2.25, 6]
            # for dis_idx, distance_range in enumerate(agent.distance_spread[1:]):#modify by laker#agent.distance_spread[1:]
            if 1: #modify by laker
                dis_idx=0 #modify by laker
                # distance_range=1.0 #modify by laker
                for prey_idx, prey in enumerate(good_agents):
                    collision_num = 0
                    self_collision = 0
                    for predator in adversaries:
                        if self.is_collision(predator, prey, collision_level=1.0):
                            collision_num += 1
                            if predator == agent:  # 自己
                                self_collision += 1
                    # more catching,more reward 
                    if dis_idx == 0 and self_collision==1:#laker no reward here
                        shape_rew=0  #0.1*collision_num*self.rewards_base
                        

                    if dis_idx == 0 and self_collision==1 and collision_num >= self.successed_round_up:  
                        # addition reward 增长
                        rew = self.rewards_base
                        #结束
                        self.end_without_supports[agent.idx] = True
                        # rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                        # self.adversary_episode_max_rewards[agent.idx] = rew
                        return rew+shape_rew

                    elif dis_idx == 0 and self_collision==1 and collision_num <self.successed_round_up and agent.press_down_step > self.max_step_before_punishment:
                                            
                        # TODO 队友没有在几步之内救自己，受到惩罚并且结束
                        # modify by laker,受到惩罚但不结束
                        # self.end_without_supports[agent.idx] = True
                        # 惩罚 reward
                        if collision_num==1:
                            rew = 0
                        elif collision_num>=2:
                            rew = 0
                        return rew+shape_rew
                    elif dis_idx == 0 and self_collision==0 and collision_num <self.successed_round_up and agent.press_down_step > self.max_step_before_punishment:
                                            
                        # TODO 自己没救队友，受到惩罚并且结束
                        # modify by laker,受到惩罚但不结束
                        # self.end_without_supports[agent.idx] = True
                        # 惩罚 reward
                        if collision_num==1:
                            rew = 0
                        elif collision_num>=2:
                            rew = 0
                        else:
                            rew = 0
                        return rew+shape_rew
                    else:
                        return shape_rew
                        
        return step_penalize

    def observation(self, agent, world): 
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_size = 0  #jone.add 2020-12.29
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                entity_size = entity.size #jone.add 2020-12.29  

        other_pos = []
        other_force = []
        other_vel = [] 
        repulsive_force=np.zeros(2)

        # partner's probabelly intention
        for partner in adversaries: 
            if partner is agent: continue  # 跳过当前 agent 
            s_order=good_agents[0].state.p_pos- partner.state.p_pos+(good_agents[0].state.p_vel-partner.state.p_vel)*world.dt                  
            attractive_force_order=2*s_order/(world.dt*world.dt) #未归一化
            force_order = attractive_force_order
            if np.max(np.fabs(force_order)) > 1:#force, force_x belong [-1,1],force_y belong [-1,1]
                force_order = force_order/np.max(np.fabs(force_order)) 
            other_force.append(force_order) #归一化

        # potentional feild force
        for other in world.agents: 
            if other is agent: continue  # 跳过当前 agent 
            if other.adversary:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                force_direction=agent.state.p_pos-other.state.p_pos
                dist_temp1=np.linalg.norm(force_direction) #两个位置之间的距离-->np.sum(np.square(force_direction)))

                margin=0#agent.size 
                if dist_temp1<agent.size+other.size+margin:#consider repulsive force
                    dist_temp2=dist_temp1-agent.size
                    if dist_temp2==0 or dist_temp2 <= 0.000001:
                        dist_temp2 = 0.000001
                    # repulsive_force=repulsive_force+ unit_force_direction*1*pow(pow(1/dist_temp2,0.3)-pow(1/margin,0.3),2)
                    repulsive_force=repulsive_force + force_direction*10*pow(agent.size/dist_temp2,2)  # force_attractive
                else:
                    pass
            # print("repulsive_force",repulsive_force)
            if not other.adversary:  # 我的观察里有被捕食者，添加其速度 

                other_vel.append(other.state.p_vel) 

                s_order=other.state.p_pos- agent.state.p_pos+(other.state.p_vel-agent.state.p_vel)*world.dt                  
                attractive_force_order=2*s_order/(world.dt*world.dt) 

                force_order=attractive_force_order+ 0*repulsive_force

                if np.max(np.fabs(force_order)) > 1:#force, force_x belong [-1,1],force_y belong [-1,1]
                    force_order = force_order/np.max(np.fabs(force_order)) 

                other_pos.append(force_order) #important feature
                other_pos.append(s_order)#potential better than other.state.p_pos - agent.state.p_pos
                # other_pos.append(other.state.p_pos - agent.state.p_pos)#potential 
                
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos +  other_vel + other_pos)
        if agent.adversary:  
            if self.adv_policies == "pgddpg_exp"  or self.adv_policies == "ddpg":
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + [[entity_size]] + other_force +  other_vel + other_pos)
            else:
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + [[entity_size]] +  other_vel + other_pos)
        else:
            # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + [[entity_size]] +  other_vel + other_pos)
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +  [] +  other_vel + other_pos)

    # Add the end condition.
    def done(self, agent, world,printing):
        # 只要达到了coordination 就终止（不管最优次优）
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # TODO: For each action
        collision_adv = 0
        collision_agents = set()

        for idx, ag in enumerate(agents):  # for each good agent, test whether there is a collision
            for adv in adversaries:
                if self.is_collision(adv, ag, collision_level=1):#modify by laker: add collision_level=1. important here
                    collision_adv += 1
                    collision_agents.add(adv.idx)
                    if printing:
                        print("all:",ag.state.p_pos,adv.state.p_pos,self.is_collision(adv, ag, collision_level=1,printing=True))
                        print("collision_adv",collision_adv)
        # For coordination
        # if len(collision_agents) == self.num_adversaries or any(self.end_without_supports):
        

        if collision_adv == self.num_adversaries or any(self.end_without_supports): 
            return True
        return False

    def collision_number(self, agent, world):
        # 只要达到了coordination 就终止（不管最优次优）
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        result = {idx: 0 for idx in range(self.num_good_agents)}
        # TODO: For each action
        for idx, ag in enumerate(agents):  # for each good agent, test whether there is a collision
            collision_adv = 0
            for adv in adversaries:
                if self.is_collision(ag, adv):
                    collision_adv += 1
            # For coordination
            if collision_adv == 2:
                result[idx] = 1
        return result
