import numpy as np
from env.multiagent.core import World, Agent, Landmark
from env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, num_adversaries,num_good_agents):

       # self.good_colors = [np.array([0x66, 0x00, 0x66,]) / 255, np.array([0x00, 0x99, 0xff]) / 255, np.array([0x66,0xff,0xff])/255]
        self.good_colors = [np.array([0x00, 0x99, 0xff]) / 255] * num_good_agents

        # action number
        self.num_good_agents = num_good_agents
        # agent number
        self.num_adversaries = num_adversaries

        # self.prey_init_pos = np.random.uniform(-1, +1, 2)

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = self.num_good_agents
        # good_sizes = [0.06, 0.055, 0.05]
        # good_sizes = [0.05] * 3 # modify

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
            agent.spread_rewards = True if i < num_adversaries else False
            # agent.size = 0.03 if agent.adversary else 0.03
            agent.size = 0.075 if agent.adversary else 0.05
            # agent.accel = 1.0 if agent.adversary else 1.3
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.max_speed = 0.5 if agent.adversary else 0.65
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False 
            landmark.size = 0.05
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
        for agent in world.agents:
            # print('world.dim_p: ', world.dim_p)  # 2
            # print('world.dim_c: ', world.dim_c)  # 2
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
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
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

    def is_collision(self, predator, prey, collision_level=Agent.distance_spread[1]):
        delta_pos = predator.state.p_pos - prey.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = prey.size + predator.size * collision_level
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
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    # print('collision...')
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                # print('agent is out...')
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        # TODO: if bounded, then hidden this code.
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

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

    def set_pressed(self, prey):
        prey.pressed = True

    def set_unpressed(self, prey):
        prey.pressed = False

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

    def set_arrested_pressed_watched(self, world):
        # print('set_arrested_pressed_watched')
        
        #modify by go
        return 0
        print("#########################")
        ##


        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
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
                    if collision_num == 2:
                        self.set_arrested(prey)
                        # print("Setting arrested....")
                    elif collision_num == 1:
                        self.set_pressed(prey)
                        # print("Setting pressed....")
                    elif collision_num == 0:
                        self.set_unarrested(prey)
                        self.set_unpressed(prey)

                elif dis_idx == 1:
                    if collision_num >= 1:
                        self.set_watched(prey)
                    else:
                        self.set_unwatched(prey)



    def adversary_reward(self, agent, world):# modify
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
                # rew =rew+ 0.4-0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents]) #modify
        
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        

        #add by go
        agents = self.good_agents(world)
        adversaries = self.adversaries(world) 
        for ag_ in agents: # for same good, if collision occurs at least once 
            whether_attacked=0
            count_attacked=0
            for adv_ in adversaries: # the number of arrounded
                whether_attacked=max(whether_attacked,self.is_collision(ag_, ag_))#
                delta_pos=ag_.state.p_pos - adv_.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                surrounded_level=1.2# agent.distance_spread[1] #2.25
                dist_min = ag_.size + adv_.size * surrounded_level
                if dist<dist_min:
                    #print("distance-",dist)
                    count_attacked +=1        
                    
            if whether_attacked:#collision occurs at least once 
                rew = rew+10*(count_attacked-1)
            
            # 持续接近多少步？
            near_step=0

            if count_attacked==self.num_adversaries:
                print("all near...",count_attacked,self.num_adversaries)
                near_step+=1
                print("ok",near_step,adv_.press_down_step)
            
            if near_step>10:
                print("success chasing {} steps".format(near_step))
                self.end_without_supports[0] = True ## adv_.idx
                rew = rew+10*count_attacked
        #add end
        
        return rew
     


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue  # 跳过当前 agent
            comm.append(other.state.c)  # 2
            other_pos.append(other.state.p_pos - agent.state.p_pos)  # 2
            if not other.adversary:  # 我的观察里有被捕食者，添加其速度
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        # 2, 2, 2 * 2, 3 * 2, [2] = 16 / 14

    # Add the end condition.
    def done(self, agent, world):
        # 只要达到了coordination 就终止（不管最优次优）
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # TODO: For each action
        collision_adv = 0
        collision_agents = set()
        for idx, ag in enumerate(agents):  # for each good agent, test whether there is a collision
            for adv in adversaries:
                if self.is_collision(adv, ag):
                    collision_adv += 1
                    collision_agents.add(adv.idx)
        # For coordination
        if len(collision_agents) == self.num_adversaries or any(self.end_without_supports):#modify
        # if len(collision_agents) == self.num_adversaries or any(self.end_without_supports):#modify
            # for ag in adversaries:
            #     ag.print_info()
            # for ag in agents:
            #     ag.print_info()
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
