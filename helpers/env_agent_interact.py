from helpers.agent.multi_agent import MultiAgent
from helpers.env.env_utils import Init_Env
from itertools import count
import numpy as np
import torch
import glob
import time
import os
import gc
import sys
LEAVE_PRINT_EVERY_N_SECS = 300

class Env_Agent_Mix:
    # initializerrrrrrr
    def __init__(self, filename:str, training_info:dict, policy_info:dict, 
                value_info:dict, buffer_info:dict, seed:int, optim_update:int,
                checkpoints:int, storage_loc:str):
        # create environment
        self.env_class = Init_Env(os.path.join(filename))

        # initialize environment
        self.env, self.brain_name = self.env_class.make_env_func(seed)

        # get state_size, number of agents, number of actions
        self.state_size, self.num_agents, \
        self.num_actions = self.env_class.reset(self.env, self.brain_name,
                                                train_mode=False, flag=True)
    
        # environment bounds
        self.low_bounds = np.array([-1] * self.num_actions)
        self.high_bounds = np.array([1] * self.num_actions)

        # seed
        self.seed = seed
        self.optim_update = optim_update

        # general training information
        self.update_steps = training_info["update_every_steps"]
        self.warm_up_batch = training_info["n_warmup_batches"]
        self.weight_mix_ratio = training_info["weight_mix_ratio"]

        # initialize multi agents
        self.multi_agent_init(policy_info, value_info, buffer_info)

        # storage information
        self.checkpoints = checkpoints
        self.storage_loc = storage_loc
        
    # multiple agents
    def multi_agent_init(self, policy_info:dict, value_info:dict, buffer_info:dict):
        general_info = {"weight_mix_ratio": self.weight_mix_ratio,
                        "bounds": (self.low_bounds, self.high_bounds),
                        "state_size": self.state_size,
                        "num_agents": self.num_agents,
                        "num_actions": self.num_actions}
        
        # creating the agents
        self.multi_agents = MultiAgent(general_info, policy_info, value_info, buffer_info, self.seed)        
        
    # interaction step
    def env_agent_inter_step(self, states:np.ndarray, min_samples:int):
        actions = []
        # take random walks to initialize the and warm up the network
        if self.multi_agents.buffer.storage_length() < min_samples:
            for (agent, state) in zip(self.multi_agents.agents, states):
                # append to 
                actions.append(agent.policy_model.select_random_action(state))
        else:
            for (agent, state) in zip(self.multi_agents.agents, states):
                # append to 
                actions.append(agent.policy_model.select_action(state))

        # take step and obtain rewards
        parameters = self.env_class.step(self.env, self.brain_name, np.array(actions))
        
        dones_a = []
        ## prep parameters to be stored in the array
        for idx in range(self.num_agents):
            is_truncated = parameters["max_reached"][idx]
            dones_a.append(parameters["dones"][idx] and not is_truncated) 

        # store in buffer array -> (states, action, reward, next_state, done)
        self.multi_agents.buffer.store(states, np.array(actions),
                                       parameters["rewards"], 
                                       parameters["next_states"],
                                       parameters["dones"])

        # tracking parameters
        self.episode_timestep[-1] += 1       
        self.episode_exploration[-1] += np.array([agent.policy_model.exploration_ratio\
                                                for agent in self.multi_agents.agents]).mean()

        return parameters["next_states"], parameters["rewards"], parameters["dones"]

    # interaction step
    def env_agent_inter_step_eval(self, states:np.ndarray):
        actions = []
        for (agent, state) in zip(self.multi_agents.agents, states):
            # append to 
            actions.append(agent.policy_model.select_greedy_action(state))

        # take step and obtain rewards
        parameters = self.env_class.step(self.env, self.brain_name, np.array(actions))
        
        return parameters["next_states"], parameters["rewards"], parameters["dones"]

    # evaluation mode
    def evaluate(self, n_episodes=1, stage="train"):
        rs = []
        rs_per_agent = []
        for episode in range(n_episodes):
            states = self.env_class.reset(self.env, self.brain_name, 
                                                train_mode=False, flag=False)
            dones = False
            reward_tracker = np.zeros(self.num_agents)
            
            for idx in count():
                # combines actions for each agent
                states, rewards, dones = self.env_agent_inter_step_eval(states)                
                # store in array
                reward_tracker += np.array(rewards)

                if True in dones: 
                    break
            # append rewards
            if stage.lower() =="train":
                rs.append(reward_tracker.mean())
            else:
                rs_per_agent.append(reward_tracker)
                rs.append(reward_tracker.mean())

        # return mean and standard deviation
        if stage.lower() =="train":
            return np.mean(rs), np.std(rs)
        else:
            return rs, np.std(rs), np.array(rs_per_agent)


    def train(self, env_info:dict):
        # intialize constants
        goal_mean_100_reward = env_info["goal_mean_100_rewards"]
        max_minutes  = env_info["max_minutes"]
        max_episodes = env_info["max_episodes"] 
        gamma        = env_info["gamma"]

        # initialize tracking parameters
        self.evaluation_scores = []
        self.episode_timestep  = []
        self.episode_seconds   = []
        self.episode_reward    = []
        self.episode_exploration = []
        self.reward_tracker = np.zeros((max_episodes, self.num_agents))

        # loop and tracking parameters
        training_start, last_debug_time = time.time(), float("-inf")
        training_time = 0
        total_steps = 0

        # result for storage
        results = np.empty((env_info["max_episodes"] , 7))
        fin_episode = 0
        # storage training loop
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            
            # refresh environment
            states = self.env_class.reset(self.env, self.brain_name, 
                                          train_mode=True, flag=False)

            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)
            
            # warmup samples
            min_samples = self.warm_up_batch * self.multi_agents.batch_size

            for step in count():
                # get the next state, and terminal conditions
                states, rewards, dones = self.env_agent_inter_step(states, min_samples)
                self.reward_tracker[episode-1] += np.array(rewards)

                if self.multi_agents.buffer.storage_length() > min_samples:

                    for _ in range(self.optim_update):
                        # extract from memory
                        idx_batch, memory_batch, ISweights = self.multi_agents.buffer.sample()
                        # optimize agents
                        self.multi_agents.optimize(idx_batch, memory_batch, ISweights, gamma)

                        # update network
                        if np.sum(self.episode_timestep) % self.update_steps == 0:
                                self.multi_agents.update_networks_multi()

                if True in dones:
                    self.episode_reward.append(np.max(self.reward_tracker[episode-1]))
                    gc.collect()
                    break

            # stat tracking
            episode_elapsed = time.time() - episode_start
            training_time += episode_elapsed
            self.episode_seconds.append(episode_elapsed)       

            # evaluation 
            evaluation_score, _ = self.evaluate()
            for idx, agent in enumerate(self.multi_agents.agents):
                self.save_checkpoint(episode - 1, agent.policy_model, idx)

            total_steps = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

           # mean and std calculations
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])

            min_100_reward = np.min(self.episode_reward[-100:])
            max_100_reward = np.max(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])

            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            min_100_eval_score = np.min(self.evaluation_scores[-100:])
            max_100_eval_score = np.max(self.evaluation_scores[-100:])
            
            lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            results[episode - 1] = total_steps, mean_100_reward,\
                                   min_100_reward, max_100_reward, \
                                   mean_100_eval_score, min_100_eval_score, \
                                   max_100_eval_score

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            # termination criteria check
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

            #print debug message
            self.debug_message(episode, total_steps, mean_10_reward, std_10_reward,
                               mean_100_reward, std_100_reward, mean_100_exp_rat, 
                               std_100_exp_rat, mean_100_eval_score, std_100_eval_score,
                               training_start)

            if training_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                fin_episode = episode
                break

        # re-evaluate for 100 steps
        final_eval_score, score_std, agent_scores = self.evaluate(n_episodes=100, stage="fin")
        wallclock_time = time.time() - training_start

        # print final message post evalualuation
        self.final_message(np.mean(final_eval_score), score_std, training_time, wallclock_time)

        # clean up the checkpoints
        self.get_cleaned_checkpoints()

        self.env.close() ; del self.env

        return np.array(results), np.array(final_eval_score), np.array(agent_scores),\
               training_time, wallclock_time, fin_episode

    def final_message(self, final_eval_score, score_std, training_time, wallclock_time):
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
               ' {:.2f}s wall-clock time.\n'.format(final_eval_score, score_std, 
                                                   training_time, wallclock_time))
        

    def debug_message(self, episode, total_steps, mean_10_reward, std_10_reward,
                      mean_100_reward, std_100_reward, mean_100_exp_rat, 
                      std_100_exp_rat, mean_100_eval_score, std_100_eval_score,
                      training_start):
        # message string
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
        debug_message = 'el {}, ep {:04}, ts {:07}, '
        debug_message += 'ar_10 ts {:05.1f} \u00B1 {:05.1f}, '
        debug_message += 'ar_100 ts {:05.1f} \u00B1 {:05.1f}, '
        debug_message += 'ex 100 {:02.1f} \u00B1 {:02.1f}, '
        debug_message += 'ev {:05.1f} \u00B1 {:05.1f}'
        debug_message = debug_message.format(elapsed_str, episode - 1, total_steps,
                                             mean_10_reward, std_10_reward, 
                                             mean_100_reward, std_100_reward,
                                             mean_100_exp_rat, std_100_exp_rat,
                                             mean_100_eval_score, std_100_eval_score)
        print(debug_message)

    def save_checkpoint(self, episode_idx, model, rank):
        root_dir = os.path.join(self.storage_loc, "checkpoints", 
                                "{}".format(self.seed),
                                "model_{}".format(rank))
        
        # check if directory exists and create if not
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        # save model
        torch.save(model.state_dict(), 
                   os.path.join(root_dir, 'ep_{}.tar'.format(episode_idx)))

    def get_cleaned_checkpoints(self):
        for rank in range(self.num_agents):
            checkpoint_paths = {}
            paths = glob.glob(os.path.join(self.storage_loc, "checkpoints", 
                                           "{}".format(self.seed), 
                                           "model_{}".format(rank), 
                                           "*.tar"))
            paths_dic = {int(path.split('.')[0].split("_")[-1]): path for path in paths}
            last_ep = max(paths_dic.keys())
            
            checkpoint_idxs = np.linspace(1, last_ep + 1, self.checkpoints, endpoint=True, dtype=np.int) - 1

            for idx, path in paths_dic.items():
                if idx in checkpoint_idxs:
                    checkpoint_paths[idx] = path
                else:
                    os.unlink(path)    
    
    
                    
                    






