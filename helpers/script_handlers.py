from helpers.env_agent_interact import Env_Agent_Mix
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import glob
import sys
import os

def train_script(config_params):
    comb_results = []
    best_agent, best_eval_score = None, float("-inf")

    # initialize environment and parameters
    agent = Env_Agent_Mix(filename=config_params.filename,
                          training_info=config_params.training_info, 
                          policy_info=config_params.policy_info, 
                          value_info=config_params.value_info, 
                          buffer_info=config_params.buffer_info,
                          seed=config_params.seed, optim_update=config_params.optim_iter,
                          checkpoints=config_params.checkpoints,
                          storage_loc=config_params.result_storage)
    # train for all seeds 
    results, final_eval_score, agent_scores, training_time, \
    wallclock_time, fin_episode = agent.train(config_params.env_info)
    
    # add to list
    comb_results = results[:fin_episode].T
    agent_scores = agent_scores.T
    # plot preparations 
    ts = comb_results[0]
    mean_r, min_r, max_r = comb_results[1], comb_results[2], comb_results[3]  
    mean_s, min_s, max_s =  comb_results[4], comb_results[5], comb_results[6]
    mean_best_score = final_eval_score
    agent_0 = agent_scores[0]
    agent_1 = agent_scores[1] 
    x_axis = np.arange(len(mean_s))
    
    # plot 
    fig, axs = plt.subplots(2,1, figsize=(15, 10), sharey=False, sharex=True)
    # training
    axs[0].plot(max_r, 'g', linewidth=1)
    axs[0].plot(min_r, 'g', linewidth=1)
    axs[0].plot(mean_r, 'k:', linewidth=2)
    axs[0].fill_between(x_axis, min_r, max_r, facecolor="g", alpha=0.3)
    # evaluation
    axs[1].plot(max_s, 'g', linewidth=1)
    axs[1].plot(min_s, 'g', linewidth=1)
    axs[1].plot(mean_s, 'k:', linewidth=2)
    axs[1].fill_between(x_axis, min_s, max_s, facecolor="g", alpha=0.3)
    
    axs[0].set_title("Moving Avg Reward [Training] per model")
    axs[1].set_title("Moving Avg Reward [Evaluation] per model")
    plt.xlabel("Episodes")
    plt.show()

    x_axis = np.arange(len(mean_best_score))
    fig, axs = plt.subplots(2,1, figsize=(15, 10), sharey=True, sharex=True)
    # evaluation of best model
    axs[0].plot(agent_0, 'k--', label="Agent_0", linewidth=1)
    axs[0].plot(agent_1, 'b', label="Agent_1", linewidth=1)
    axs[0].fill_between(x_axis, agent_0, agent_1, facecolor="g", alpha=0.3)

    axs[1].plot(mean_best_score, 'k:', linewidth=2)
    
    axs[0].set_title("Avg Reward per agent [Evaluation] best model")
    axs[1].set_title("Avg Reward mean [Evaluation] best model")
    plt.xlabel("Episodes")
    plt.show()

def eval_script(config_params, chckpt=None):
    result_loc = os.path.join(config_params.result_storage, "checkpoints",
                              str(config_params.seed))
    # initialize environment and parameters
    agent_m = Env_Agent_Mix(filename=config_params.filename,
                          training_info=config_params.training_info, 
                          policy_info=config_params.policy_info, 
                          value_info=config_params.value_info, 
                          buffer_info=config_params.buffer_info,
                          seed=config_params.seed, optim_update=config_params.optim_iter,
                          checkpoints=config_params.checkpoints,
                          storage_loc=config_params.result_storage)
    checkpoint_loc = []

    # load the agents
    for rank in range(agent_m.num_agents):
      paths = glob.glob(os.path.join(result_loc, "model_{}".format(rank), "*.tar"))
      paths_dic = {int(path.split('.')[0].split("_")[-1]): path for path in paths}
      last_ep = max(paths_dic.keys())
      
      # if available checkpoint is given load that
      if chckpt is not None and chckpt in paths_dic.keys():
        checkpoint_loc.append(paths_dic[chckpt])        
      # else load the final
      else:
        checkpoint_loc.append(paths_dic[last_ep])

    # load checkpoints
    agent_m.multi_agents.load_models(checkpoint_loc)
    
    # run simulation
    states = agent_m.env_class.reset(agent_m.env, agent_m.brain_name, train_mode=False, flag=False)
    score = np.zeros(agent_m.num_agents)

    start_time = time.time()
    while True:
      actions = []
      for (curr_agent, state) in zip(agent_m.multi_agents.agents, states):
        # append to 
        actions.append(curr_agent.policy_model.select_greedy_action(state))

      # take step and obtain rewards
      parameters = agent_m.env_class.step(agent_m.env, agent_m.brain_name, np.array(actions))
      score += np.array(parameters["rewards"])
      states = parameters["next_states"]
      dones = parameters["dones"]

      if True in dones:
        break

    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print("Terminal state reached. Scores for agents are:{}. Episode runtime: {} ".format(list(score), elapsed_time))
      

