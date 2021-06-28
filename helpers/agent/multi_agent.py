from helpers.utils.replay_buffer import Memory
from helpers.agent.agent import Agent
import numpy as np
import torch
import sys

# multi-agent workings 
class MultiAgent:
    def __init__(self, general_info:dict={}, policy_info:dict={},
                 value_info:dict={}, buffer_info:dict={}, seed:int=0):
        # initialize multiple
        self.agents = [Agent(general_info, policy_info, value_info, seed)\
                       for rank in range(general_info["num_agents"])]
        self.num_agents = general_info["num_agents"]

        # initialize replay_buffer
        self.buffer = Memory(buffer_info["capacity"], buffer_info["batch_size"], seed)
        self.batch_size = buffer_info["batch_size"]

        # device
        self.device = self.agents[0].policy_model.device

    def load_models(self, checkpoints):
        for agent, checkpoint in zip(self.agents, checkpoints):
            agent.load_checkpoint(checkpoint)


    def unwrap_experiences(self, experience_list:tuple) -> tuple:
        """
            Converts experience list into individual parameters
            Args:
                experience_list: (list) contains sampled (state, action, reward, 
                                                          nextstate, done)
            Returns Tensors for:
                state, action, reward, nextstate, done
        """
        states  = np.vstack([[e.states] for e in experience_list if e is not None])
        actions = np.vstack([[e.actions] for e in experience_list if e is not None])
        rewards = np.vstack([e.rewards for e in experience_list if e is not None])
        next_states = np.vstack([[e.next_states] for e in experience_list if e is not None])
        dones = np.vstack([e.dones for e in experience_list if e is not None]).astype(np.uint8)
    
        return states, actions, rewards, next_states, dones

    def torch_conv(self, memory_batch:(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray),
                   ISweights:np.ndarray): 
        # unwrap (SARS')
        states, actions, rewards, next_states, dones = self.unwrap_experiences(memory_batch)       
        device = self.device

        # convert importance sampling weights to torch tensor as well
        ISweights = torch.from_numpy(ISweights).float().to(device)
        
        # states, actions, rewards, next_states, dones to torch tensors        
        next_states = torch.from_numpy(next_states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        states = torch.from_numpy(states).float().to(device)              
        dones = torch.from_numpy(dones).float().to(device)     

        # update general information
        return states, actions, rewards, next_states, dones, ISweights    

    def agents_dict_create(self, states:torch.tensor, actions:torch.tensor, rewards:torch.tensor,
                             next_states:torch.tensor, dones:torch.tensor):
        # formatting to meet requirements of the policy network
        self.agents_dict = {rank:{} for rank in range(self.num_agents)}

        for rank in range(len(self.agents)):
            self.agents_dict[rank]["states"] = states[:,rank,...]
            self.agents_dict[rank]["actions"] = actions[:,rank,...]
            self.agents_dict[rank]["next_states"] = next_states[:,rank,...]
            self.agents_dict[rank]["rewards"] = rewards[:,rank]
            self.agents_dict[rank]["dones"] = dones[:,rank]

    def critic_dict_create(self, states:torch.tensor, actions:torch.tensor, next_states:torch.tensor):
        # formatting to meet requirements of the value network
        self.critic_dict = {}        
        self.critic_dict["next_states"] = torch.cat([next_states[:,idx] for idx in range(self.num_agents)], dim=1)        
        self.critic_dict["actions"] = torch.cat([actions[:,idx] for idx in range(self.num_agents)], dim=1)        
        self.critic_dict["states"] = torch.cat([states[:,idx] for idx in range(self.num_agents)], dim=1)       
    

    # extract from memory and create dictionary of sampled parameters
    def load(self, idx_batch:np.ndarray, memory_batch:np.ndarray, ISweights:np.ndarray):
        # get unwrapped torch tensors
        states, actions, rewards, next_states, dones, ISweights = self.torch_conv(memory_batch, ISweights)
        
        # get formated representation of states, actions, rewards, next_states, dones
        self.agents_dict_create(states, actions, rewards, next_states, dones)        
        self.critic_dict_create(states, actions, next_states)        
        self.ISweights = ISweights
        self.idxs = idx_batch

    def update_networks_multi(self):
        for agent in self.agents:
            agent.update_value_networks()

    def optimize(self, idx_batch:np.ndarray, memory_batch:np.ndarray, ISweights:np.ndarray, gamma:float):
        # load memory
        self.load(idx_batch, memory_batch, ISweights)

        # absolute error keep
        abs_error = []
        ## Alpha optimizer and action predictions
        alphas, curr_actions, log_pis = [], [], []
        pred_nxt_actions, log_pis_nxt = [], []
        
        for idx, (agent, params) in enumerate(zip(self.agents, self.agents_dict.values())):
            # optimize alpha and predict actions based on current agent state
            alpha, curr_action, log_pi = agent.alpha_opt(params["states"], self.ISweights)
            alphas.append(alpha); curr_actions.append(curr_action); log_pis.append(log_pi)

            # predict actions based on current agent next_state
            pred_nxt_action, log_pi_nxt = agent.policy_model.full_pass(params["next_states"])
            pred_nxt_actions.append(pred_nxt_action); log_pis_nxt.append(log_pi_nxt)
        
        curr_actions_merged = torch.cat([curr_actions[idx] for idx in range(len(self.agents))], dim=1)
        pred_nxt_action_merged = torch.cat([pred_nxt_actions[idx] for idx in range(len(self.agents))], dim=1)

        # ensure proper dimensions for sampling weights  
        sampling_weights = torch.reshape(self.ISweights, (self.batch_size, 1))

        for idx, agent in enumerate(self.agents):
            shape = (self.batch_size, 1)
            
            # ensure proper matrix dimensions
            actor_log = torch.reshape(log_pis[idx], shape)
            actor_log_nxt = torch.reshape(log_pis_nxt[idx], shape)
            actor_rewards = torch.reshape(self.agents_dict[idx]["rewards"], shape)
            actor_dones = torch.reshape(self.agents_dict[idx]["dones"], shape)

            # run forward pass 
            q_sa, q_spap, q_sa_a, q_sa_b, v_pred, v_targ = agent.forward_pass(self.critic_dict,
                                                                              self.agents_dict[idx], 
                                                                              curr_actions[idx], 
                                                                              pred_nxt_actions[idx])

            ## Policy loss
            # calculate advantage
            advantage = q_sa - v_targ.detach()

            # policy loss equation
            policy_loss = ((alphas[idx] * actor_log * sampling_weights) - advantage).mean()
            
            ## TD target calculation
            q_target = (actor_rewards + gamma * v_targ * (1 - actor_dones)).detach()
                        
            # td_target based on predicted q values from state action pairs    
            q_spap = q_spap - (alphas[idx] * actor_log_nxt * sampling_weights)
            
            # calculate the target_q_sa
            target_q_sa = (actor_rewards + gamma * q_spap * (1 - actor_dones)).detach()
            targ = torch.cat((q_target,target_q_sa), dim=1)
            targ = torch.mean(targ, dim=1).reshape(shape)

            # Q_loss
            qa_diff = (q_sa_a - targ) * sampling_weights
            qb_diff = (q_sa_b - targ) * sampling_weights
            qa_loss = qa_diff.pow(2).mul(0.5).mean() 
            qb_loss = qb_diff.pow(2).mul(0.5).mean()

            # vf_loss
            v_target = q_sa - alphas[idx] * actor_log
            vf_loss = (v_pred - v_target.detach()).pow(2).mul(0.5).mean()

            # policy optimizer
            agent.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.policy_model.parameters(),
                                           agent.policy_max_grad_norm)
            agent.policy_optimizer.step()    
        
            # value_optimizer
            agent.value_optimizer_a.zero_grad()
            qa_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.online_value_model_a.parameters(),
                                           agent.value_max_grad_norm)
            agent.value_optimizer_a.step()

            agent.value_optimizer_b.zero_grad()
            qb_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.online_value_model_b.parameters(),
                                           agent.value_max_grad_norm)
            agent.value_optimizer_b.step()

            # train value function
            agent.actor_state_opt.zero_grad()
            vf_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.online_state_model_v.parameters(),
                                           agent.value_max_grad_norm)
            agent.actor_state_opt.step()
            
            # for batch update
            abs_error.append(torch.abs(torch.max(qa_diff, qb_diff)).cpu().detach().numpy())

        # update priority array
        abs_error = np.array(abs_error).mean(axis=0)
        self.buffer.batch_update(self.idxs, abs_error.squeeze())
        

        



        