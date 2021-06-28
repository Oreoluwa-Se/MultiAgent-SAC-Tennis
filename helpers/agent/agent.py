from helpers.neural.critic_net import FCQSA, FCSV
from helpers.neural.actor_net import FCGP
import torch.optim as optim
import torch

# contains all actor and critic parameters
class Agent:
    def __init__(self, general_info, policy_info, value_info, seed):
        # setup the actor
        self.actor_setup(general_info, policy_info, seed)

        # setup the critic
        self.critic_setup(general_info, value_info, seed)

    def actor_setup(self, general_info:dict={}, policy_info:dict={}, seed:int=0):
        # single policy model
        self.policy_model  = FCGP(general_info["state_size"], general_info["bounds"], 
                                  hidden_dims=policy_info["hidden_dims"],
                                  seed=seed)
        # policy optimizer
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(),
                                           lr=policy_info["learning_rate"])
        self.policy_max_grad_norm = policy_info["max_grad_norm"]

    def load_checkpoint(self, chckpoint):
        self.policy_model.load_checkpoint(chckpoint)

    def critic_setup(self, general_info:dict={}, value_info:dict={}, seed:int=0):
        total_state = general_info["state_size"] 
        total_actions = general_info["num_actions"] 

        # online and target q_value state action predictor
        self.online_value_model_a = FCQSA(total_state, total_actions, value_info["hidden_dims"], seed=seed)
        self.target_value_model_a = FCQSA(total_state, total_actions, value_info["hidden_dims"], seed=seed)

        self.online_value_model_b = FCQSA(total_state, total_actions, value_info["hidden_dims"], seed=seed)
        self.target_value_model_b = FCQSA(total_state, total_actions, value_info["hidden_dims"], seed=seed)

        # state value predictor
        self.online_state_model_v = FCSV(general_info["state_size"], value_info["hidden_dims"], seed=seed)
        self.target_state_model_v = FCSV(general_info["state_size"], value_info["hidden_dims"], seed=seed)

        # value optimizers
        self.value_optimizer_a = optim.Adam(self.online_value_model_a.parameters(),
                                            lr=value_info["learning_rate"])
        self.value_optimizer_b = optim.Adam(self.online_value_model_b.parameters(),
                                            lr=value_info["learning_rate"])
        self.actor_state_opt = optim.Adam(self.online_state_model_v.parameters(),
                                          lr=value_info["learning_rate"])

        self.value_max_grad_norm = value_info["max_grad_norm"]

        # ensure same origin for value networks
        self.update_value_networks(tau=1)

        # soft weight for updating the network mixture
        self.soft_weight_tau = general_info["weight_mix_ratio"]

    def update_value_networks(self, tau=None):
        tau = self.soft_weight_tau if tau is None else tau
        # copy value model a
        for target, online in zip(self.target_value_model_a.parameters(), 
                                  self.online_value_model_a.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        # copy value model b
        for target, online in zip(self.target_value_model_b.parameters(), 
                                  self.online_value_model_b.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        # copy state value weights
        for target, online in zip(self.target_state_model_v.parameters(), 
                                  self.online_state_model_v.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def forward_pass(self, critic_dict:dict, agent_dict:dict, curr_actions_merged:torch.Tensor, pred_nxt_action_merged:torch.Tensor):
        # qvalue based on current states and current predicted actions by the policy
        current_q_sa_a = self.online_value_model_a(agent_dict["states"], curr_actions_merged)
        current_q_sa_b = self.online_value_model_b(agent_dict["states"], curr_actions_merged)
        # take minimum
        q_sa = torch.min(current_q_sa_a, current_q_sa_b)

        # q value based on next states and predicted actions by the policy
        q_spap_a = self.target_value_model_a(agent_dict["next_states"], pred_nxt_action_merged)
        q_spap_b = self.target_value_model_b(agent_dict["next_states"], pred_nxt_action_merged)
        # take minimum 
        q_spap = torch.min(q_spap_a, q_spap_b)

        # q value based on current state and current actions 
        q_sa_a = self.online_value_model_a(agent_dict["states"], agent_dict["actions"])
        q_sa_b = self.online_value_model_b(agent_dict["states"], agent_dict["actions"])

        # state_value prediction for current state and next state
        v_pred = self.online_state_model_v(agent_dict["states"])
        v_targ = self.target_state_model_v(agent_dict["next_states"])

        return q_sa, q_spap, q_sa_a, q_sa_b, v_pred, v_targ

    def alpha_opt(self, state, ISweights):
        ## OPTIMIZE ALPHA: 
        # pass states through model to generate current actions
        curr_actions, logpi_s = self.policy_model.full_pass(state)

        # calculate current target alpha
        target_alpha = (logpi_s + self.policy_model.target_entropy).detach()

        # use current target alpha to calculate loss
        alpha_loss = -(self.policy_model.logalpha * target_alpha * ISweights).mean()

        # optimization step
        self.policy_model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.policy_model.alpha_optimizer.step()
        
        # get the alpha value after optimization
        alpha = self.policy_model.logalpha.exp()

        return alpha, curr_actions, logpi_s
