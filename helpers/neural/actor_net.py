# Modified from https://github.com/mimoralea/gdrl
# import necessary pacakges
from torch.distributions import Normal
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

# initializes the weight and bias data
def layer_init(layer, w_scale:float =1.0, gain_func="relu") -> nn.Linear:
    nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain(gain_func))
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def activation_layers(activation:str):
    if activation.lower() == "relu":
        return F.relu
    if activation.lower() == "tanh":
        return F.tanh

def _format(inp_array:np.ndarray, device:str) -> torch.Tensor:
    x = inp_array
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device,
                        dtype=torch.float32)
        x = x.unsqueeze(0)
    return x

# Fully connected gaussian policy
class FCGP(nn.Module):
    def __init__(self, input_dim:int, action_bounds:tuple, hidden_dims:tuple=(32,32), 
                 log_std_min:int=-20, log_std_max:int=2, entropy_lr:float=0.001,
                 activation:str="relu", activation_out:str="tanh", seed:int=0):
        # initialize super class
        super(FCGP, self).__init__()
        # activation layer
        self.activation_fc = activation_layers(activation)
        self.activation_out = activation_layers(activation_out)
        
        # action bounds
        self.env_min, self.env_max = action_bounds
        
        # log standard
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max 
        
        # input layer
        self.input_layer = layer_init(nn.Linear(input_dim, hidden_dims[0]))
        
        # hidden layer build
        self.hidden_layers = nn.ModuleList()
        for idx in range(len(hidden_dims) - 1):
            hidden_layer = layer_init(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            self.hidden_layers.append(hidden_layer)
        
        # output layer
        self.output_layer_mean = layer_init(nn.Linear(hidden_dims[-1], len(self.env_max)),
                                             gain_func=activation_out)
        self.output_layer_log_std = layer_init(nn.Linear(hidden_dims[-1], len(self.env_max)),
                                                 gain_func=activation_out)

        # device 
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        # setup the limit
        self.limits_setup()
        
        # entropy 
        self.entropy_setup(entropy_lr)

        # seed inititalization
        self.rand_generator = np.random.RandomState(seed)
        torch.manual_seed(seed)


    def limits_setup(self):
        self.env_min = torch.tensor(self.env_min,
                                    device=self.device,
                                    dtype=torch.float32)
        self.env_max = torch.tensor(self.env_max,
                                    device=self.device,
                                    dtype=torch.float32)
        self.nn_min = self.activation_out(torch.Tensor([float("-inf")])).to(self.device)
        self.nn_max = self.activation_out(torch.Tensor([float("inf")])).to(self.device)
        self.rescale_fn = lambda x: self.env_min + ((x - self.nn_min) * (self.env_max - self.env_min)/\
                                                    (self.nn_max - self.nn_min))

    def entropy_setup(self, entropy_lr):
        self.target_entropy = -np.prod((self.env_max.shape,)).item()
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.logalpha], lr=entropy_lr)

    

    def forward(self, state:np.ndarray):
        x = _format(state, self.device)
        
        # input layer
        x = self.activation_fc(self.input_layer(x))

        # hidden layer loop
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        # mean and log standard deviation
        x_mean = self.output_layer_mean(x)

        x_log_std = self.output_layer_log_std(x)
        x_log_std = torch.clamp(x_log_std,
                                self.log_std_min,
                                self.log_std_max)
        return x_mean, x_log_std
    
    def full_pass(self, state, epsilon=1e-6):
        # get the mean and log standard deviation
        mean, log_std = self.forward(state)

        # normal distribution
        pi_s = Normal(mean, log_std.exp())

        # reparameterization trick
        pre_tanh_action = pi_s.rsample()
        tanh_action = self.activation_out(pre_tanh_action)
        action = self.rescale_fn(tanh_action)

        # log probabilities
        log_prob = pi_s.log_prob(pre_tanh_action) - torch.log((1 - tanh_action.pow(2)).clamp(0,1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def _update_exploration_ratio(self, greedy_action:np.ndarray, action_taken:np.ndarray):
        # environment bounds
        env_min, env_max = self.env_min.cpu().numpy(), self.env_max.cpu().numpy()
        self.exploration_ratio = np.mean(abs((greedy_action - action_taken) / (env_max - env_min)))

    def _get_actions(self, state):
        # forward pass
        mean, log_std = self.forward(state)

        # sample action from distribution
        action = torch.distributions.Normal(mean, log_std.exp()).sample()
        action = self.rescale_fn(self.activation_out(action))

        # greedy actions
        greedy_action = self.rescale_fn(self.activation_out(mean))
        random_action = self.rand_generator.uniform(low=self.env_min.cpu().numpy(),
                                                    high=self.env_max.cpu().numpy())
        action_shape = self.env_max.cpu().numpy().shape

        # reshape
        action = action.detach().cpu().numpy().reshape(action_shape)
        greedy_action = greedy_action.detach().cpu().numpy().reshape(action_shape)
        random_action = random_action.reshape(action_shape)

        return action, greedy_action, random_action

    def select_random_action(self, state):
        _, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, random_action)
        return random_action

    def select_greedy_action(self, state):
        _, greedy_action, _ = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, greedy_action)
        return greedy_action

    def select_action(self, state):
        action, greedy_action, _ = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, action)
        return action

    def load_checkpoint(self, chckpoint):
        self.load_state_dict(torch.load(chckpoint))