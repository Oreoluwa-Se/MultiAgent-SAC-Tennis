# Modified from https://github.com/mimoralea/gdrl
# import necessary pacakges
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

# fully connected q values from state action pair
class FCQSA(nn.Module):
    def __init__(self, input_dim:int, output_dim:int,
                 hidden_dims:tuple=(32,32), activation="relu",
                 seed:int=0):
        # initialize super class
        super(FCQSA, self).__init__()
        # activation layer
        self.activation_fc = activation_layers(activation)
        
        # input layer
        self.input_layer = layer_init(nn.Linear(input_dim + output_dim, hidden_dims[0]))
        
        # hidden layer build
        self.hidden_layers = nn.ModuleList()
        for idx in range(len(hidden_dims) - 1):
            hidden_layer = layer_init(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            self.hidden_layers.append(hidden_layer)
        
        # output layer
        self.output_layer = layer_init(nn.Linear(hidden_dims[-1], 1))

        # device 
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        # seed inititalization
        self.rand_generator = np.random.RandomState(seed)
        torch.manual_seed(seed)

    def forward(self, state, action):
        x, u = _format(state, self.device), _format(action, self.device)

        # input layers
        x = self.activation_fc(self.input_layer(torch.cat((x, u), 
                               dim=1)))

        # pass through hidden layers
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        return self.output_layer(x)

# fully connected q values from state action pair
class FCSV(nn.Module):
    def __init__(self, input_dim:int, hidden_dims:tuple=(32,32), 
                 activation="relu", seed:int=0):
        # initialize super class
        super(FCSV, self).__init__()

        # activation layer
        self.activation_fc = activation_layers(activation)
        
        # input layer
        self.input_layer = layer_init(nn.Linear(input_dim, hidden_dims[0]))
        
        # hidden layer build
        self.hidden_layers = nn.ModuleList()
        for idx in range(len(hidden_dims) - 1):
            hidden_layer = layer_init(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            self.hidden_layers.append(hidden_layer)
        
        # output layer
        self.output_layer = layer_init(nn.Linear(hidden_dims[-1], 1))

        # device 
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        # seed inititalization
        self.rand_generator = np.random.RandomState(seed)
        torch.manual_seed(seed)

    def forward(self, state):
        x= _format(state, self.device)

        # input layers
        x = self.activation_fc(self.input_layer(x))
        
        # pass through hidden layers
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        return self.output_layer(x)