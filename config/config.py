# Configuration class
class CONFIG:
    def __init__(self):
        # initialize parameters
        self.filename = "Tennis.exe"
        self.seed = 12

        # information for policy model
        self.policy_info = {"hidden_dims":[256, 256],
                            "learning_rate":0.0005,
                            "max_grad_norm":float("inf")}

        # information for value model
        self.value_info = {"hidden_dims":[256, 256],
                           "learning_rate":0.0005,
                           "max_grad_norm":float("inf")}

        # general training information
        self.training_info = {"update_every_steps": 1,
                              "n_warmup_batches": 2,
                              "weight_mix_ratio": 0.005}

        # environment information
        self.env_info = {"gamma": 0.95,
                         "max_minutes": 300,
                         "max_episodes": 2000,
                         "goal_mean_100_rewards": 0.5}
        # buffer_information
        self.buffer_info = {"capacity":100000,
                           "batch_size": 1024}

        # number of training checkpoints to be saved
        self.checkpoints = 4

        # results storage folder location
        self.result_storage = "results"

        # number of optimized iterations
        self.optim_iter = 1  

       

