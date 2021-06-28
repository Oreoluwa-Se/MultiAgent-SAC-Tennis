from helpers.script_handlers import train_script, eval_script
from config.config import CONFIG
import argparse

"""
    ### SCRIPT FOR TRAINING AND EVALUATING THE TENNIS ENVIRONMENT.
    Configuration files can be found in config folder. Current setting are
        # current seed parameter
        seed = 12

        # information for actor [action prediction]
        policy_info = {"hidden_dims":[256, 256],
                       "learning_rate":0.0005,
                       "max_grad_norm":float("inf")}

        # information for critics and state value estimators
        value_info = {"hidden_dims":[256, 256],
                      "learning_rate":0.0005,
                      "max_grad_norm":float("inf")}

        # general training information
        training_info = {"update_every_steps": 1,
                         "n_warmup_batches": 2,
                         "weight_mix_ratio": 0.005}

        # environment information
        env_info = {"gamma": 0.95,
                    "max_minutes": 300,
                    "max_episodes": 2000,
                    "goal_mean_100_rewards": 0.5}
        
        # replay buffer storage
        buffer_info = {"capacity":100000,
                       "batch_size": 1024}

        # number of training checkpoints to be saved 
        checkpoints = 4

        # results storage folder location
        result_storage = "results"

        # number of times we sample the memory buffer per update step
        optim_iter = 1 
"""    

if __name__ == "__main__":
    # configuration parameters
    config_params = CONFIG()

    # train or evaluation
    ap = argparse.ArgumentParser(description="Tennis Environment Trainer")
    ap.add_argument("-m","--mode", type=str, default="train",
                    choices=["train", "evaluate"], help="select if train or evaluation mode")
    args = ap.parse_args()

    if args.mode.lower() == "train":
        train_script(config_params)
    else:
        msg = "Check results\\checkpoints\\... folder for options\n"
        msg += "or insert from [0, 548, 1096, 1644]\n"
        msg += "or insert '-1' for Fully -Trained Agent: "
        
        # train or evaluation
        inp = input(msg)
        if int(inp) == -1:
            eval_script(config_params)
        else:
            eval_script(config_params, int(inp))