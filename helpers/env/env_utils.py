from unityagents import UnityEnvironment
import numpy as np

class Init_Env:
	# initialize environment class
	def __init__(self, path:str):
		# stores the path to the environment
		self.path = path

	def make_env_func(self, seed:int):
		# initializes environment and brain name
		env = UnityEnvironment(file_name=self.path, seed=seed)
		brain_name = env.brain_names[0]

		return env, brain_name

	def reset(self, env:UnityEnvironment, brain_name:str, train_mode:bool=True, flag:bool=False):
		"""
			Resets environment either in train or evaluation mode
			Args:
				flag: boolean for if we are initializing 
					  program or taking a step
				brain_name: brain for the environment
				train_mode: either True or False
				env: unity environment			 
				
		"""
		env_info = env.reset(train_mode=train_mode)[brain_name]

		# if in initializing mode
		if flag:
			# initialize parameters
			brain = env.brains[brain_name]
			state_size = env_info.vector_observations.shape[1]
			num_agents = env_info.vector_observations.shape[0]
			action_size = brain.vector_action_space_size

			return state_size, num_agents, action_size

		return env_info.vector_observations

	def step(self, env:UnityEnvironment, brain_name:str, actions:np.ndarray)->dict:
		"""
			returns the information from taking actions per agents
			args:
				actions: (list) actions per agent
		"""
		# concatenate all agent actions
		env_info = env.step(actions)[brain_name]

		"""
			env_info parameters:
				visual_observation,
                vector_observation,
                text_observations,
                memory,
                rewards: list of rewards per agent,
                agents: list of agent id's per agent,
                local_done: list of booleans indicating if agent is in terminal state,
                vector_action: list of stored vector actions per agent, 
                text_action: list of stored text actions per agent,
                max_reached: states if we have reached maximum number of steps - similar to truncated
		"""
		returned_params = {"next_states":env_info.vector_observations,
						   	"rewards":env_info.rewards,
						   	"dones":env_info.local_done,
						   	"max_reached": env_info.max_reached}
		return returned_params


