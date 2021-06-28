from unityagents import UnityEnvironment
import numpy as np

class EnvironmentInit:
	# initialize environment class
	def __init__(self, path):
		# stores the path to environment name and seed
		self.path = path


	def make_env_func(self, seed):
		"""
			initializes the environment and brain name
		"""
		env = UnityEnvironment(file_name=self.path, seed=seed)
		brain_name = env.brain_names[0]

		return env, brain_name

	def reset(self, env, brain_name, train_mode=True, flag=False):
		"""
			resets environment either in train or evaluation mode
			args:
				train_mode: (bool) 
		"""

		env_info = env.reset(train_mode=train_mode)[brain_name]

		if flag:
			# initialize parameters
			brain = env.brains[brain_name]
			state_size = env_info.vector_observations.shape[1]
			num_agents = len(env_info.agents)
			action_size = brain.vector_action_space_size
			"""
				env_info parameters:
					visual_observation,
	                vector_observation,
	                text_observations,
	                memory,
	                reward: list of rewards per agent,
	                agents: list of agent id's per agent,
	                local_done: list of booleans indicating if agent is in terminal state,
	                vector_action: list of stored vector actions per agent, 
	                text_action: list of stored text actions per agent,
	                max_reached: states if we have reached maximum number of steps - similar to truncated
			"""
			return state_size, num_agents, action_size

		return env_info.vector_observations

	def step(self, env, brain_name, actions):
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
						   	"done":env_info.local_done,
						   	"max_reached": env_info.max_reached}
		return returned_params





		
