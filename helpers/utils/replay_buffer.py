"""
    Builds the sumtree and memory classes which are used to keep track of experiences
    Reference:  https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/
"""
# import necessary packages
import numpy as np
from collections import namedtuple
import math
import sys

# build sumtree class
class SumTree(object):
    # our friendly looper
    data_pointer = 0

    # tree initialization
    def __init__(self, capacity:int):
        """
            Args:
                Capcity: Number of leaf nodes that contains experiences
            Steps:
                Generate tree with all nodes = 0
                Tree height = 2*capacity - 1
        """
        self.capacity = capacity
        # using a binary tree. subtract 1 for root node
        self.tree = np.zeros(2*capacity - 1)
        # array for storing incoming data. Here we store objects
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority:float, data:namedtuple):
        """
            Stores data in tree
            Args:
                priority: (float) current priority of data added
                data: (tuple) [state, action, reward, next_state, done]
        """
        # index to put the experience in
        tree_index = self.data_pointer + self.capacity - 1
        # update the data 
        self.data[self.data_pointer] = data        
        # update the leaf
        self.update(tree_index, priority)
        # increment data pointer
        self.data_pointer += 1
        # cicrular array
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index:int, priority:float):
        """
            Updates the current treee priorities
            Args:
                tree_indx: (int) location whose priority is to be changed
                priroty: (float) current priority based on absolute error from neural network
        """
        # calculate priority difference
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # propagate the change through the tree
        while tree_index != 0:
            # calculate parent tree
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value:float):
        """
            get the leaf node 
            Args:
                value: (float) the sampled priority we are searching for
        """
        parent_idx, leaf_idx = 0, 0
        # search begins
        while True:
            # calculate left child and right child
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            # if we have reached the bottom end search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # binary search 
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        # extract the data
        data_idx = leaf_idx - self.capacity + 1
        # return
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        # returns trhe root node
        return self.tree[0]

    @property
    def get_size(self):
        # return  current data size
        if self.data_pointer < self.capacity:
            return self.data_pointer
        else:
            return self.capacity

# Memory: uses the sum tree to store experiences
class Memory():
    """
    HYPERPARAMETERS:
        epsilon: (float) The small tolerance we use to prevent having probability of 0
        alpha  : (float) Tradeoff between random sampling and high priority sampling
        beta   : (float) Importance sampling parameter increasing towards 1
        beta_increment : (float) Beta increment per sampling
        err_clip: (float) Upper limit for error clipping
    """
    epsilon, alpha, beta, beta_inc, err_clip = 0.01, 0.6, 0.4, 0.001, 1.0

    def __init__(self, capacity:int, batch_size:int, seed:int):
        # initialize sumtree, tuple for experiences, and random generator seed
        self.tree = SumTree(capacity)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["states", "actions", 
                                      "rewards", "next_states", "dones"])
        self.rand_generator = np.random.RandomState(seed)
        self.current_storage_size = 0
        
    def store(self, state:list, action:list, reward:float, next_state:list, done:bool):
        """
            Stores new experience in the sumtree.
            Args:
                content to store in sum tree (state, action, reward, nextstate, done)
        """
        # convert 
        data_tup = self.experiences(state, action, reward, next_state, done)
        # find the max priority 
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        # if current maximum priority is 0, set to maximum allowed
        if max_priority == 0:
            max_priority = self.err_clip
        # add it to the tree
        self.tree.add(max_priority, data_tup)        
        self.current_storage_size = self.storage_length()

    def storage_length(self) -> int:
        return self.tree.get_size
    
    def sample(self):
        curr_size = self.storage_length()
        # if we have enough for random sampling
        if self.current_storage_size > self.batch_size and  curr_size <= self.tree.capacity:
            return self.sample_random()

        # if we have enough for priotized sampling
        elif self.current_storage_size > self.tree.capacity:
            return self.sample_per()

    def sample_per(self) -> tuple:
        """
            Samples batch_size buffer from the stored experiences
            Args:
                batch_size: Batch size for sampling

            Steps:
                - Sample a minibatch of batch_size size, the range [0, priority_total] is divided into n ranges
                - A value is uniformly sampled from each range
                - Search sumtree for the matching experience
                - Calculate the importance sampling weight
        """
        batch_size = self.batch_size

        # storage array
        memory_batch = []

        # initialize index, and weight tracker
        idx_batch, ISweight_batch = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, 1), dtype=np.float32)
        # prioirty segment
        priority_segment = self.tree.total_priority / batch_size

        # increase beta each time we sample
        self.beta = np.min([1.0, self.beta + self.beta_inc])
        
        # calculate the probability of smallest of priority: used for normalization later on
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * batch_size) ** (-self.beta)
        
        # sampling
        for idx in range(batch_size):
            # uniformly sample value from each range
            a, b = priority_segment * idx, priority_segment * (idx + 1)
            
            value = self.rand_generator.uniform(a, b)

            # get leaf: index of currently sampled data, priority of data, data
            index, priority, experience = self.tree.get_leaf(value)

            # sampling probability
            samp_prob = priority / self.tree.total_priority

            # store tree index
            idx_batch[idx] = index

            # store experience
            memory_batch.append(experience)

            # store the importance weight
            ISweight_batch[idx, 0] = np.power(batch_size * samp_prob, -self.beta) / max_weight
        
        return idx_batch, memory_batch, ISweight_batch

    def sample_random(self) -> tuple:
        idxs = self.rand_generator.choice(self.storage_length(), self.batch_size, replace=False)
        memory_batch = self.tree.data[idxs]

        # return the index and memory batch
        return idxs, memory_batch, np.ones(memory_batch.shape)

    def batch_update(self, tree_indexes:list, abs_errors:list):
        """
            Updates the priority tree
            Args:
                tree_indexes: (array int) indexes to be updates
                abs_errors: (array float) priorty update values
        """
        abs_errors += self.epsilon
        # clip errors
        clipped_errors = np.minimum(abs_errors, self.err_clip)
        # calculate priority
        ps = np.power(clipped_errors, self.alpha)
        # call update priority method
        for tree_index, priority in zip(tree_indexes, ps):
            self.tree.update(tree_index, priority)

    