import collections
import dataclasses
import logging

from openpi_client import websocket_client_policy as _websocket_client_policy

import time
import numpy as np

from vlaps.environments.libero_utils import \
    construct_policy_input, \
    build_libero_env_and_task_suite, \
    get_imgs_from_obs, \
    simulate_action_chunk_in_env,\
    load_action_dictionary
        
class UniformSamplingAgent:
    def __init__(
            self,  
            action_chunk_library_path,
            replan_steps : int = -1,
        ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.action_chunk_library_path = action_chunk_library_path
        self.action_chunk_library = load_action_dictionary(self.action_chunk_library_path)
        self.num_action_chunks = self.action_chunk_library.shape[0]

        self.replan_steps = replan_steps

        self.action_queue = collections.deque()

        self.num_vla_calls = 0
        self.all_sampled_chunks = []

    def get_action(self, obs, state):

        if not self.action_queue:
            
            idx = np.random.randint(self.num_action_chunks)  # Random index from 0 to k-1
            action_chunk = self.action_chunk_library[idx]

            if self.replan_steps != -1:
                assert (
                    len(action_chunk) >= self.replan_steps
                ), f"We want to replan every {self.replan_steps} steps, but beam search only predicted {len(action_chunk)} steps."
                self.action_queue.extend(action_chunk[: self.replan_steps])
            else:
                self.action_queue.extend(action_chunk)

        action = self.action_queue.popleft()

        return action