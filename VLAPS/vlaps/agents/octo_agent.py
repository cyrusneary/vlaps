import collections
import dataclasses
import logging

import time

from functools import partial
import jax

import numpy as np

from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import supply_rng
from octo.libero.libero_utils import normalize_gripper_action, invert_gripper_action


from vlaps.environments.libero_utils import \
    construct_policy_input, \
    build_libero_env_and_task_suite, \
    get_imgs_from_obs, \
    simulate_action_chunk_in_env

from vlaps.environments.libero_hash_map import LiberoHashMap

class OctoVLA:
    def __init__(
        self,
        resize_size : int = 256,
        wrist_resize_size : int = 128,
        generation_temperature : float = 0.0,
        logprob_calc_temp : float = 1.0,
        checkpoint_path : str = "/home/mila/n/nearyc/scratch/octo/octo-libero-90-pretrained",
        checkpoint : int = 60000,
        dataset_statistics : str = "libero_90",
        use_wrist : bool = True,
        use_cache : bool = True,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.resize_size = resize_size
        self.wrist_resize_size = wrist_resize_size
        self.generation_temperature = generation_temperature
        self.logprob_calc_gemp = logprob_calc_temp
        self.checkpoint_path = checkpoint_path
        self.checkpoint = checkpoint
        self.dataset_statistics = dataset_statistics
        self.use_wrist = use_wrist
        self.use_cache = use_cache

        self.num_vla_calls = 0
        self.vla_call_times = []
        self.all_sampled_chunks = []

        # assert 'checkpoint_path' in self.vla_config, "Please specify a checkpoint path from which to load the Octo model"
        # assert "checkpoint" in self.vla_config, "Please specify a checkpoint to load"

        logging.info("Loading finetuned model...")

        model = OctoModel.load_pretrained(self.checkpoint_path, self.checkpoint)

        policy_fn = supply_rng(
            partial(
                model.sample_actions,
                unnormalization_statistics=model.dataset_statistics[self.dataset_statistics]["action"],#dataset_stats["action"],
                # unnormalization_statistics=model.dataset_statistics["action"],#dataset_stats["action"],
            ),
        )

        self.model = model
        self.policy_fn = policy_fn

        if self.use_cache:
            self.hash_map = LiberoHashMap()
            self.state_list = []
            self.num_cache_accesses = 0

    def infer(self, obs, task_description, state):
        """
        Infer an action chunk from the given observation.
        
        Inputs
        ------
        obs : dict
            The observation
        task_description : str
            A string describing the task to complete
        state : np.ndarray
            The environment state (for caching).
        """
        if self.use_cache and self.hash_map.contains(state, task_description):
            action_chunk = self.hash_map.get(state, task_description)['action_chunk']
            self.num_cache_accesses += 1
        else:
            # Construct policy input.
            input = construct_policy_input(
                obs=obs, 
                task_description=task_description, 
                img_resize_size=self.resize_size,
                wrist_img_resize_size=self.wrist_resize_size,
            )
            if self.use_wrist:
                input_dict = {
                    'image_primary': np.expand_dims(input['observation/image'], axis=0),
                    'image_wrist' : np.expand_dims(input['observation/wrist_image'], axis=0),
                    'timestep_pad_mask' : np.array([1.]) # TODO: This is currently a hack to deal with the timestep_pad_mask when observation history is only of length 1.
                }
            else:
                input_dict = {
                    'image_primary': np.expand_dims(input['observation/image'], axis=0),
                    'timestep_pad_mask' : np.array([1.]) # TODO: This is currently a hack to deal with the timestep_pad_mask when observation history is only of length 1.
                }
            input_dict = jax.tree_map(lambda x: x[None], input_dict)

            start_time = time.time()
            task = self.model.create_tasks(texts=[task_description])
            output = self.policy_fn(input_dict, task)

            # Since we are assuming we are not calling batches of inputs, ignore the batch dimension. 
            # Result should be a (T, d) array where T is the action chunk time horizon and d is the action dimension.
            action_chunk = np.array(output[0])
            end_time = time.time()

            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action_chunk = normalize_gripper_action(action_chunk, binarize=True)
            # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            action_chunk = invert_gripper_action(action_chunk)

            self.num_vla_calls += 1
            self.vla_call_times.append(end_time - start_time)

            self.all_sampled_chunks.append(action_chunk)

            if self.use_cache:
                value = {
                    'action_chunk' : action_chunk
                }
                self.hash_map.set(state, task_description, value)
                self.state_list.append(state)

        return action_chunk

        
# class OctoAgent:
#     def __init__(
#             self,  
#             vla,
#             task_description,
#             replan_steps : int = -1,
#         ):
#         """
#         Octo Agent that queries an Octo VLA with observations and a task description to get action chunks, 
#         and follows those action chunks until they are completed.

#         Inputs
#         ------
#         vla : OctoVLA
#             The vla to query
#         task_description : str
#             A description of the current task
#         replan_steps : int
#             The number of steps to execute before replanning. -1 will default to the chunk length of the VLA.
#             Cannot be longer than the chunk length.
#         """
#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.task_description = task_description
#         self.vla = vla

#         self.replan_steps = replan_steps

#         self.action_queue = collections.deque()

#         # self._initialize_vla()

#     # def _initialize_vla(self):
#     #     assert 'checkpoint_path' in self.vla_config, "Please specify a checkpoint path from which to load the Octo model"
#     #     assert "checkpoint" in self.vla_config, "Please specify a checkpoint to load"

#     #     logging.info("Loading finetuned model...")

#     #     model = OctoModel.load_pretrained(self.vla_config.checkpoint_path, self.vla_config.checkpoint)

#     #     policy_fn = supply_rng(
#     #         partial(
#     #             model.sample_actions,
#     #             unnormalization_statistics=model.dataset_statistics[self.vla_config['dataset_statistics']]["action"],#dataset_stats["action"],
#     #             # unnormalization_statistics=model.dataset_statistics["action"],#dataset_stats["action"],
#     #         ),
#     #     )

#     #     self.model = model
#     #     self.policy_fn = policy_fn

#     #     self.set_task(self.task_description)

#     def set_task(self, task_description):
#         """
#         Create the task in the right format to pass into the Octo model.

#         Inputs
#         ------
#         task_description : str
#             A natural language description of the task.
#         """
#         self.task_description = task_description
#         self.task = self.vla.model.create_tasks(texts=[self.task_description])

#     def get_action(self, obs, state):

#         if not self.action_queue:
#             # Finished executing previous action chunk -- compute new chunk

#             action_chunk = self.vla.infer(obs, self.task_description)

#             # # Construct policy input.
#             # input = construct_policy_input(
#             #     obs=obs, 
#             #     task_description=self.task_description, 
#             #     img_resize_size=self.vla_config.resize_size
#             # )
#             # input_dict = {
#             #     'image_primary': np.expand_dims(input['observation/image'], axis=0),
#             #     'timestep_pad_mask' : np.array([1.]) # TODO: This is currently a hack to deal with the timestep_pad_mask when observation history is only of length 1.
#             # }
#             # input_dict = jax.tree_map(lambda x: x[None], input_dict)

#             # start_time = time.time()
#             # output = self.policy_fn(input_dict, self.task)

#             # # Since we are assuming we are not calling batches of inputs, ignore the batch dimension. 
#             # # Result should be a (T, d) array where T is the action chunk time horizon and d is the action dimension.
#             # action_chunk = np.array(output[0])
#             # end_time = time.time()

#             # # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
#             # action_chunk = normalize_gripper_action(action_chunk, binarize=True)
#             # # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
#             # action_chunk = invert_gripper_action(action_chunk)

#             # self.num_vla_calls += 1
#             # self.vla_call_times.append(end_time - start_time)

#             # self.all_sampled_chunks.append(action_chunk)

#             if self.replan_steps != -1:
#                 assert (
#                     len(action_chunk) >= self.replan_steps
#                 ), f"We want to replan every {self.replan_steps} steps, but beam search only predicted {len(action_chunk)} steps."
#                 self.action_queue.extend(action_chunk[: self.replan_steps])
#             else:
#                 self.action_queue.extend(action_chunk)

#         action = self.action_queue.popleft()

#         return action