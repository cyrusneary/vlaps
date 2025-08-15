import wandb
from vlaps.environments.libero_utils import build_libero_env_and_task_suite
import datetime
import logging
from vlaps.environments.libero_utils import LIBERO_DUMMY_ACTION, write_video
from vlaps.agents.vla_agent import VLAAgent

import pathlib
import time
import json
import numpy as np
import pickle

import copy

from PIL import Image

import time
import datetime

class LiberoMultiEpisodeRunner():

    def __init__(
        self,
        config,
        seed = 42,
        run_name = None
    ):
        self.cfg = config

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.results = {
            "num_runs" : 0,
            "num_successes" : 0,
            "run_times" : {},
            "run_successes" : [],
            "run_failures" : [],
        }
        self.task_list = self.cfg.runner_config.tasks
        self.task_name_to_id_list = {}
        for task_name in self.task_list:
            self.task_name_to_id_list[task_name] = self.cfg.runner_config[task_name]

        self.init_states_ids = self.cfg.runner_config.init_states_ids

        if run_name is None:
            now = datetime.datetime.now()
            self.run_name = now.strftime(
                "%Y-%m-%d_%H-%M-%S_" + self.cfg.experiment_name
            )
        else:
            self.run_name = run_name

        self.results_save_path = pathlib.Path(self.cfg.results_out_path) / self.run_name
        self.results_save_path.mkdir(parents=True, exist_ok=True)

        self._construct_vla()

    def _construct_vla(self):
        """
        Construct the VLA to use for the experiment.
        """

        assert "vla_type" in self.cfg.agent_config, "Must specify a VLA type in the agent config."
        assert "vla_config" in self.cfg.agent_config, "Must specify a VLA config dict in the agent config."

        if self.cfg.agent_config.vla_type == "octo":
            from vlaps.agents.octo_agent import OctoVLA
            self.vla = OctoVLA(**self.cfg.agent_config.vla_config)
            

    def run(self):
        
        for task_name in self.task_list:
            for task_id in self.task_name_to_id_list[task_name]:
                for init_state_id in self.init_states_ids:
                    seed = self.rng.integers(1e9)

                    logging.info(f"Running task {task_name} with ID {task_id} from initial state {init_state_id} and seed {seed}...")
                    
                    episode_runner = LiberoSingleEpisodeRunner(
                        self.cfg,
                        task_name,
                        task_id,
                        vla=self.vla,
                        init_state_id=init_state_id,
                        seed=seed,
                        run_name=self.run_name,
                    )
                    
                    run_start_time = time.time()
                    episode_success = episode_runner.run()
                    run_end_time = time.time()
                    run_time = run_end_time - run_start_time

                    self.results['run_times'][task_name + "_" + str(task_id) + "_init_state_" + str(init_state_id)] = run_time

                    self.results['num_runs'] += 1
                    if episode_success:
                        self.results['num_successes'] += 1
                        self.results['run_successes'].append(f"{task_name}_{task_id}_init_state_{init_state_id}")
                    else:
                        self.results['run_failures'].append(f"{task_name}_{task_id}_init_state_{init_state_id}")

                    curr_success_rate = self.results['num_successes'] / self.results['num_runs']

                    logging.info(f"Run complete: task {task_name} with ID {task_id} and seed {seed}... Result: {episode_success} after {run_time} seconds.")
                    logging.info(f"Cumulative success rate: {curr_success_rate}. Successful runs: {self.results['num_successes']}. Total runs: {self.results['num_runs']}")

                    # Save results dictionary at end of each run as a json file
                    with open(self.results_save_path / f"multi_run_statistics.json", "w") as f:
                        json.dump(self.results, f, indent=4)

class LiberoSingleEpisodeRunner():

    def __init__(
        self,
        config,
        task_suite_name,
        task_id,
        vla,
        init_state_id = 0,
        seed = 42,
        run_name = None,
    ):
        self.cfg = config
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.seed = seed
        self.init_state_id = init_state_id
        self.vla = vla

        if "max_wall_time" in self.cfg.runner_config:
            self.max_wall_time = self.cfg.runner_config['max_wall_time']
        else:
            self.max_wall_time = -1

        self.results = {
            'task_success' : {},
            'num_vla_calls' : {},
            'num_cache_accesses' : {},
            'get_action_times' : {},
            'vla_call_times' : {},
        }

        if run_name is None:
            now = datetime.datetime.now()
            self.run_name = now.strftime(
                "%Y-%m-%d_%H-%M-%S_" + self.cfg.experiment_name
            )
        else:
            self.run_name = run_name

    def instantiate_env_and_agent(self):
        # Create the environment
        logging.info(f"Creating environment for the f{self.task_suite_name} task with id {self.task_id}...")
        self.env, self.task_description, self.task, self.task_suite, self.num_tasks_in_suite, _ = \
            build_libero_env_and_task_suite(
                self.task_suite_name,
                self.task_id, 
                self.cfg.env_config.resolution, 
                self.seed,
            )
        logging.info(f"Creating environment copy for the f{self.task_suite_name} task with id {self.task_id}...")
        env_copy, _, _, _, _, _ = \
            build_libero_env_and_task_suite(
                self.task_suite_name,
                self.task_id, 
                self.cfg.env_config.resolution, 
                self.seed,
            )
        self.results['task_description'] = self.task_description
        
        self.env.reset()
        env_copy.reset()

        logging.info(f"Instantiating agent of type f{self.cfg.agent_config.agent_type}...")
        if self.cfg.agent_config.agent_type == "mcts_agent":
            from VLAPS.agents.vlaps_agent import VLAPSAgent
            self.agent = VLAPSAgent(
                env_copy,
                self.vla,
                self.task_description,
                **self.cfg.agent_config.mcts_agent_args,
            )
        elif self.cfg.agent_config.agent_type == "vla_agent":
            self.agent = VLAAgent(
                vla=self.vla,
                task_description=self.task_description,
            )
        else:
            raise RuntimeError(f"Agent type {self.cfg.agent_config.agent_type} not recognized")

    def run(self):
        
        # Create the environment
        self.instantiate_env_and_agent()

        init_states = self.task_suite.get_task_init_states(self.task_id)
        obs = self.env.set_init_state(init_states[self.init_state_id])

        # Setup
        t = 0
        task_success = False
        replay_images = []
        replay_wrist_images = []

        wall_time = time.time()

        logging.info(f"Starting episode {self.task_suite_name}, ID: {self.task_id}...")
        while t < self.cfg.env_config.max_steps + self.cfg.env_config.num_steps_wait:
            try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < self.cfg.env_config.num_steps_wait:
                    obs, reward, done, info = self.env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Get preprocessed image
                img, wrist_img = get_imgs_from_obs(obs, self.cfg.agent_config.vla_config.resize_size)

                # Save preprocessed image for replay video
                replay_images.append(img)
                replay_wrist_images.append(wrist_img)

                state = self.env.get_sim_state()

                action_start_time = time.time()
                if self.cfg.agent_config.agent_type == 'mcts_agent':
                    from experiments.libero.vlaps_agent import get_current_controller_state
                    ctrl = get_current_controller_state(self.env)
                    action = self.agent.get_action(obs, state, ctrl)
                else:
                    action = self.agent.get_action(obs, state)
                action_end_time = time.time()
                action_time = action_end_time - action_start_time

                # Execute action in environment
                obs, reward, done, info = self.env.step(action.tolist())
                if done:
                    task_success = True

                    self.results['task_success'][t] = task_success
                    self.results['num_vla_calls'][t] = self.agent.vla.num_vla_calls
                    self.results['num_cache_accesses'][t] = self.agent.vla.num_cache_accesses
                    self.results['get_action_times'][t] = action_time

                    break
                t += 1

                if not done and (not self.max_wall_time == -1) and (time.time() - wall_time) > self.max_wall_time:
                    self.results['task_success'][t] = False
                    self.results['num_vla_calls'][t] = self.agent.vla.num_vla_calls
                    self.results['num_cache_accesses'][t] = self.agent.vla.num_cache_accesses
                    self.results['get_action_times'][t] = action_time

                    break

                self.results['task_success'][t] = task_success
                self.results['num_vla_calls'][t] = self.agent.vla.num_vla_calls
                self.results['num_cache_accesses'][t] = self.agent.vla.num_cache_accesses
                self.results['get_action_times'][t] = action_time

            except Exception as e:
                logging.error(f"Caught exception: {e}")
                break

        # Save results
        this_episode_folder_name = self.task_suite_name + '_' + str(self.task_id) + '_init_state_' + str(self.init_state_id)

        # Add the current time as a unique identifier in case doing multiple tries of the same task and task_id
        now = datetime.datetime.now()
        curr_time = now.strftime("_%H-%M-%S")
        this_episode_folder_name = this_episode_folder_name + curr_time

        if task_success:
            this_episode_folder_name = this_episode_folder_name + '_success'
        else:
            this_episode_folder_name = this_episode_folder_name + '_failure'

        results_save_path = pathlib.Path(self.cfg.results_out_path) / self.run_name / this_episode_folder_name
        results_save_path.mkdir(parents=True, exist_ok=True)

        # Save as a json file
        with open(results_save_path / f"results.json", "w") as f:
            json.dump(self.results, f, indent=4)

        # if self.cfg.agent_config.agent_type == "mcts_agent":
        #     with open(results_save_path / f"root_nodes.pkl", "wb") as f:
        #         pickle.dump(self.agent.tree_roots, f)

        # # If the agent is a beam search, save the search tree history
        # if self.cfg.agent_config.agent_type == "beam_search_agent":
        #     with open(results_save_path / f"beam_search_leaf_nodes.pkl", "wb") as f:
        #         pickle.dump(self.agent.beam_search_leaf_nodes, f)

        with open(results_save_path / f"all_sampled_chunks.pkl", "wb") as f:
            pickle.dump(self.agent.vla.all_sampled_chunks, f)
        # with open(results_save_path / f"all_sampled_logprobs.pkl", "wb") as f:
        #     pickle.dump(self.agent.all_sampled_logprobs, f)
        # with open(results_save_path / f"all_sampled_tokens.pkl", "wb") as f:
        #     pickle.dump(self.agent.all_sampled_tokens, f)
        # # with open(results_save_path / f"all_sampled_logits.pkl", "wb") as f:
        #     pickle.dump(self.agent.all_sampled_logits, f)

        # Save videos.
        if self.cfg.save_run_video or self.cfg.save_run_wrist_video:
            video_save_path = pathlib.Path(self.cfg.results_out_path) / self.run_name / this_episode_folder_name / "videos"
            video_save_path.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_run_video:
            write_video(
                replay_images,
                video_save_path,
                f"fixed_cam.mp4",
            )
        if self.cfg.save_run_wrist_video:
            write_video(
                replay_wrist_images,
                video_save_path,
                f"wrist_cam.mp4",
            )

        return task_success

    def log_step_results(self):
        self.results

