
import pathlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vlaps.environments.libero_utils import get_imgs_from_obs, construct_policy_input, _get_libero_env, build_libero_env_and_task_suite, write_video

import pickle

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def visualize_action_dictionaries(action_chunk_dictionary, seed, task_suite_name, task_id, video_out_path=None):
    dictionary_size = len(action_chunk_dictionary)

    if video_out_path is None:
        video_out_path = pathlib.Path(f"experiments/libero/action_chunk_dictionaries/octo/videos/dictionary_{dictionary_size}")

    # Set random seed
    np.random.seed(seed)

    env, task_description, task, task_suite, num_tasks_in_suite, max_steps = \
        build_libero_env_and_task_suite(task_suite_name, task_id)
    
    init_states = task_suite.get_task_init_states(task_id)
    init_state = init_states[0]

    for id, action_chunk in enumerate(tqdm(action_chunk_dictionary)):

        img_list = []
        wrist_img_list = []

        env.env.timestep = 0 # Hack to prevent the env running out of timesteps
        obs = env.set_init_state(init_state)

        total_reward = 0
        for action in action_chunk[:10, :]:
            obs, reward, done, info = env.step(action.tolist())
            total_reward += reward
            img, wrist_img = get_imgs_from_obs(obs, LIBERO_ENV_RESOLUTION)
            img_list.append(img)
            wrist_img_list.append(wrist_img)
        
        file_name = f"chunk_{id}.mp4"

        write_video(img_list, video_out_path, file_name)

def visualize_all_action_dictionaries_in_one_video(action_chunk_dictionary, seed, task_suite_name, task_id, video_out_path=None):
    dictionary_size = len(action_chunk_dictionary)

    if video_out_path is None:
        video_out_path = pathlib.Path(f"experiments/libero/action_chunk_dictionaries/octo/videos/dictionary_{dictionary_size}")

    # Set random seed
    np.random.seed(seed)

    env, task_description, task, task_suite, num_tasks_in_suite, max_steps = \
        build_libero_env_and_task_suite(task_suite_name, task_id)
    
    init_states = task_suite.get_task_init_states(task_id)
    init_state = init_states[0]

    img_list = []
    wrist_img_list = []

    for id, action_chunk in enumerate(tqdm(action_chunk_dictionary)):

        env.env.timestep = 0 # Hack to prevent the env running out of timesteps
        obs = env.set_init_state(init_state)

        total_reward = 0
        for action in action_chunk[:10, :]:
            obs, reward, done, info = env.step(action.tolist())
            total_reward += reward
            img, wrist_img = get_imgs_from_obs(obs, LIBERO_ENV_RESOLUTION)
            img_list.append(img)
            wrist_img_list.append(wrist_img)
        
    file_name = f"all_actions_one_video.mp4"

    write_video(img_list, video_out_path, file_name)

if __name__ == "__main__":
    file_name = "2025-06-20_octo_spatial_goal_objcet_successes_only_2000_medoids.pkl"
    folder_path = pathlib.Path("experiments/libero/action_chunk_dictionaries/octo")
    file_path = folder_path / file_name

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    video_out_path = folder_path / file_name.split(".pkl")[0]

    # visualize_action_dictionaries(data, 42, "libero_spatial", 0, video_out_path)
    visualize_all_action_dictionaries_in_one_video(data, 42, "libero_spatial", 0, video_out_path)