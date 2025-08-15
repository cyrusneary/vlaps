
import pathlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vlaps.environments.libero_utils import get_imgs_from_obs, construct_policy_input, _get_libero_env, build_libero_env_and_task_suite, write_video

from vlaps.action_chunk_dictionaries.action_chunk_distributions import sample_k_chunks_from_library

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

# def sample_k_chunks_from_library(library, vla_chunk, k=5, alpha=1.0, epsilon=0.1, include_vla_chunk=True):
#     """
#     Samples exactly k different chunks from the library, biased toward a VLA-sampled seed chunk.
    
#     Args:
#         library: (N, T, d) array of action chunks.
#         vla_chunk: (T, d) array.
#         k: number of chunks to sample.
#         alpha: scaling factor for softmax sharpness.
#         epsilon: probability of random exploration.
#         include_vla_chunk: whether to include the VLA chunk in the sampled chunks.
        
#     Returns:
#         (k + 1, T, 7) array of k + 1 sampled chunks (including the original input vla_chunk), each of shape (T, 7).
#     """
#     N, T, d = library.shape
#     Tp, dp = vla_chunk.shape
#     assert k <= N, "Cannot sample more unique chunks than library size!"
#     assert d == 7, "Expected action dimension to be 7 (pose+orientation+gripper)"
#     assert d == dp and T == Tp, "Library and VLA chunk dimensions must match"

#     vla_chunk_flat = vla_chunk.flatten()
#     library_flat = library.reshape(N, -1)

#     # Precompute distances once
#     distances = np.linalg.norm(library_flat - vla_chunk_flat, axis=1)
#     probs = np.exp(-alpha * distances)
#     probs /= probs.sum()

#     sampled_indices = set()
#     chunks = []

#     if include_vla_chunk:
#         # Include the VLA chunk if specified
#         chunks.append(vla_chunk)
#         # sampled_indices.add(np.where((library_flat == vla_chunk_flat).all(axis=1))[0][0])

#     while len(chunks) < k:
#         available_indices = list(set(range(N)) - sampled_indices)

#         if len(available_indices) == 0:
#             break  # Safety: no more available chunks (shouldn't happen if k <= N)

#         if np.random.rand() < epsilon:
#             # Random exploration
#             idx = np.random.choice(available_indices)
#         else:
#             # Biased sampling
#             adjusted_probs = probs[available_indices]
#             adjusted_probs /= adjusted_probs.sum()  # Normalize over available only
#             idx = np.random.choice(available_indices, p=adjusted_probs)

#         sampled_indices.add(idx)
#         chunks.append(library[idx])

#     return np.stack(chunks)

def sample_near_query(query, tau=0.1, k=10):
    """
    Sample from a Gaussian centered at the query point.

    Args:
        query: (T, d) ndarray
        tau:   standard deviation (spread of the samples)
        num_samples: number of samples to draw

    Returns:
        (num_samples, d) samples
    """
    T, d = query.shape
    noise = np.random.randn(k, T, d) * tau
    sampled = query[None, :, :] + noise  # broadcast query to (k, T, d)
    return np.concatenate([query[None, :, :], sampled], axis=0)

if __name__ == "__main__":
    file_name = "2025-07-14_octo_all_task_suites_successes_only_2000_medoids.pkl"
    folder_path = pathlib.Path("experiments/libero/action_chunk_dictionaries/octo")
    file_path = folder_path / file_name

    with open(file_path, 'rb') as f:
        chunk_library = pickle.load(f)

    alpha = 10.0
    k = 10
    epsilon = 0.1
    chunk_id = 1000

    sampled_chunks = sample_k_chunks_from_library(
        chunk_library, chunk_library[chunk_id], k=k, alpha=alpha, epsilon=epsilon, include_vla_chunk=True
    )
    video_out_path = folder_path / (file_name.split(".pkl")[0] + f"_chunkid_{chunk_id}_alpha_{alpha}_epsilon_{epsilon}_k_{k}")

    # k = 10
    # tau = 0.5
    # chunk_id = 1000

    # sampled_chunks = sample_near_query(chunk_library[chunk_id], k=k, tau=tau)
    # video_out_path = folder_path / (file_name.split(".pkl")[0] + f"_gaussian_sampling_chunkid_{chunk_id}_k_{k}_tau_{tau}")

    # visualize_action_dictionaries(data, 42, "libero_spatial", 0, video_out_path)
    visualize_all_action_dictionaries_in_one_video(sampled_chunks, 42, "libero_spatial", 0, video_out_path)