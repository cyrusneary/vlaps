import numpy as np

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def uniform_sample_k_chunks_from_library(library, vla_chunk=None, k=5):
    """
    Samples exactly k different chunks from the library uniformly.
    
    Args:
        library: (N, T, 7) array of action chunks.
        k: number of chunks to sample.
        
    Returns:
        (k, T, 7) array of k sampled chunks, each of shape (10,7).
    """
    N = len(library)
    assert k <= N, "Cannot sample more unique chunks than library size!"
    
    indices = np.random.choice(N, size=k, replace=False)
    return library[indices]

def get_chunk_probs_softmax_around_vla_chunk(chunk_array, vla_chunk, alpha=1.0):
    """
    Given a VLA chunk and an array of k candidate chunks, compute softmax-based probabilities
    using negative distance to the VLA chunk.

    Args:
        vla_chunk: (T, 7) np.array, the reference chunk
        chunk_array: (k, T, 7) np.array, k candidate chunks
        alpha: controls sharpness (higher alpha = more peaked)

    Returns:
        probs: (k,) np.array of normalized softmax probabilities
    """
    assert vla_chunk.shape[-1] == 7, "vla_chunk must have shape (T, 7)"
    assert chunk_array.ndim == 3 and chunk_array.shape[-1] == 7, "chunk_array must have shape (k,T,7)"

    vla_flat = vla_chunk.flatten()  # (70,)
    chunk_flat = chunk_array.reshape(chunk_array.shape[0], -1)  # (k, 70)

    # Compute L2 distances to vla_chunk
    distances = np.linalg.norm(chunk_flat - vla_flat[None, :], axis=1)  # (k,)

    # Compute softmax probabilities (numerical stability trick included)
    scaled = -alpha * distances
    scaled -= np.max(scaled)  # shift for numerical stability
    probs = np.exp(scaled)
    probs /= np.sum(probs)

    return probs

def get_chunk_probs_uniform(chunk_array, vla_chunk=None):
    """
    Assign equal probability to all chunks in the array
    
    Args:
        chunk_array: (k, 10, 7) np.array, k candidate chunks
    Returns:
        probs: (k,) np.array of uniform probabilities over k
    """
    assert chunk_array.ndim == 3 and chunk_array.shape[1:] == (10, 7), "chunk_array must have shape (k, 10, 7)"

    k = chunk_array.shape[0]
    probs = np.ones(k) / k  # Uniform distribution over k elements

    return probs

def sample_k_chunks_from_library(library, vla_chunk, k=5, alpha=1.0, epsilon=0.1, include_vla_chunk=True):
    """
    Samples exactly k different chunks from the library, biased toward a VLA-sampled seed chunk.
    
    Args:
        library: (N, 10, 7) array of action chunks.
        vla_chunk: (10, 7) array.
        k: number of chunks to sample.
        alpha: scaling factor for softmax sharpness.
        epsilon: probability of random exploration.
        include_vla_chunk: whether to include the VLA chunk in the sampled chunks.
        
    Returns:
        (k, 10, 7) array of k sampled chunks, each of shape (10,7).
    """
    N = len(library)
    assert k <= N, "Cannot sample more unique chunks than library size!"

    vla_chunk_flat = vla_chunk.flatten()
    library_flat = library.reshape(N, -1)

    # Precompute distances once
    distances = np.linalg.norm(library_flat - vla_chunk_flat, axis=1)
    probs = np.exp(-alpha * distances)
    probs /= probs.sum()

    sampled_indices = set()
    chunks = []

    if include_vla_chunk:
        # Include the VLA chunk if specified
        chunks.append(vla_chunk)
        # sampled_indices.add(np.where((library_flat == vla_chunk_flat).all(axis=1))[0][0])

    while len(chunks) < k:
        available_indices = list(set(range(N)) - sampled_indices)

        if len(available_indices) == 0:
            break  # Safety: no more available chunks (shouldn't happen if k <= N)

        if np.random.rand() < epsilon:
            # Random exploration
            idx = np.random.choice(available_indices)
        else:
            # Biased sampling
            adjusted_probs = probs[available_indices]
            adjusted_probs /= adjusted_probs.sum()  # Normalize over available only
            idx = np.random.choice(available_indices, p=adjusted_probs)

        sampled_indices.add(idx)
        chunks.append(library[idx])

    return np.stack(chunks)