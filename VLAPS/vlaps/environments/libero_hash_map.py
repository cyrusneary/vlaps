import numpy as np
import hashlib

class LiberoHashMap:
    def __init__(self, atol: float = 1e-9):
        """
        Initialize the cache.

        Args:
            atol (float): Absolute tolerance for state equivalence.
        """
        self.cache = {}
        self.atol = atol

    def _hash_state(self, state: np.ndarray) -> str:
        """
        Generate a hash for a state, rounding values to eliminate numerical noise.
        """
        # Quantize to the given tolerance
        quantized_int = np.round(state / self.atol).astype(np.int64)
        shape_bytes = np.array(quantized_int.shape, dtype=np.int32).tobytes()
        array_bytes = quantized_int.tobytes()
        return hashlib.sha256(shape_bytes + array_bytes).hexdigest()

        # quantized = np.round(state / self.atol) * self.atol
        # # quantized = quantized.astype(np.float32)
        # state_bytes = quantized.tobytes()
        # shape_bytes = np.array(quantized.shape, dtype=np.int32).tobytes()
        # return hashlib.sha256(shape_bytes + state_bytes).hexdigest()

    def _hash_task(self, task: str) -> str:
        return hashlib.sha256(task.encode('utf-8')).hexdigest()

    def _make_key(self, state: np.ndarray, task: str) -> str:
        state_hash = self._hash_state(state)
        task_hash = self._hash_task(task)
        return hashlib.sha256((state_hash + task_hash).encode('utf-8')).hexdigest()

    def get(self, state: np.ndarray, task: str):
        key = self._make_key(state, task)
        return self.cache.get(key, None)

    def set(self, state: np.ndarray, task: str, value):
        key = self._make_key(state, task)
        self.cache[key] = value

    def contains(self, state: np.ndarray, task: str) -> bool:
        key = self._make_key(state, task)
        return key in self.cache
    

def test_set_and_get():
    cache = LiberoHashMap()

    state = np.array([[1.0, 2.0], [3.0, 4.0]])
    task = "pick up cube"

    cache.set(state, task, "SUCCESS")
    assert cache.get(state, task) == "SUCCESS"
    print("Basic set/get test passed.")

def test_diff_tasks():
    cache = LiberoHashMap()

    state = np.array([[1.0, 2.0], [3.0, 4.0]])
    task = "pick up cube"

    cache.set(state, task, "SUCCESS")

    task2 = "stack cube"
    assert not cache.contains(state, task2)
    print("Different task gives different key.")

def test_tolerance_behavior():

    cache = LiberoHashMap()

    state = np.array([[1.0, 2.0], [3.0, 4.0]])
    task = "pick up cube"

    cache.set(state, task, "SUCCESS")

    state2 = state + 1e-10  # within tolerance (default atol=1e-9)
    assert cache.get(state2, task) == "SUCCESS"
    print("Tolerance test passed.")

    state3 = state + 1e-6  # outside default atol=1e-9
    assert not cache.contains(state3, task)
    print("Outside-tolerance difference leads to cache miss.")

def test_overwrite():

    cache = LiberoHashMap()

    state = np.array([[1.0, 2.0], [3.0, 4.0]])
    task = "pick up cube"

    cache.set(state, task, "SUCCESS")

    cache.set(state, task, "UPDATED")
    assert cache.get(state, task) == "UPDATED"
    print("Overwrite test passed.")

if __name__ == "__main__":
    test_diff_tasks()
    test_overwrite()
    test_set_and_get()
    test_tolerance_behavior()