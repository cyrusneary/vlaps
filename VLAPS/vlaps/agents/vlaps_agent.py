import numpy as np
import copy
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import collections

import logging
import copy

from PIL import Image

from vlaps.environments.libero_utils import \
    construct_policy_input, \
    build_libero_env_and_task_suite, \
    get_imgs_from_obs, \
    load_action_dictionary, \
    write_video

from vlaps.action_chunk_dictionaries.action_chunk_distributions import sample_k_chunks_from_library,\
     uniform_sample_k_chunks_from_library, get_chunk_probs_softmax_around_vla_chunk,\
        get_chunk_probs_uniform

from vlaps.agents.uniform_sampling_agent import UniformSamplingAgent

import time

class MCTSNode:

    def __init__(
            self, 
            state,
            obs,
            ctrl,
            node_reward,
            parent=None,
            terminal_node=False,
            task_complete=False,
            prior_value=None,
            instantiated=False,
            expanded=False,
        ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.state = state # Current state of the node
        self.obs = obs # Current observation at the node
        self.ctrl = ctrl # Value of VLAPSAgent.env_copy.sim.data.ctrl when this node is created
        self.node_reward = node_reward # Reward received during the step from the parent node to this node
        self.prior_value = prior_value # An estimated value of the node
        self.instantiated = instantiated # Has the node been instantiated?
        self.expanded = expanded # Has the node been fully expanded?
        self.prior_policy = None
        self.vla_chunk = None # The chunk sample from the VLA for this node.

        self.parent = parent # Parent node
        self.children = {} # Initialize an empty dictionary to store child nodes

        # Statistics of runs involving this node
        self.n_rollout_backups = 0
        self.n_v_estimate_backups = 0
        self.w_rollout_backups = 0
        self.w_v_estimate_backups = 0
        self.value = 0.0 

        self.sampled_action_chunks = None

        self.terminal_node = terminal_node # Is the node terminal?
        self.task_complete = task_complete # Was the task completed along the path from the parent to this node?

        self.depth = 0 if parent is None else parent.depth + 1
    

def count_nodes_at_depth(root, target_depth):
    count = 0

    def dfs(node):
        nonlocal count
        if node.depth == target_depth:
            count += 1
        for child in node.children.values():
            dfs(child)

    dfs(root)
    return count

def decrement_depths(node):
    node.depth -= 1
    for child in node.children.values():
        decrement_depths(child)

def get_current_controller_state(env):
    ctrl = {
        'ctrl' : env.sim.data.ctrl.copy(),
        'gripper_action' : env.robots[0].gripper.current_action.copy(),
    }
    return ctrl

def set_state_and_init_control_state(env, state, ctrl):
    """
    Set the state of the environment and the initial controller state
    """

    # Set the gripper current action to avoid having the gripper continue
    # whatever it was doing before the reset.
    env.robots[0].gripper.current_action = ctrl['gripper_action'].copy()

    # Set the [qpos, qvel] Mujoco state describing the environment.
    obs = env.set_init_state(state)

    return obs

def simulate_action_chunk_in_env(env, state, ctrl, action_chunk, steps_to_simulate=10):
    """
    Simulate the action chunk in the environment to generate a new node.
    """
    
    obs = set_state_and_init_control_state(env, state, ctrl)

    total_reward = 0
    any_done = False
    for action in action_chunk[:steps_to_simulate, :]:
        obs, reward, done, info = env.step(action.tolist())
        total_reward += reward
        if done:
            any_done = True
    new_state = env.get_sim_state()
    new_ctrl = get_current_controller_state(env)

    return new_state, obs, new_ctrl, total_reward, any_done

class VLAPSAgent:
    def __init__(
            self, 
            env_copy,
            vla,
            task_description,
            prior_policy_config,
            beta_distribution_config,
            rollout_policy_config,
            value_function_config,
            num_mc_samples : int = 100, # The total number of MCTS steps to execute each time VLAPSAgent.run() is called.
            exploration_weight : float = 1.4, # The exploration weighting parameter used in the selection scoring function.
            discount : float = 0.99, # The discount factor to apply for all reward and value "backup" computations.
            num_expansions_per_node : int = 10, # The total number of children to expand for each node. I.e., the tree's branching factor.
            num_expansions_per_mcts_step : int = 1, # The number of children to expand at a leaf node in a single MCTS step.
            final_action_sampling_temp : float = 1.0,
            max_rollout_length : int = 100, # The maximum number of steps to allow when simulating rollouts from leaf nodes.
            lambda_value_mixing_param : float = 1.0,
            chunk_steps_to_simulate : int = 10,
            max_tree_depth : int = 20,
            replan_steps : int = -1,
            terminate_search_on_simulated_success : bool = False, # Whether or not to end the MCTS procedure as soon as any trajectory reaching the goal has been found
            max_wall_time : int = -1, # Maximum allowable wall-clock time for search. An integer time in seconds, or -1 if no max is to be set.
        ):
        """
        Initialize the MCTS agent.
        """
        self.use_prior_value = False # Hardcoded to false for now since we don't have any prior values.

        self.logger = logging.getLogger(self.__class__.__name__)

        self.vla = vla

        self.env_copy = env_copy
        self.task_description = task_description

        self.prior_policy_config = prior_policy_config
        self.beta_distribution_config = beta_distribution_config
        self.rollout_policy_config = rollout_policy_config
        self.value_function_config = value_function_config

        self.num_mc_samples = num_mc_samples
        self.exploration_weight = exploration_weight
        self.discount = discount
        self.num_expansions_per_node = num_expansions_per_node
        self.num_expansions_per_mcts_step = num_expansions_per_mcts_step
        self.final_action_sampling_temp = final_action_sampling_temp
        self.max_rollout_length = max_rollout_length
        self.max_wall_time = max_wall_time

        assert lambda_value_mixing_param >= 0.0 and lambda_value_mixing_param <= 1.0, "The mixing parameter must be in the range [0, 1]."
        self.lambda_value_mixing_param = lambda_value_mixing_param

        self.chunk_steps_to_simulate = chunk_steps_to_simulate
        self.max_tree_depth = max_tree_depth
        self.replan_steps = replan_steps

        self.terminate_search_on_simulated_success = terminate_search_on_simulated_success

        self.total_mc_samples = 0
        self.total_simulations = 0

        self.dont_divide_by_zero_const = 1e-6

        self.root = None
        self.solving_leaves = []
        self.solving_action_chunks = []

        self.statistics = {
            'search_depth_list' : [],
            'search_plus_sim_depth_list' : [],
        }

        self.action_queue = collections.deque()

        self.tree_roots = []
        self.search_times = []
        self.all_sampled_chunks = []

        self._build_beta_distribution_sampler()
        self._build_policy_distribution_function()
        self._build_rollout_agent()

    def get_action(self, obs, state, ctrl):

        if not self.action_queue:
            # Nothing left in action queue. Run MCTS to get a new chunk.
            start_time = time.time()

            # # If the migrated root from the previous search step matches the state from the current environment step, 
            # # re-use that part of the search tree.
            # if self.root is not None and np.max(np.abs(self.root.state - state)) < 1e-9:
            #     root = self.root
            # else: # otherwise, create a new search tree using the observation and state, and start a new search tree.
            root = MCTSNode(
                state=state,
                obs=obs,
                ctrl=ctrl,
                node_reward=0.0,
                parent=None,
                terminal_node=False,
                task_complete=False,
                prior_value=0.0,
                expanded=False
            )
            self.root = root

            root = self.run(root)
            # self.tree_roots.append(copy.deepcopy(root))
            end_time = time.time()
            search_time = end_time - start_time
            self.search_times.append(search_time)

            action_chunk, child = self.get_action_chunk_from_solved_root_note(root)

            # # Migrate the root for the next search
            # self.root = copy.deepcopy(child)
            # self.root.parent = None # If its the new root it shouldn't have a parent.
            # decrement_depths(self.root) # Reduce the depth by one throughout the whole tree below the new root.

            if self.replan_steps != -1:
                assert (
                    len(action_chunk) >= self.replan_steps
                ), f"We want to replan every {self.replan_steps} steps, but beam search only predicted {len(action_chunk)} steps."
                self.action_queue.extend(action_chunk[: self.replan_steps])
            else:
                self.action_queue.extend(action_chunk)

        action = self.action_queue.popleft()

        return action
    
    def get_action_chunk_from_solved_root_note(self, root):
        """
        After having completed MCTS on the root node,
        return the action chunk to follow in the true environment, and the expected child/leaf node.
        """

        # # If the search has managed to reach a solving leaf node during the search, return the path to that node.
        # if self.solving_leaves:
        #     self.logger.info("Completed the search and found a terminal node.")
        #     chunks_list = []
        #     chunk_len_list = []
        #     for leaf in self.solving_leaves:
        #         action_chunks, _, _ = self.get_action_chunks_from_leaf_node(leaf)
        #         action_chunks = np.concatenate(action_chunks)
        #         chunks_list.append(action_chunks)
        #         chunk_len_list.append(action_chunks.shape[0])

        #     # Get the shortest chunk
        #     min_idx = np.argmin(chunk_len_list)
        #     shortest_chunk = chunks_list[min_idx]
        #     child = self.solving_leaves[min_idx]

        #     return shortest_chunk, child
        
        # If the search has managed to reach a solving leaf node during the search, return the path to that node.
        if self.solving_action_chunks:
            self.logger.info("Completed the search and found a terminal node.")
            chunks_list = []
            chunk_len_list = []
            for chunk in self.solving_action_chunks:
                chunks_list.append(chunk)
                chunk_len_list.append(chunk.shape[0])

            # Get the shortest chunk
            min_idx = np.argmin(chunk_len_list)
            shortest_chunk = chunks_list[min_idx]
            child = self.solving_leaves[min_idx]

            return shortest_chunk, child
            
        # otherwise, sample a child node from the root and return the action chunk to reach that child.
        else:
            self.logger.info("Exhausted search budget without finding terminal node. Migrating to new root.")

            probs = self.get_action_sample_dist(root)
            k = root.sampled_action_chunks.shape[0]
            chunk_idx = int(np.random.choice(k, p=probs))

            if chunk_idx not in root.children:
                self.logger.warning(f"Sampled unexpanded chunk {chunk_idx}. Returning chunk but no child node.")
                return root.sampled_action_chunks[chunk_idx], None

            return root.sampled_action_chunks[chunk_idx], root.children[chunk_idx]
    
    def get_action_chunks_from_leaf_node(self, node):
        """
        Trace back from a leaf node to the root, collecting action chunks, states, and observations.

        Args:
            node: The leaf node to trace back from.

        Returns:
            action_chunks: list of (T, d) arrays from root to leaf
            states: list of environment states from root to leaf
            obs: list of observations from root to leaf
        """
        action_chunks = []
        states = []
        obs = []

        this_node = node
        states.append(this_node.state)
        obs.append(this_node.obs)

        while this_node is not self.root:
            parent = this_node.parent

            # Find the chunk index that maps to this_node
            matching_keys = [k for k, v in parent.children.items() if v is this_node]
            assert len(matching_keys) == 1, "Node should appear exactly once in its parent's children."

            chunk_idx = matching_keys[0]
            action_chunks.append(parent.sampled_action_chunks[chunk_idx])

            this_node = parent
            states.append(this_node.state)
            obs.append(this_node.obs)

        # Reverse to get root-to-leaf order
        action_chunks.reverse()
        states.reverse()
        obs.reverse()

        return action_chunks, states, obs

    def run(self, root):
        """Run the MCTS algorithm starting from the root."""

        self.solving_action_chunks = []
        self.solving_leaves = []

        wall_time = time.time()

        for _ in range(self.num_mc_samples):

            self.logger.info(f"Running MCTS loop for step {_}.")

            if self.solving_action_chunks and self.terminate_search_on_simulated_success:
                break

            self.total_mc_samples += 1
            node = root

            # Selection
            while node.expanded and not node.terminal_node and node.depth < self.max_tree_depth:
                node, _ = self.select(node)

            # Handle hitting max depth
            if node.depth >= self.max_tree_depth:
                self.logger.info(f"[MCTS] Hit max tree depth at node with depth {node.depth}. Skipping expansion.")
                # Optional: run simulation or assign 0 reward
                self.backpropagate_rollout(node, 0.0)
                self.backpropagate_v_est(node, 0.0)
                continue  # Skip expansion for this node and go to next iteration

            # If the node we've reached through the search is terminal, we don't need to expand it.
            # Just backup the reward and continue with the next simulation.
            if node.terminal_node:
                if node.task_complete:
                    reward = 1.0
                else:
                    reward = 0.0
                self.backpropagate_rollout(node, reward)
                self.backpropagate_v_est(node, node.prior_value)

                # If the task is complete at the node, stop the search
                if node.task_complete:
                    self.solving_leaves.append(node)

                    chunk, _, _ = self.get_action_chunks_from_leaf_node(node)
                    self.solving_action_chunks.append(np.concatenate(chunk))
            else:
                # Otherwise, the node is not terminal and not fully expanded. We need to instantiate/expand it.

                if not node.instantiated:
                    self.instantiate_node(node)

                self.expand(node)

            # If the elapsed time is larger than the allowable maximum time, stop the search.
            if (not self.max_wall_time == -1) and (time.time() - wall_time) > self.max_wall_time:
                break

        return root
    
    def instantiate_node(self, node):
        """
        Query the VLA for the node, and save the infered chunk, 
        as well as the corresponding sampled chunks, and a prior policy distribution over them.
        """
        depth = node.depth
        width_at_depth = count_nodes_at_depth(self.root, depth)
        self.logger.info(f"[MCTS] Instantiating node at depth {depth}, tree width at this depth: {width_at_depth}")

        # Sample the vla
        vla_chunk = self.vla.infer(node.obs, self.task_description, node.state)

        node.vla_chunk = vla_chunk

        # Sample candidate action chunks for MCTS expansion
        action_chunks = self.sample_beta_distribution(vla_chunk, self.num_expansions_per_node)
        self.all_sampled_chunks.append(action_chunks)

        node.sampled_action_chunks = action_chunks

        # Set the prior policy for the node, so that the next time we get here, we can use it
        # to bias the selection process.
        node.prior_policy = self.get_prior_chunk_probabilities(action_chunks, vla_chunk)

        node.instantiated = True

    def expand(self, node):
        """Expand the tree by adding one or more new child nodes."""

        assert node.instantiated, "A node is being expanded before being instantiated."

        # Chose the child(ren) to expand.

        top_actions_to_expand = self.get_top_n_unexpanded_chunks(node)
        if len(top_actions_to_expand) == 0:
            return None  # Node is fully expanded
        
        for i in top_actions_to_expand:

            # Simulate action chunks and generate child nodes
            action_chunk = node.sampled_action_chunks[i]
            self.env_copy.env.timestep = 0 # Hack to prevent the env running out of timesteps
            # for _ in range(4):
            new_state, new_obs, new_ctrl, new_reward, done = simulate_action_chunk_in_env(
                self.env_copy, 
                node.state,
                node.ctrl, 
                action_chunk,
                steps_to_simulate=self.chunk_steps_to_simulate,
            )
            new_node = MCTSNode(
                state=new_state,
                obs=new_obs,
                ctrl=new_ctrl,
                node_reward=new_reward,
                parent=node,
                terminal_node=done,
                task_complete=done,
                prior_value=0.0,
                expanded=False,
                instantiated=False,
            )
            node.children[i] = new_node

            # If the node is terminal, end the search and backpropagate reward.
            if node.children[i].terminal_node:
                if node.children[i].task_complete:
                    reward = 1.0
                else:
                    reward = 0.0
                self.backpropagate_rollout(node.children[i], reward)
                self.backpropagate_v_est(node.children[i], node.children[i].prior_value)

                # If the task is complete at the node, add it to the list of solving nodes in the search tree
                if node.children[i].task_complete:
                    self.solving_leaves.append(node.children[i])

                    chunk, _, _ = self.get_action_chunks_from_leaf_node(node.children[i])
                    self.solving_action_chunks.append(np.concatenate(chunk))

            # Otherwise, simulate a rollout.
            else:
                new_state, new_obs, new_ctrl, total_reward, action_chunk, done, rollout_steps = self.simulate(node.children[i])
                self.backpropagate_rollout(node.children[i], total_reward)
                self.backpropagate_v_est(node.children[i], node.children[i].prior_value)

                # print(self.vla.num_cache_accesses)

                # If the simulation completed the task, save the action chunk leading from the root to task completion.
                if done and total_reward > 0.0:
                    # TODO, fix the fact that this isn't saving a valuable leaf node. 
                    # It is saving the resetted state and obs after the agent finishes the task.
                    # maybe this is the behavior we want. Need to think about it.
                    solving_node = MCTSNode( 
                        state=new_state,
                        obs=new_obs,
                        ctrl=new_ctrl,
                        node_reward=total_reward,
                        parent=node,
                        terminal_node=done,
                        task_complete=done,
                        prior_value=0.0,
                        expanded=False,
                        instantiated=False,
                    )
                    self.solving_leaves.append(solving_node)

                    solving_chunk, _, _ = self.get_action_chunks_from_leaf_node(node.children[i])
                    solving_chunk.append(action_chunk)
                    self.solving_action_chunks.append(np.concatenate(solving_chunk))

        # Verify whether the node is now fully expanded (if all sampled action chunks have been expanded to generate corresponding child nodes).
        all_indices = set(range(node.sampled_action_chunks.shape[0]))
        expanded_indices = set(node.children.keys())
        if all_indices == expanded_indices:
            node.expanded = True
        
        # if len(list(node.children.keys())) == node.sampled_action_chunks.shape[0]:
        #     node.expanded = True

    def get_top_n_unexpanded_chunks(self, node):
        """
        Get the top n unexpanded action candidates, ranked according to their probabilities
        under the node.prior_policy in descending order.

        n is given by self.num_expansions_per_mcts_step
        """
        k = node.sampled_action_chunks.shape[0]
        all_indices = np.arange(k)
        
        # Identify unexpanded indices
        unexpanded_mask = np.array([i not in node.children for i in all_indices])
        
        # Get the prior probabilities for unexpanded chunks only
        unexpanded_probs = node.prior_policy[unexpanded_mask]
        unexpanded_indices = all_indices[unexpanded_mask]
        
        # Sort unexpanded chunks by prior probability (descending)
        sorted_indices = unexpanded_indices[np.argsort(-unexpanded_probs)]
        
        # Select top-n
        top_n = sorted_indices[:self.num_expansions_per_mcts_step]
        
        return top_n

    def select(self, node):
        """Select the best child using PUCT algorithm from Alphazero Go paper."""        
        action_scores = self.pucb_scores(node)
        max_child, max_score = max(action_scores, key=lambda x: x[1])
        return max_child, max_score
    
    def pucb_scores(self, node):
        exploitation_scores = self.get_children_exploitation_scores(node)
        exploration_scores = self.get_pucb_exploration_scores(node)
        return [
            (child, exploitation + exploration) for 
            child, exploitation, exploration in 
            zip(node.children.values(), exploitation_scores, exploration_scores)
        ]

    def get_children_exploitation_scores(self, node):
        exploitation_scores = [self.get_exploitation_score(child) for child in node.children.values()]
        return exploitation_scores

    def get_exploitation_score(self, node):
        # score = node.value / node.visits
        score = self.get_mcts_value_estimate(node)
        if self.use_prior_value:
            v_est = self.get_prior_backup_value_estimate(node)
            score = (1.0 - self.lambda_value_mixing_param) * v_est + self.lambda_value_mixing_param * score
        return score

    def get_mcts_value_estimate(self, node):
        return node.w_rollout_backups / (node.n_rollout_backups + self.dont_divide_by_zero_const)

    def get_prior_backup_value_estimate(self, node):
        return node.w_v_estimate_backups / (node.n_v_estimate_backups + self.dont_divide_by_zero_const)
    
    def get_pucb_exploration_scores(self, node):
            exploration_scores = []
            for child, p_val in zip(node.children.values(), node.prior_policy):
                exploration_scores.append(self.exploration_weight * p_val * (np.sqrt(node.n_rollout_backups) / (child.n_rollout_backups + 1)))
            return exploration_scores

    def simulate(self, node):
        """Simulate a random rollout to a terminal state."""
        self.total_simulations = self.total_simulations + 1

        total_reward = 0
        done = False

        temp_discount = 1.0
        rollout_steps = 0

        action_list = []
        
        # self.logger.info("Running simulated rollout")

        self.env_copy.env.timestep = 0 # Hack to prevent the env running out of timesteps
        obs = set_state_and_init_control_state(self.env_copy, node.state, node.ctrl)
        # self.env_copy.env.sim.data.ctrl = node.ctrl.copy()
        # self.env_copy.set_init_state(node.state)
        obs = node.obs

        while not done and rollout_steps < self.max_rollout_length:

            state = self.env_copy.get_sim_state()

            action = self.rollout_agent.get_action(obs, state)
            obs, reward, done, info = self.env_copy.step(action.tolist())

            action_list.append(action)

            total_reward += temp_discount * reward
            temp_discount = temp_discount * self.discount
            rollout_steps += 1
        
        new_state = self.env_copy.get_sim_state()
        new_ctrl = get_current_controller_state(self.env_copy)
        # new_ctrl = self.env_copy.env.sim.data.ctrl.copy()

        if action_list:
            action_chunk = np.stack(action_list)
        else:
            action_chunk = None
        return new_state, obs, new_ctrl, total_reward, action_chunk, done, rollout_steps
    
    def backpropagate_rollout(self, node, reward):
        """Propagate simulation results up the tree."""
        num_timesteps_between_nodes = np.min([self.root.sampled_action_chunks.shape[1], self.chunk_steps_to_simulate])
        gamma_t = self.discount ** num_timesteps_between_nodes

        temp_discount = 1.0
        while node is not None:
            node.n_rollout_backups += 1
            node.w_rollout_backups += temp_discount * reward
            temp_discount = temp_discount * gamma_t
            node = node.parent

    def backpropagate_v_est(self, node, value_estimate):
        """Propagate simulation results up the tree."""
        num_timesteps_between_nodes = np.min([self.root.sampled_action_chunks.shape[1], self.chunk_steps_to_simulate])
        gamma_t = self.discount ** num_timesteps_between_nodes

        temp_discount = 1.0
        while node is not None:
            node.n_v_estimate_backups += 1
            node.w_v_estimate_backups += temp_discount * value_estimate
            temp_discount = temp_discount * gamma_t
            node = node.parent

    def get_action_w_max_value(self, node):
        """Get the action with the maximum q value."""
        return max(
            node.children.items(), key=lambda item: self.get_exploitation_score(item[1])
        )[0]

    def get_action_sample_dist(self, node):
        """
        Get a probability distribution over all sampled action chunks at this node,
        using visit counts (with temperature) for expanded children.
        Unexpanded chunks get a default visit count of 1e-6.
        """
        k = node.sampled_action_chunks.shape[0]
        visit_counts = np.full(k, 1e-6)  # small baseline for unvisited

        for chunk_idx, child in node.children.items():
            visit_counts[chunk_idx] = child.n_rollout_backups

        # Apply temperature
        temp = self.final_action_sampling_temp
        logits = visit_counts ** (1 / temp)

        # Normalize
        action_distribution = logits / logits.sum()

        return action_distribution
    
        # """Get the action distribution from the current node."""
        # # visits_list = []
        # # for i in range(len(node.children)):
        # #     visits_list.append(node.children[i].n_rollout_backups)
        # action_visits = np.array([child.n_rollout_backups for child in node.children.values()])
        # action_distribution = action_visits**(1/self.final_action_sampling_temp) / sum(action_visits**(1/self.final_action_sampling_temp))
        # return action_distribution
    
    def _build_beta_distribution_sampler(self):
        """
        Build the beta distribution sampler for sampling action chunks.
        """
        params = self.beta_distribution_config
        assert params['beta_distribution_type'] in ['uniform', 'softmax'], "Beta distribution type must be either 'uniform' or 'softmax'."
        
        self.beta_distribution_chunk_library = load_action_dictionary(params['chunk_library_path'])

        if params['beta_distribution_type'] == 'softmax':
            self._chunk_sampler = sample_k_chunks_from_library
            self._chunk_sampler_args = {
                'alpha': params['distribution_parameters']['alpha'],
                'epsilon': params['distribution_parameters']['epsilon'],
                'include_vla_chunk': params['distribution_parameters']['include_vla_chunk'],
            }
        elif params['beta_distribution_type'] == 'uniform':
            self._chunk_sampler = uniform_sample_k_chunks_from_library
            self._chunk_sampler_args = {
                'epsilon': params['distribution_parameters']['epsilon'],
            }

        def samle_beta_distribution(vla_chunk=None, num_chunks=5):

            """
            Sample k action chunks from the beta distribution.
            Args:
                vla_chunk: (10, 7) array.
                num_chunks: number of chunks to sample.
            Returns:
                (k, 10, 7) array of k sampled chunks, each of shape (10,7).
            """
            return self._chunk_sampler(
                self.beta_distribution_chunk_library,
                vla_chunk,
                num_chunks,
                **self._chunk_sampler_args
            )
        self.sample_beta_distribution = samle_beta_distribution

    def _build_policy_distribution_function(self):

        params = self.prior_policy_config
        assert params['prior_policy_type'] in ['uniform', 'softmax'], "Prior policy distribution type must be either uniform or softmax"

        if params['prior_policy_type'] == "softmax":
            self._probability_function = get_chunk_probs_softmax_around_vla_chunk
            self._probability_function_params = {
                'alpha' : params['distribution_parameters']['alpha']
            }

        elif params['prior_policy_type'] == "uniform":
            self._probability_function = get_chunk_probs_uniform
            self._probability_function_params = {}

        def get_prior_chunk_probabilities(chunk_list, vla_chunk=None):
            return self._probability_function(
                chunk_list,
                vla_chunk,
                **self._probability_function_params
            )
        
        self.get_prior_chunk_probabilities = get_prior_chunk_probabilities


    def _build_rollout_agent(self):

        params = self.rollout_policy_config
        assert params["rollout_policy_type"] in ['uniform', 'vla'], "Rollout policy type must be either 'uniform' or 'vla'."

        if params['rollout_policy_type'] == "vla":
            from vla_agent import VLAAgent
            rollout_agent = VLAAgent(
                self.vla,
                self.task_description,
            )
        elif params["rollout_policy_type"] == "uniform":
            rollout_agent = UniformSamplingAgent(
                action_chunk_library_path=params['chunk_library_path'],
                replan_steps=params['replan_steps']
            )

        self.rollout_agent = rollout_agent