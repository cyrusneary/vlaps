import collections
import dataclasses
import logging

class VLAAgent:
    def __init__(
            self,  
            vla,
            task_description,
            replan_steps : int = -1,
        ):
        """
        VLA Agent that queries a VLA with observations and a task description to get action chunks, 
        and follows those action chunks until they are completed.

        Inputs
        ------
        vla : 
            The vla to query
        task_description : str
            A description of the current task
        replan_steps : int
            The number of steps to execute before replanning. -1 will default to the chunk length of the VLA.
            Cannot be longer than the chunk length.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.task_description = task_description
        self.vla = vla

        self.replan_steps = replan_steps

        self.action_queue = collections.deque()

    def get_action(self, obs, state):

        if not self.action_queue:
            # Finished executing previous action chunk -- compute new chunk
            action_chunk = self.vla.infer(obs, self.task_description, state)

            if self.replan_steps != -1:
                assert (
                    len(action_chunk) >= self.replan_steps
                ), f"We want to replan every {self.replan_steps} steps, but beam search only predicted {len(action_chunk)} steps."
                self.action_queue.extend(action_chunk[: self.replan_steps])
            else:
                self.action_queue.extend(action_chunk)

        action = self.action_queue.popleft()

        return action