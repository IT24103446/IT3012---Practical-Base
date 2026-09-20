# agent.py
class SimpleReflexAgent:
    """A stateless agent that acts solely from its current local percept."""

    def sense_and_act(self, percept: dict) -> str:
        # Condition-action rules: no memory or percept history is stored here.
        if percept.get('food_here'):
            return 'Up'
        if percept.get('wall_ahead'):
            return 'Up'
        return 'Right'


class ModelBasedAgent:
    """A reflex agent with memory of prior percepts and attempted actions."""

    def __init__(self):
        self.percept_history = []
        self.visited_states = set()
        self.last_action = None
        self.actions_pool = ['Up', 'Right', 'Down', 'Left']
        self.next_action_index = 0

    def sense_and_act(self, percept: dict) -> str:
        # Sensor model: remember the local reading received after the last action.
        local_state = (
            percept.get('wall_ahead', False),
            percept.get('food_here', False),
            percept.get('smells_toxin', False),
        )
        self.percept_history.append(local_state)

        # Transition model: associate the new perceived state with the action
        # that led to it, so repeated wall states trigger a different response.
        if self.last_action is not None:
            self.visited_states.add((local_state, self.last_action))

        if percept.get('wall_ahead'):
            action = next(action for action in self.actions_pool if action != self.last_action)
        elif percept.get('smells_toxin'):
            action = 'Left' if self.last_action != 'Left' else 'Down'
        else:
            action = self.actions_pool[self.next_action_index]
            self.next_action_index = (self.next_action_index + 1) % len(self.actions_pool)

        self.last_action = action
        return action


# Retained as an alias for earlier code that imported the starter-agent name.
GreedyGridAgent = SimpleReflexAgent
