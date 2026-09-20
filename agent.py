# agent.py
from collections import deque
import heapq
import math


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


class SearchAgent:
    """A goal-based agent that plans a route to food before moving."""

    ACTIONS = (
        ('Up', 0, 1),
        ('Right', 1, 0),
        ('Down', 0, -1),
        ('Left', -1, 0),
    )

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def manhattan_distance(self, pos, goal):
        """Estimate 4-way grid distance by horizontal plus vertical steps."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        """Return the straight-line distance between two grid coordinates."""
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])

    @classmethod
    def _successors(cls, position, walls, grid_size):
        width, height = grid_size
        for action, dx, dy in cls.ACTIONS:
            next_position = (position[0] + dx, position[1] + dy)
            if (
                0 <= next_position[0] < width
                and 0 <= next_position[1] < height
                and next_position not in walls
            ):
                yield action, next_position

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Find a shortest unit-cost path using a FIFO frontier."""
        start, goal, wall_set = tuple(start_pos), tuple(goal_pos), set(walls)
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            position, path = frontier.popleft()
            if position == goal:
                return path
            for action, next_position in self._successors(position, wall_set, grid_size):
                if next_position not in reached:
                    reached.add(next_position)
                    frontier.append((next_position, path + [action]))
        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Find a path using a LIFO frontier; it is not guaranteed shortest."""
        start, goal, wall_set = tuple(start_pos), tuple(goal_pos), set(walls)
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            position, path = frontier.pop()
            if position == goal:
                return path
            for action, next_position in self._successors(position, wall_set, grid_size):
                if next_position not in reached:
                    reached.add(next_position)
                    frontier.append((next_position, path + [action]))
        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Find a lowest-cost path using a priority queue ordered by g(n)."""
        start, goal, wall_set = tuple(start_pos), tuple(goal_pos), set(walls)
        frontier = [(0, 0, start, [])]
        reached_cost = {start: 0}
        tie_breaker = 0

        while frontier:
            cost, _, position, path = heapq.heappop(frontier)
            if cost != reached_cost.get(position):
                continue
            if position == goal:
                return path
            for action, next_position in self._successors(position, wall_set, grid_size):
                next_cost = cost + 1
                if next_cost < reached_cost.get(next_position, float('inf')):
                    reached_cost[next_position] = next_cost
                    tie_breaker += 1
                    heapq.heappush(frontier, (next_cost, tie_breaker, next_position, path + [action]))
        return None

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """Find a lowest-cost path using the A* evaluation f(n) = g(n) + h(n)."""
        start, goal, wall_set = tuple(start_pos), tuple(goal_pos), set(walls)
        heuristic = self.euclidean_distance if heuristic_type.lower() == 'euclidean' else self.manhattan_distance
        frontier = [(heuristic(start, goal), 0, start, [])]
        reached_states = set()

        while frontier:
            _, g_cost, current_pos, path_taken = heapq.heappop(frontier)
            if current_pos in reached_states:
                continue
            if current_pos == goal:
                return path_taken

            reached_states.add(current_pos)
            for action, next_position in self._successors(current_pos, wall_set, grid_size):
                if next_position not in reached_states:
                    new_g_cost = g_cost + 1
                    new_h_cost = heuristic(next_position, goal)
                    new_f_cost = new_g_cost + new_h_cost
                    heapq.heappush(
                        frontier,
                        (new_f_cost, new_g_cost, next_position, path_taken + [action]),
                    )

    def sense_and_act(self, percept: dict) -> str:
        """Plan to the closest visible food only when no remaining plan exists."""
        if not self.plan:
            start = tuple(percept['agent_pos'])
            food_positions = [tuple(food) for food in percept['all_food']]
            if not food_positions:
                return 'Stay'

            goal = min(food_positions, key=lambda food: self.manhattan_distance(start, food))
            algorithm = self.active_algo.upper()
            if algorithm == 'DFS':
                self.plan = self.dfs_search(start, goal, percept['walls'], percept['grid_size']) or []
            elif algorithm == 'UCS':
                self.plan = self.ucs_search(start, goal, percept['walls'], percept['grid_size']) or []
            elif algorithm == 'ASTAR':
                self.plan = self.astar_search(start, goal, percept['walls'], percept['grid_size']) or []
            else:
                self.plan = self.bfs_search(start, goal, percept['walls'], percept['grid_size']) or []

        return self.plan.pop(0) if self.plan else 'Stay'


# Retained as an alias for earlier code that imported the starter-agent name.
GreedyGridAgent = SimpleReflexAgent
