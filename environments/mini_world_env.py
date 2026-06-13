"""
Mini Research World — a two-layer research-lab gridworld for tabular RL.

Layer 0:
    Collect door key, open locked door, activate terminal, collect portal core,
    then use portal to enter Layer 1.

Layer 1:
    Avoid pits, collect lab key, use machine, submit final result at goal.

This environment is designed for:
    - Random agents
    - Q-learning
    - SARSA
    - Expected SARSA
    -  DQN / Dueling vaiants of DQN/ PPO / and other algorithms
    - MARL / world-model extensions
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------
# Fixed 8x8 layouts
# ---------------------------------------------------------------------

LAYER_0_LAYOUT = [
    "WWWWWWWW",
    "WA..K..W",
    "W.WW.W.W",
    "W..D.T.W",
    "W.W..W.W",
    "W..C.O.W",
    "W......W",
    "WWWWWWWW",
]

LAYER_1_LAYOUT = [
    "WWWWWWWW",
    "WA.....W",
    "W.WX.W.W",
    "W...L..W",
    "W.X..W.W",
    "W...MG.W",
    "W......W",
    "WWWWWWWW",
]

# Layer-1 portal entry point.
# The 'A' in Layer 1 is not the episode start.
# It is the arrival cell after portal transition.
LAYER_1_ENTRY = (1, 1)


# ---------------------------------------------------------------------
# Action indices
# ---------------------------------------------------------------------

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_INTERACT = 4

ACTION_NAMES = {
    ACTION_UP: "up",
    ACTION_DOWN: "down",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
    ACTION_INTERACT: "interact",
}


# ---------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------

REWARD_MOVE = -0.05
REWARD_INVALID_MOVE = -0.5
REWARD_INVALID_INTERACT = -0.5

REWARD_DOOR_NO_KEY = -1.0
REWARD_PORTAL_NOT_READY = -1.0
REWARD_MACHINE_NO_KEY = -1.0

REWARD_COLLECT_KEY = 1.0
REWARD_OPEN_DOOR = 2.0
REWARD_ACTIVATE_TERMINAL = 2.0
REWARD_COLLECT_CORE = 2.0
REWARD_USE_PORTAL = 3.0

REWARD_PIT = -5.0

REWARD_COLLECT_LAB_KEY = 2.0
REWARD_USE_MACHINE = 3.0

REWARD_SUBMIT_SUCCESS = 10.0
REWARD_SUBMIT_FAIL = -2.0

REWARD_TIMEOUT = -5.0


class MiniResearchWorldEnv(gym.Env):
    """
    Two-layer research-lab gridworld.

    Observation:
        [
            layer,
            row,
            col,
            has_key,
            door_open,
            terminal_activated,
            has_portal_core,
            portal_open,
            has_lab_key,
            machine_used,
            has_result
        ]

    Actions:
        0 = up
        1 = down
        2 = left
        3 = right
        4 = interact
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, max_steps=100):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        # Store layouts as char arrays for easy lookup.
        self._layouts = np.array(
            [
                [list(row) for row in LAYER_0_LAYOUT],
                [list(row) for row in LAYER_1_LAYOUT],
            ]
        )

        # Five discrete actions:
        # up, down, left, right, interact
        self.action_space = spaces.Discrete(5)

        # MultiDiscrete observation:
        # layer: 0/1
        # row: 0-7
        # col: 0-7
        # all remaining flags: 0/1
        self.observation_space = spaces.MultiDiscrete(
            [2, 8, 8, 2, 2, 2, 2, 2, 2, 2, 2]
        )

        self.layer = 0
        self.row = 1
        self.col = 1

        self.has_key = False
        self.door_open = False
        self.terminal_activated = False
        self.has_portal_core = False
        self.portal_open = False

        self.has_lab_key = False
        self.machine_used = False
        self.has_result = False

        self.steps = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Agent always starts in Layer 0 at row 1, col 1.
        self.layer = 0
        self.row = 1
        self.col = 1

        self.has_key = False
        self.door_open = False
        self.terminal_activated = False
        self.has_portal_core = False
        self.portal_open = False

        self.has_lab_key = False
        self.machine_used = False
        self.has_result = False

        self.steps = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        self.steps += 1

        terminated = False
        truncated = False

        if action == ACTION_INTERACT:
            reward, terminated = self._interact()
        else:
            reward = self._move(action)

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            reward += REWARD_TIMEOUT

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the current layer as text.

        The agent is overlaid as 'A'.
        Collected/used objects are hidden from display.
        """
        grid = self._build_render_grid()

        output = "\n".join("".join(row) for row in grid)
        output += f"\nLayer {self.layer} | step {self.steps}/{self.max_steps}"

        if self.render_mode == "human":
            print(output)

        return output

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        return np.array(
            [
                self.layer,
                self.row,
                self.col,
                int(self.has_key),
                int(self.door_open),
                int(self.terminal_activated),
                int(self.has_portal_core),
                int(self.portal_open),
                int(self.has_lab_key),
                int(self.machine_used),
                int(self.has_result),
            ],
            dtype=np.int64,
        )

    def _get_info(self):
        return {
            "layer": self.layer,
            "position": (self.row, self.col),
            "has_key": self.has_key,
            "door_open": self.door_open,
            "terminal_activated": self.terminal_activated,
            "has_portal_core": self.has_portal_core,
            "portal_open": self.portal_open,
            "has_lab_key": self.has_lab_key,
            "machine_used": self.machine_used,
            "has_result": self.has_result,
            "steps": self.steps,
        }

    # ------------------------------------------------------------------
    # Movement logic
    # ------------------------------------------------------------------

    def _move(self, action):
        """
        Attempt to move in the given direction.

        Returns:
            reward
        """

        delta = {
            ACTION_UP: (-1, 0),
            ACTION_DOWN: (1, 0),
            ACTION_LEFT: (0, -1),
            ACTION_RIGHT: (0, 1),
        }.get(action)

        if delta is None:
            return REWARD_INVALID_MOVE

        new_row = self.row + delta[0]
        new_col = self.col + delta[1]

        # Walls and closed doors block movement.
        if self._is_wall(self.layer, new_row, new_col):
            return REWARD_INVALID_MOVE

        if self._is_door_blocking(self.layer, new_row, new_col):
            return REWARD_INVALID_MOVE

        previous_row, previous_col = self.row, self.col

        self.row = new_row
        self.col = new_col

        # Pit logic:
        # If the agent enters a pit, punish and bounce back.
        # Episode does not terminate.
        if self.layer == 1 and self._cell_at(self.layer, new_row, new_col) == "X":
            self.row = previous_row
            self.col = previous_col
            return REWARD_PIT

        return REWARD_MOVE

    # ------------------------------------------------------------------
    # Interaction logic
    # ------------------------------------------------------------------

    def _interact(self):
        """
        Handle the interact action.

        Returns:
            reward, terminated
        """

        cell = self._cell_at(self.layer, self.row, self.col)

        # --------------------------------------------------------------
        # Layer 0 interactions
        # --------------------------------------------------------------
        if self.layer == 0:
            if cell == "K":
                if not self.has_key:
                    self.has_key = True
                    return REWARD_COLLECT_KEY, False
                return REWARD_INVALID_INTERACT, False

            if cell == "T":
                if not self.terminal_activated:
                    self.terminal_activated = True
                    return REWARD_ACTIVATE_TERMINAL, False
                return REWARD_INVALID_INTERACT, False

            if cell == "C":
                if not self.has_portal_core:
                    self.has_portal_core = True
                    return REWARD_COLLECT_CORE, False
                return REWARD_INVALID_INTERACT, False

            if cell == "O":
                return self._try_portal(), False

            # Door is opened from an adjacent cell because closed door
            # blocks direct movement.
            door_reward = self._try_open_door()
            if door_reward is not None:
                return door_reward, False

        # --------------------------------------------------------------
        # Layer 1 interactions
        # --------------------------------------------------------------
        if self.layer == 1:
            if cell == "L":
                if not self.has_lab_key:
                    self.has_lab_key = True
                    return REWARD_COLLECT_LAB_KEY, False
                return REWARD_INVALID_INTERACT, False

            if cell == "M":
                if self.machine_used:
                    return REWARD_INVALID_INTERACT, False

                if not self.has_lab_key:
                    return REWARD_MACHINE_NO_KEY, False

                self.machine_used = True
                self.has_result = True
                return REWARD_USE_MACHINE, False

            if cell == "G":
                if self.has_result:
                    return REWARD_SUBMIT_SUCCESS, True
                return REWARD_SUBMIT_FAIL, False

        return REWARD_INVALID_INTERACT, False

    def _try_open_door(self):
        """
        Try opening the locked door.

        Door can be opened only when:
            - door is still closed
            - agent is adjacent to D
            - agent has collected K
            - agent takes interact
        """

        if self.door_open:
            return None

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_row = self.row + dr
            adj_col = self.col + dc

            if self._cell_at(self.layer, adj_row, adj_col) == "D":
                if self.has_key:
                    self.door_open = True
                    return REWARD_OPEN_DOOR

                return REWARD_DOOR_NO_KEY

        return None

    def _try_portal(self):
        """
        Try using the portal.

        Portal works only when:
            - has_key=True
            - door_open=True
            - terminal_activated=True
            - has_portal_core=True

        Successful portal usage moves agent to Layer 1 entry.
        """

        if self.portal_open:
            return REWARD_INVALID_INTERACT

        ready = (
            self.has_key
            and self.door_open
            and self.terminal_activated
            and self.has_portal_core
        )

        if not ready:
            return REWARD_PORTAL_NOT_READY

        self.portal_open = True
        self.layer = 1
        self.row, self.col = LAYER_1_ENTRY

        return REWARD_USE_PORTAL

    # ------------------------------------------------------------------
    # Grid helper methods
    # ------------------------------------------------------------------

    def _cell_at(self, layer, row, col):
        """
        Return map symbol at a given location.

        Out-of-bounds positions are treated as walls.
        """

        if row < 0 or row >= 8:
            return "W"

        if col < 0 or col >= 8:
            return "W"

        return self._layouts[layer, row, col]

    def _is_wall(self, layer, row, col):
        return self._cell_at(layer, row, col) == "W"

    def _is_door_blocking(self, layer, row, col):
        return self._cell_at(layer, row, col) == "D" and not self.door_open

    def _build_render_grid(self):
        """
        Build display grid for current layer.

        This does not modify the real layout.
        It only controls what is shown during render.
        """

        grid = []

        for r in range(8):
            row_chars = []

            for c in range(8):
                if r == self.row and c == self.col:
                    row_chars.append("A")
                    continue

                symbol = self._cell_at(self.layer, r, c)

                # 'A' in layouts only marks spawn/entry locations.
                # It should not be treated as a persistent object.
                if symbol == "A":
                    symbol = "."

                # Hide objects after they are collected/used.
                if symbol == "K" and self.has_key:
                    symbol = "."
                elif symbol == "D" and self.door_open:
                    symbol = "."
                elif symbol == "T" and self.terminal_activated:
                    symbol = "."
                elif symbol == "C" and self.has_portal_core:
                    symbol = "."
                elif symbol == "O" and self.portal_open:
                    symbol = "."
                elif symbol == "L" and self.has_lab_key:
                    symbol = "."
                elif symbol == "M" and self.machine_used:
                    symbol = "."

                row_chars.append(symbol)

            grid.append(row_chars)

        return grid


# ----------------------------------------------------------------------
# Manual successful-path demo
# ----------------------------------------------------------------------

def run_manual_solution_demo():
    """
    Run a hardcoded action sequence that solves the environment.

    This is useful before training RL algorithms because it proves:
        1. The map is solvable.
        2. Rewards work.
        3. Termination works.
        4. Portal transition works.
    """

    env = MiniResearchWorldEnv(render_mode="human", max_steps=100)

    obs, info = env.reset()

    print("\n========== MINI RESEARCH WORLD: MANUAL SOLUTION DEMO ==========\n")
    print("Initial observation:", obs)
    env.render()

    # Successful route:
    #
    # Layer 0:
    # Start (1,1)
    # Go to K at (1,4): right, right, right, interact
    # Go to door-adjacent cell (3,4): down, down, interact
    # Go to T at (3,5): right, interact
    # Go to C at (5,3): left, left, down, down, interact
    # Go to O at (5,5): right, right, interact
    #
    # Layer 1:
    # Entry at (1,1)
    # Go to L at (3,4): right, right, right, down, down, interact
    # Go to M at (5,4): down, down, interact
    # Go to G at (5,5): right, interact

    manual_actions = [
        ACTION_RIGHT,
        ACTION_RIGHT,
        ACTION_RIGHT,
        ACTION_INTERACT,  # collect K

        ACTION_DOWN,
        ACTION_DOWN,
        ACTION_INTERACT,  # open D from adjacent cell

        ACTION_RIGHT,
        ACTION_INTERACT,  # activate T

        ACTION_LEFT,
        ACTION_LEFT,
        ACTION_DOWN,
        ACTION_DOWN,
        ACTION_INTERACT,  # collect C

        ACTION_RIGHT,
        ACTION_RIGHT,
        ACTION_INTERACT,  # use O portal

        ACTION_RIGHT,
        ACTION_RIGHT,
        ACTION_RIGHT,
        ACTION_DOWN,
        ACTION_DOWN,
        ACTION_INTERACT,  # collect L

        ACTION_DOWN,
        ACTION_DOWN,
        ACTION_INTERACT,  # use M

        ACTION_RIGHT,
        ACTION_INTERACT,  # submit at G
    ]

    total_reward = 0.0

    for step_idx, action in enumerate(manual_actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"\nStep {step_idx:02d} | "
            f"action={ACTION_NAMES[action]:>8} | "
            f"reward={reward:>6.2f} | "
            f"total_reward={total_reward:>6.2f} | "
            f"obs={obs} | "
            f"terminated={terminated} | "
            f"truncated={truncated}"
        )

        env.render()

        if terminated or truncated:
            break

    print("\n========== DEMO FINISHED ==========")
    print("Final observation:", obs)
    print("Final info:", info)
    print("Total reward:", total_reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)


# ----------------------------------------------------------------------
# Random smoke test
# ----------------------------------------------------------------------

def run_random_smoke_test(num_steps=10):
    """
    Run random actions just to check that the environment does not crash.
    This is not expected to solve the task.
    """

    env = MiniResearchWorldEnv(render_mode="human", max_steps=100)

    obs, info = env.reset()

    print("\n========== MINI RESEARCH WORLD: RANDOM SMOKE TEST ==========\n")
    print("Initial observation:", obs)
    env.render()

    for i in range(num_steps):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"\nStep {i + 1:02d} | "
            f"action={ACTION_NAMES[action]:>8} | "
            f"reward={reward:>6.2f} | "
            f"obs={obs} | "
            f"terminated={terminated} | "
            f"truncated={truncated}"
        )   

        env.render()

        if terminated or truncated:
            print("Episode ended.")
            break


if __name__ == "__main__":
    # First run the manual solution to prove the environment is solvable.
    run_manual_solution_demo()

    # Then optionally run a random smoke test.
    # Uncomment this if needed.
    # run_random_smoke_test(num_steps=10)