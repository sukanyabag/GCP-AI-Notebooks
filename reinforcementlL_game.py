#We can try and save the gumdrop ourselves! This is a common game people begin their Reinforcement Learning journey with, and is included in the OpenAI's python package Gym and is aptly named FrozenLake-v0 (code). No time to waste, let's get the environment up and running. Run the below to install the needed libraries if they are not installed already.
# Ensure the right version of Tensorflow is installed.
!pip install tensorflow==2.1 --user
!pip install gym==0.12.5 --user

'''
There are four methods from Gym that are going to be useful to us in order to save the gumdrop.

make allows us to build the environment or game that we can pass actions to
reset will reset an environment to it's starting configuration and return the state of the player
render displays the environment for human eyes
step takes an action and returns the player's next state.
Let's make, reset, and render the game. The output is an ANSI string with the following characters:

S for starting point
F for frozen
H for hole
G for goal
A red square indicates the current position
'''
import gym
import numpy as np
import random
env = gym.make('FrozenLake-v0', is_slippery=False)
state = env.reset()
env.render()
print(state)
def print_state(state, done):
    statement = "Still Alive!"
    if done:
        statement = "Cocoa Time!" if state == 15 else "Game Over!" 
    print(state, "-", statement)
    
'''
We can control the gumdrop ourselves with the step method. 
Run the below cell over and over again trying to move from the starting position to the goal. Good luck!
'''
#0 left
#1 down
#2 right
#3 up

# Uncomment to reset the game
#env.reset()
action = 2  # Change me, please!
state, _, done, _ = env.step(action)
env.render()
print_state(state, done)
'''
Were you able to reach the hot chocolate? If so, great job! 
There are multiple paths through the maze. One solution is [1, 1, 2, 2, 1, 2]. 
Let's loop through our actions in order to get used to interacting with the environment programmatically.
'''
def play_game(actions):
    state = env.reset()
    step = 0
    done = False

    while not done and step < len(actions):
        action = actions[step]
        state, _, done, _ = env.step(action)
        env.render()
        step += 1
        print_state(state, done)
        
actions = [1, 1, 2, 2, 1, 2]  # Replace with your favorite path.
play_game(actions)

'''
Nice, so we know how to get through the maze, but how do we teach that to the gumdrop? It's just some bytes in an android phone. 
It doesn't have our human insight.

We could give it our list of actions directly, but then it would be copying us and not really learning. 
This was a tricky one to the mathematicians and computer scientists originally trying to solve this problem.
How do we teach a machine to do this without human insight?
'''
'''
Value Iteration
Let's turn the clock back on our time machines to 1957 to meet Mr. Richard Bellman. 
Bellman started his academic career in mathematics, but due to World War II, left his postgraduate studies at John Hopkins to teach electronics as part of the 
war effort (as chronicled by J. J. O'Connor and E. F. Robertson here). When the war was over, and it came time for him to focus on his next area of research, 
he became fascinated with Dynamic Programming: the idea of breaking a problem down into sub-problems and using recursion to solve the larger problem.

Eventually, his research landed him on Markov Decision Processes. 
These processes are a graphical way of representing how to make a decision based on a current state. 
States are connected to other states with positive and negative rewards that can be picked up along the way.

Sound familiar at all? Perhaps our Frozen Lake?

In the lake case, each cell is a state. The Hs and the G are a special type of state called a "Terminal State", meaning they can be entered, 
but they have no leaving connections. What of rewards? Let's say the value of losing our life is the negative opposite of getting to the goal and staying alive.
Thus, we can assign the reward of entering a death hole as -1, and the reward of escaping as +1.

Bellman's first breakthrough with this type of problem is now known as Value Iteration (his original paper). 
He introduced a variable, gamma (γ), to represent discounted future rewards. He also introduced a function of policy (π) that takes a state (s), 
and outputs corresponding suggested action (a). The goal is to find the value of a state (V), given the rewards that occur when following an action 
in a particular state (R).

Gamma, the discount, is the key ingredient here. If my time steps were in days, and my gamma was .9, 
$100 would be worth $100 to me today, $90 tomorrow, $81 the day after, and so on. Putting this all together, we get the Bellman Equation

source: Wikipedia

In other words, the value of our current state, current_values, is equal to the discount times the value of the next state, next_values, 
given the policy the agent will follow. For now, we'll have our agent assume a greedy policy: it will move towards the state with the highest calculated value. 
If you're wondering what P is, don't worry, we'll get to that later.

Let's program it out and see it in action! We'll set up an array representing the lake with -1 as the holes, and 1 as the goal. 
Then, we'll set up an array of zeros to start our iteration.
'''

LAKE = np.array([[0,  0,  0,  0],
                 [0, -1,  0, -1],
                 [0,  0,  0, -1],
                 [-1, 0,  0,  1]])
LAKE_WIDTH = len(LAKE[0])
LAKE_HEIGHT = len(LAKE)

DISCOUNT = .9  # Change me to be a value between 0 and 1.
current_values = np.zeros_like(LAKE)

'''
The Gym environment class has a handy property for finding the number of states in an environment called observation_space. 
In our case, there a 16 integer states, so it will label it as "Discrete". Similarly, action_space will tell us how many actions are available to the agent.

Let's take advantage of these to make our code portable between different lakes sizes.
'''
print("env.observation_space -", env.observation_space)
print("env.observation_space.n -", env.observation_space.n)
print("env.action_space -", env.action_space)
print("env.action_space.n -", env.action_space.n)

STATE_SPACE = env.observation_space.n
ACTION_SPACE = env.action_space.n
STATE_RANGE = range(STATE_SPACE)
ACTION_RANGE = range(ACTION_SPACE)

'''
We'll need some sort of function to figure out what the best neighboring cell is. 
The below function take's a cell of the lake, and looks at the current value mapping (to be called with current_values,
and see's what the value of the adjacent state is corresponding to the given action.
'''
def get_neighbor_value(state_x, state_y, values, action):
    """Returns the value of a state's neighbor.
    
    Args:
        state_x (int): The state's horizontal position, 0 is the lake's left.
        state_y (int): The state's vertical position, 0 is the lake's top.
        values (float array): The current iteration's state values.
        policy (int): Which action to check the value for.
        
    Returns:
        The corresponding action's value.
    """
    left = [state_y, state_x-1]
    down = [state_y+1, state_x]
    right = [state_y, state_x+1]
    up = [state_y-1, state_x]
    actions = [left, down, right, up]
    
    direction = actions[action]
    check_x = direction[1]
    check_y = direction[0]
        
    is_boulder = check_y < 0 or check_y >= LAKE_HEIGHT \
        or check_x < 0 or check_x >= LAKE_WIDTH
    
    value = values[state_y, state_x]
    if not is_boulder:
        value = values[check_y, check_x]
        
    return value
  '''
  But this doesn't find the best action, and the gumdrop is going to need that if it wants to greedily get off the lake. 
  The get_max_neighbor function we've defined below takes a number corresponding to a cell as state_number and the same value mapping as get_neighbor_value.
  '''
  def get_state_coordinates(state_number):
    state_x = state_number % LAKE_WIDTH
    state_y = state_number // LAKE_HEIGHT
    return state_x, state_y

def get_max_neighbor(state_number, values):
    """Finds the maximum valued neighbor for a given state.
    
    Args:
        state_number (int): the state to find the max neighbor for
        state_values (float array): the respective value of each state for
            each cell of the lake.
    
    Returns:
        max_value (float): the value of the maximum neighbor.
        policy (int): the action to take to move towards the maximum neighbor.
    """
    state_x, state_y = get_state_coordinates(state_number)
    # No policy or best value yet
    best_policy = -1
    max_value = -np.inf

    # If the cell has something other than 0, it's a terminal state.
    if LAKE[state_y, state_x]:
        return LAKE[state_y, state_x], best_policy
    
    for action in ACTION_RANGE:
        neighbor_value = get_neighbor_value(state_x, state_y, values, action)
        if neighbor_value > max_value:
            max_value = neighbor_value
            best_policy = action
        
    return max_value, best_policy
  def iterate_value(current_values):
    """Finds the future state values for an array of current states.
    
    Args:
        current_values (int array): the value of current states.

    Returns:
        next_values (int array): The value of states based on future states.
        next_policies (int array): The recommended action to take in a state.
    """
    next_values = []
    next_policies = []

    for state in STATE_RANGE:
        value, policy = get_max_neighbor(state, current_values)
        next_values.append(value)
        next_policies.append(policy)
    
    next_values = np.array(next_values).reshape((LAKE_HEIGHT, LAKE_WIDTH))
    return next_values, next_policies

next_values, next_policies = iterate_value(current_values)
next_values
np.array(next_policies).reshape((LAKE_HEIGHT ,LAKE_WIDTH))
current_values = DISCOUNT * next_values
current_values
next_values, next_policies = iterate_value(current_values)
print("Value")
print(next_values)
print("Policy")
print(np.array(next_policies).reshape((4,4)))
current_values = DISCOUNT * next_values
def play_game(policy):
    state = env.reset()
    step = 0
    done = False

    while not done:
        action = policy[state]  # This line is new.
        state, _, done, _ = env.step(action)
        env.render()
        step += 1
        print_state(state, done)

play_game(next_policies)

env = gym.make('FrozenLake-v0', is_slippery=True)
state = env.reset()
env.render()

play_game(next_policies)
def find_future_values(current_values, current_policies):
    """Finds the next set of future values based on the current policy."""
    next_values = []

    for state in STATE_RANGE:
        current_policy = current_policies[state]
        state_x, state_y = get_state_coordinates(state)

        # If the cell has something other than 0, it's a terminal state.
        value = LAKE[state_y, state_x]
        if not value:
            value = get_neighbor_value(
                state_x, state_y, current_values, current_policy)
        next_values.append(value)

    return np.array(next_values).reshape((LAKE_HEIGHT, LAKE_WIDTH))
  
  def find_best_policy(next_values):
    """Finds the best policy given a value mapping."""
    next_policies = []
    for state in STATE_RANGE:
        state_x, state_y = get_state_coordinates(state)

        # No policy or best value yet
        max_value = -np.inf
        best_policy = -1

        if not LAKE[state_y, state_x]:
            for policy in ACTION_RANGE:
                neighbor_value = get_neighbor_value(
                    state_x, state_y, next_values, policy)
                if neighbor_value > max_value:
                    max_value = neighbor_value
                    best_policy = policy
                
        next_policies.append(best_policy)
    return next_policies
  
  def iterate_policy(current_values, current_policies):
    """Finds the future state values for an array of current states.
    
    Args:
        current_values (int array): the value of current states.
        current_policies (int array): a list where each cell is the recommended
            action for the state matching its index.

    Returns:
        next_values (int array): The value of states based on future states.
        next_policies (int array): The recommended action to take in a state.
    """
    next_values = find_future_values(current_values, current_policies)
    next_policies = find_best_policy(next_values)
    return next_values, next_policies
  
  def get_locations(state_x, state_y, policy):
    left = [state_y, state_x-1]
    down = [state_y+1, state_x]
    right = [state_y, state_x+1]
    up = [state_y-1, state_x]
    directions = [left, down, right, up]
    num_actions = len(directions)

    gumdrop_right = (policy - 1) % num_actions
    gumdrop_left = (policy + 1) % num_actions
    locations = [gumdrop_left, policy, gumdrop_right]
    return [directions[location] for location in locations]
  
  def get_neighbor_value(state_x, state_y, values, policy):
    """Returns the value of a state's neighbor.
    
    Args:
        state_x (int): The state's horizontal position, 0 is the lake's left.
        state_y (int): The state's vertical position, 0 is the lake's top.
        values (float array): The current iteration's state values.
        policy (int): Which action to check the value for.
        
    Returns:
        The corresponding action's value.
    """
    locations = get_locations(state_x, state_y, policy)
    location_chance = 1.0 / len(locations)
    total_value = 0

    for location in locations:
        check_x = location[1]
        check_y = location[0]

        is_boulder = check_y < 0 or check_y >= LAKE_HEIGHT \
            or check_x < 0 or check_x >= LAKE_WIDTH
    
        value = values[state_y, state_x]
        if not is_boulder:
            value = values[check_y, check_x]
        total_value += location_chance * value
    return total_value
  
  current_values = np.zeros_like(LAKE)
policies = np.random.choice(ACTION_RANGE, size=STATE_SPACE)
np.array(policies).reshape((4,4))

next_values, policies = iterate_policy(current_values, policies)
print("Value")
print(next_values)
print("Policy")
print(np.array(policies).reshape((4,4)))
current_values = DISCOUNT * next_values


play_game(policies)

new_row = np.zeros((1, env.action_space.n))
q_table = np.copy(new_row)
q_map = {0: 0}

def print_q(q_table, q_map):
    print("mapping")
    print(q_map)
    print("q_table")
    print(q_table)

print_q(q_table, q_map)

def get_action(q_map, q_table, state_row, random_rate):
    """Find max-valued actions and randomly select from them."""
    if random.random() < random_rate:
        return random.randint(0, ACTION_SPACE-1)

    action_values = q_table[state_row]
    max_indexes = np.argwhere(action_values == action_values.max())
    max_indexes = np.squeeze(max_indexes, axis=-1)
    action = np.random.choice(max_indexes)
    return action
  
  def update_q(q_table, new_state_row, reward, old_value):
    """Returns an updated Q-value based on the Bellman Equation."""
    learning_rate = .1  # Change to be between 0 and 1.
    future_value = reward + DISCOUNT * np.max(q_table[new_state_row])
    return old_value + learning_rate * (future_value - old_value)
  
  def play_game(q_table, q_map, random_rate, render=False):
    state = env.reset()
    step = 0
    done = False

    while not done:
        state_row = q_map[state]
        action = get_action(q_map, q_table, state_row, random_rate)
        new_state, _, done, _ = env.step(action)

        #Add new state to table and mapping if it isn't there already.
        if new_state not in q_map:
            q_map[new_state] = len(q_table)
            q_table = np.append(q_table, new_row, axis=0)
        new_state_row = q_map[new_state]

        reward = -.01  #Encourage exploration.
        if done:
            reward = 1 if new_state == 15 else -1
        current_q = q_table[state_row, action]
        q_table[state_row, action] = update_q(
            q_table, new_state_row, reward, current_q)

        step += 1
        if render:
            env.render()
            print_state(new_state, done)
        state = new_state
        
        return new_state
      # Run to refresh the q_table.
random_rate = 1
q_table = np.copy(new_row)
q_map = {0: 0}

q_table, q_map = play_game(q_table, q_map, random_rate, render=True)
print_q(q_table, q_map)


for _ in range(1000):
    q_table, q_map = play_game(q_table, q_map, random_rate)
    random_rate = random_rate * .99
print_q(q_table, q_map)
random_rate


q_table, q_map = play_game(q_table, q_map, 0, render=True)

'''

Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations
under the License.

'''
