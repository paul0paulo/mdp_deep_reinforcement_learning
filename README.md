# Reinforcement Learning MDP Framework

An experimental reinforcement learning framework that simulates agent learning in stochastic Markov Decision Processes (MDPs). The system supports multiple exploration strategies, Monte Carlo value estimation, convergence detection, and visualization of learning performance over repeated episodes.

## Overview

This project implements an episodic reinforcement learning simulator where an agent interacts with a probabilistic environment defined by states, actions, transition probabilities, rewards, and action costs. The agent learns optimal policies by sampling episodes and updating empirical action-value estimates.

The framework is designed for experimentation with exploration strategies and policy convergence in stochastic decision environments.

## Features

- Monte Carlo reinforcement learning with empirical value estimation
- Multiple exploration strategies:
  - Custom adaptive strategy
  - Epsilon-greedy exploration
  - Softmax (Boltzmann) exploration
- Explicit Q-value tracking for each state-action pair
- Convergence detection based on reward stability
- Learning curve visualization using Matplotlib
- Flexible input format for defining custom MDP environments

## Exploration Strategies

The framework supports three action selection strategies:

1. **Custom Strategy**
   A normalized adaptive exploration approach that balances reward scaling and visit counts.

2. **Epsilon-Greedy**
   Chooses a random action with probability `epsilon`, otherwise selects the action with the highest estimated reward.

3. **Softmax (Boltzmann)**
   Samples actions probabilistically based on the exponentiated Q-values, giving higher probability to higher-valued actions.

## Installation

Install dependencies:

```bash
pip install matplotlib
```

Python 3.8+ is recommended.

## Usage

Run the program with:

```bash
python prog3.py input.txt [strategy]
```

Where:

- `input.txt` is the environment specification file
- `strategy` (optional) is one of:
  - `custom` (default)
  - `epsilon-greedy`
  - `softmax`

Example:

```bash
python prog3.py example.txt epsilon-greedy
```

## Input File Format

The input file defines the MDP environment.

### Line 1

```
nS nT nA nRounds printFreq M
```

| Parameter    | Description                                          |
|--------------|------------------------------------------------------|
| `nS`         | Number of non-terminal states                        |
| `nT`         | Number of terminal states                            |
| `nA`         | Number of actions                                    |
| `nRounds`    | Number of training episodes                          |
| `printFreq`  | Frequency of printing learning tables                |
| `M`          | Exploration scaling parameter (custom strategy only) |

### Line 2

Terminal rewards as space-separated pairs:

```
terminal_state reward terminal_state reward ...
```

### Line 3

Action costs as space-separated pairs:

```
action_index cost action_index cost ...
```

### Remaining Lines

Transition probabilities, one per state-action pair:

```
state:action next_state probability next_state probability ...
```

Probabilities are normalized automatically if they do not sum to 1.

## Learning Process

For each episode:

1. Start from a random non-terminal state
2. Select an action using the chosen exploration strategy
3. Transition probabilistically to the next state
4. Accumulate rewards minus action costs
5. On reaching a terminal state, update all visited state-action value estimates using the Monte Carlo return

The agent gradually shifts from exploration to exploitation as experience accumulates.

## Q-Value Estimation

The framework computes empirical action values using:

```
Q(s, a) = TotalReward(s, a) / Count(s, a)
```

Unvisited state-action pairs return `0.0` to encourage exploration of untried actions.

## Convergence Detection

Training stops early if the variance of the last 10 episode rewards falls below `1e-3`, indicating that the policy has stabilized. A message is printed when this occurs.

## Visualization

After training completes, the framework plots the learning curve:

- **X-axis:** Episode number
- **Y-axis:** Total reward per episode

This helps analyze convergence behavior and compare the effectiveness of different exploration strategies.

## Example Output

During execution, the program periodically prints:

- **Count table** — number of visits per state-action pair
- **Total reward table** — cumulative rewards per state-action pair
- **Best action per state** — the greedy action based on current estimates

After training, a Matplotlib window displays the reward progression over all episodes.

## Applications

This framework can be used to:

- Study reinforcement learning exploration strategies
- Analyze policy convergence in stochastic environments
- Experiment with different MDP configurations
- Visualize learning dynamics in episodic RL systems

## Future Improvements

Possible extensions include:

- Q-learning or TD-learning implementations
- Discount factor support for infinite-horizon problems
- Comparative analysis across multiple environment configurations
- Advanced convergence metrics and statistical evaluation
