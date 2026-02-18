import sys
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt

random.seed(0)

def read_input(path):

    with open(path, "r") as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    nS, nT, nA, nRounds, printFreq, M = raw[0].split()
    nS, nT, nA = int(nS), int(nT), int(nA)
    nRounds, printFreq, M = int(nRounds), int(printFreq), float(M)

    term_line = raw[1].split()
    terminal_rewards = {}
    for i in range(0, len(term_line), 2):
        s = int(term_line[i])
        r = float(term_line[i+1])
        terminal_rewards[s] = r

    cost_line = raw[2].split()
    action_cost = {}
    for i in range(0, len(cost_line), 2):
        a = int(cost_line[i])
        c = float(cost_line[i+1])
        action_cost[a] = c


    transitions = defaultdict(lambda: defaultdict(list))
    for ln in raw[3:]:
        parts = ln.split()
        lhs = parts[0]
        if ":" not in lhs:
            raise ValueError(f"Bad transition LHS: {lhs}")
        s_str, a_str = lhs.split(":")
        s = int(s_str)
        a = int(a_str)
        pairs = parts[1:]
        if len(pairs) % 2 != 0:
            raise ValueError(f"Bad transition RHS (needs pairs): {ln}")
        acc = []
        for i in range(0, len(pairs), 2):
            ns = int(pairs[i])
            p  = float(pairs[i+1])
            acc.append((ns, p))

        total_p = sum(p for _, p in acc)
        if total_p <= 0:
            raise ValueError(f"No probability mass for transition {s}:{a}")
        acc = [(ns, p/total_p) for ns, p in acc]
        transitions[s][a] = acc

    return {
        "nS": nS,
        "nT": nT,
        "nA": nA,
        "nRounds": nRounds,
        "printFreq": printFreq,
        "M": M,
        "terminal_rewards": terminal_rewards,
        "action_cost": action_cost,
        "transitions": transitions
    }


class Tables:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.Count = [[0 for _ in range(nA)] for __ in range(nS)]
        self.Total = [[0.0 for _ in range(nA)] for __ in range(nS)]

    def update_round(self, visited_pairs, round_reward):
        seen = set(visited_pairs)
        for (s, a) in seen:
            self.Count[s][a] += 1
            self.Total[s][a] += round_reward

    def q_value(self, s, a):
        """Compute Q(s, a) = Total[s][a] / Count[s][a].

        Returns 0.0 safely if Count[s][a] is zero (unvisited state-action pair).
        """
        if self.Count[s][a] == 0:
            return 0.0
        return self.Total[s][a] / self.Count[s][a]

    def best_actions(self):
        best = []
        for s in range(self.nS):
            if any(self.Count[s][a] == 0 for a in range(self.nA)):
                best.append("U")
                continue
            avgs = [self.Total[s][a]/self.Count[s][a] for a in range(self.nA)]
            a_star = max(range(self.nA), key=lambda a: avgs[a])
            best.append(str(a_star))
        return best

    def print_tables(self, round_idx, nS, nA):
        print(f"After {round_idx} rounds")
        print("Count:")
        for s in range(nS):
            line = []
            for a in range(nA):
                line.append(f"[{s},{a}]={self.Count[s][a]}.")
            print(" ".join(line))
        print("\nTotal:")
        for s in range(nS):
            line = []
            for a in range(nA):
                val = self.Total[s][a]
                txt = f"{val:.6g}" 
                line.append(f"[{s},{a}]={txt}.")
            print(" ".join(line))
        best = self.best_actions()
        print("\nBest action:", " ".join(f"{s}:{best[s]}." for s in range(nS)))
        print()


def choose_action(state, tables, terminal_rewards, nA, M, strategy="custom", epsilon=0.1):
    """Select an action for the given state using the specified exploration strategy.

    Parameters
    ----------
    state : int
        Current state index.
    tables : Tables
        Shared Count/Total tables.
    terminal_rewards : dict
        Mapping of terminal state -> reward (used by the custom strategy).
    nA : int
        Number of available actions.
    M : float
        Exploration parameter for the custom strategy.
    strategy : str
        One of "custom", "epsilon-greedy", or "softmax".
    epsilon : float
        Probability of random exploration for the epsilon-greedy strategy.

    Returns
    -------
    int
        Chosen action index.
    """
    if strategy == "epsilon-greedy":
        return _choose_epsilon_greedy(state, tables, nA, epsilon)
    elif strategy == "softmax":
        return _choose_softmax(state, tables, nA)
    else:
        # Default: original "custom" behaviour
        return _choose_custom(state, tables, terminal_rewards, nA, M)


def _choose_custom(state, tables, terminal_rewards, nA, M):
    """Original custom exploration strategy (unchanged)."""
    untried = [a for a in range(nA) if tables.Count[state][a] == 0]
    if untried:
        return random.choice(untried)

    avgs = [tables.Total[state][a] / tables.Count[state][a] for a in range(nA)]
    min_avg = min(avgs)

    if terminal_rewards:
        min_tr  = min(terminal_rewards.values())
        max_tr  = max(terminal_rewards.values())
    else:
        min_tr = min_avg
        max_tr = max(avgs)

    bottom = min(min_avg, min_tr)
    top    = max_tr

    if abs(top - bottom) < 1e-12:
        return random.randrange(nA)

    savg = [0.25 + 0.75 * ((avg - bottom) / (top - bottom)) for avg in avgs]
    c = sum(tables.Count[state])
    exp = c / M if M > 0 else 0.0
    ups = [pow(x, exp) for x in savg]
    tot = sum(ups)
    if tot <= 0:
        return random.randrange(nA)
    probs = [u / tot for u in ups]
    return random.choices(range(nA), weights=probs, k=1)[0]


def _choose_epsilon_greedy(state, tables, nA, epsilon):
    """Epsilon-greedy exploration strategy.

    With probability *epsilon* a uniformly random action is chosen;
    otherwise the action with the highest average reward is selected.
    Unvisited actions are treated as having an average reward of 0.0.
    """
    if random.random() < epsilon:
        return random.randrange(nA)
    avgs = [tables.q_value(state, a) for a in range(nA)]
    return max(range(nA), key=lambda a: avgs[a])


def _choose_softmax(state, tables, nA):
    """Softmax (Boltzmann) exploration strategy.

    Computes exp(Q(s,a)) for each action and samples proportionally,
    giving higher probability to actions with higher average rewards.
    """
    avgs = [tables.q_value(state, a) for a in range(nA)]
    # Subtract max for numerical stability before exponentiation
    max_avg = max(avgs)
    weights = [math.exp(avg - max_avg) for avg in avgs]
    return random.choices(range(nA), weights=weights, k=1)[0]


def sample_next(transitions, s, a):
    seq = transitions.get(s, {})
    lst = seq.get(a, [])
    if not lst:
        raise ValueError(f"No transitions defined for state {s}, action {a}")
    choices, weights = zip(*lst)
    return random.choices(choices, weights=weights, k=1)[0]


def run_round(nS, nT, nA, transitions, terminal_rewards, action_cost, tables, M,
              strategy="custom", epsilon=0.1):
    """Run a single episode and return the total reward collected.

    Parameters
    ----------
    strategy : str
        Exploration strategy forwarded to choose_action().
    epsilon : float
        Epsilon parameter forwarded to choose_action().
    """
    s = random.randrange(nS)
    visited = []
    total_reward = 0.0
    while True:
        a = choose_action(s, tables, terminal_rewards, nA, M,
                          strategy=strategy, epsilon=epsilon)
        visited.append((s, a))
        total_reward -= action_cost[a]
        ns = sample_next(transitions, s, a)
        if ns >= nS:
            tr = terminal_rewards.get(ns, 0.0)
            total_reward += tr
            break
        s = ns
    tables.update_round(visited, total_reward)
    return total_reward


def main():
    """Entry point.

    Usage
    -----
    python prog3.py input.txt [strategy]

    strategy : optional, one of "custom" (default), "epsilon-greedy", "softmax"
    """
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 prog3.py input.txt [strategy]")
        print("  strategy: custom (default) | epsilon-greedy | softmax")
        sys.exit(1)

    # CLI strategy selection (requirement 6)
    strategy = sys.argv[2] if len(sys.argv) == 3 else "custom"
    valid_strategies = {"custom", "epsilon-greedy", "softmax"}
    if strategy not in valid_strategies:
        print(f"Unknown strategy '{strategy}'. Choose from: {', '.join(sorted(valid_strategies))}")
        sys.exit(1)

    data = read_input(sys.argv[1])
    nS       = data["nS"]
    nT       = data["nT"]
    nA       = data["nA"]
    nRounds  = data["nRounds"]
    printFreq= data["printFreq"]
    M        = data["M"]
    terminal_rewards = data["terminal_rewards"]
    action_cost      = data["action_cost"]
    transitions      = data["transitions"]

    if not terminal_rewards:
        raise ValueError("No terminal rewards specified in input")

    for a in range(nA):
        if a not in action_cost:
            raise ValueError(f"Missing cost for action {a}")

    for s in range(nS):
        for a in range(nA):
            if a not in transitions[s] or not transitions[s][a]:
                raise ValueError(f"Missing or empty transition line for state {s}, action {a}")

    tables = Tables(nS, nA)

    # Reward logging (requirement 3)
    rewards = []

    for r in range(1, nRounds + 1):
        round_reward = run_round(nS, nT, nA, transitions, terminal_rewards,
                                 action_cost, tables, M,
                                 strategy=strategy)
        rewards.append(round_reward)

        # Convergence detection (requirement 5)
        if len(rewards) >= 10:
            recent = rewards[-10:]
            if max(recent) - min(recent) < 1e-3:
                print(f"Converged after {r} episodes (last-10 reward variance < 1e-3).")
                tables.print_tables(r, nS, nA)
                break

        if printFreq != 0 and (r % printFreq == 0):
            tables.print_tables(r, nS, nA)
    else:
        # Only reached when the loop completes without a break (no early convergence)
        if printFreq == 0 or (nRounds % printFreq != 0):
            tables.print_tables(nRounds, nS, nA)

    # Learning curve visualization (requirement 4) — plotted after training ends
    plt.figure()
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve: Reward per Episode")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()