"""
scenario_tree.py
================
Finite scenario tree representing a filtered probability space over [0, T]
with càdlàg piecewise-constant price processes.

The tree encodes a discrete-time filtration F_0 ⊆ F_1 ⊆ … ⊆ F_T generated
by the price process S.  Every node corresponds to a distinct information
state; terminal nodes (leaves) correspond to complete scenarios/paths.

Convention for the stochastic integral:
    (H·S)_t = Σ_{0 < u ≤ t} ⟨H_u, ΔS_u⟩
where H_u is F_{u−}-measurable (set at the *parent* node of the jump at u).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class ScenarioNode:
    """A single node in the scenario / filtration tree.

    Parameters
    ----------
    node_id:      Unique string identifier.
    time:         Calendar time of this node.
    state:        Discounted price vector S_t ∈ ℝ^d at this node.
    parent_id:    Identifier of the parent node (``None`` for the root).
    children_ids: Identifiers of all child nodes.
    cond_prob:    Conditional probability P(this node | parent node).
    info:         Auxiliary metadata (filtration labels, etc.).
    """

    node_id: str
    time: float
    state: np.ndarray
    parent_id: Optional[str]
    children_ids: List[str]
    cond_prob: float
    info: Dict = field(default_factory=dict)


@dataclass
class JumpEvent:
    """A single price jump recorded on a scenario path.

    Parameters
    ----------
    time:           Calendar time at which the jump occurs.
    pre_state:      Price vector just before the jump (S_{t−}).
    post_state:     Price vector just after the jump (S_t).
    delta_S:        Jump increment ΔS_t = S_t − S_{t−}.
    parent_node_id: Node ID of the pre-jump node.
    child_node_id:  Node ID of the post-jump node.
    """

    time: float
    pre_state: np.ndarray
    post_state: np.ndarray
    delta_S: np.ndarray
    parent_node_id: str
    child_node_id: str


# ---------------------------------------------------------------------------
# Scenario tree container
# ---------------------------------------------------------------------------

class ScenarioTree:
    """Container for a finite, non-recombining scenario tree.

    Attributes
    ----------
    nodes:   Mapping from node_id to :class:`ScenarioNode`.
    root_id: Identifier of the root node.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, ScenarioNode] = {}
        self.root_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: ScenarioNode) -> None:
        """Register *node* in the tree; sets root_id if node has no parent."""
        self.nodes[node.node_id] = node
        if node.parent_id is None:
            self.root_id = node.node_id

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def get_root(self) -> ScenarioNode:
        """Return the root node."""
        return self.nodes[self.root_id]

    def get_children(self, node_id: str) -> List[ScenarioNode]:
        """Return the list of child :class:`ScenarioNode` objects."""
        node = self.nodes[node_id]
        return [self.nodes[cid] for cid in node.children_ids]

    def get_terminal_nodes(self) -> List[ScenarioNode]:
        """Return all leaf nodes (nodes with no children)."""
        return [n for n in self.nodes.values() if not n.children_ids]

    # ------------------------------------------------------------------
    # Path queries
    # ------------------------------------------------------------------

    def get_node_path(self, terminal_id: str) -> List[ScenarioNode]:
        """Return the path from the root to *terminal_id* (inclusive)."""
        path: List[ScenarioNode] = []
        node = self.nodes[terminal_id]
        while node is not None:
            path.append(node)
            if node.parent_id is None:
                break
            node = self.nodes[node.parent_id]
        path.reverse()
        return path

    def get_scenario_prob(self, terminal_id: str) -> float:
        """Return joint scenario probability = product of cond_probs along path."""
        prob = 1.0
        node = self.nodes[terminal_id]
        while node.parent_id is not None:
            prob *= node.cond_prob
            node = self.nodes[node.parent_id]
        return prob

    def get_all_scenario_probs(self) -> Dict[str, float]:
        """Return mapping terminal_id → joint probability for every terminal."""
        return {n.node_id: self.get_scenario_prob(n.node_id)
                for n in self.get_terminal_nodes()}

    # ------------------------------------------------------------------
    # Jump structure
    # ------------------------------------------------------------------

    def get_jump_events(
        self, path: List[ScenarioNode]
    ) -> List[Tuple[ScenarioNode, ScenarioNode, np.ndarray]]:
        """Return a list of ``(parent, child, ΔS)`` tuples along *path*.

        Each tuple represents one price jump: from *parent* to *child* with
        increment ΔS = child.state − parent.state.
        """
        events: List[Tuple[ScenarioNode, ScenarioNode, np.ndarray]] = []
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            delta_S = child.state - parent.state
            events.append((parent, child, delta_S))
        return events

    def get_jump_times(self) -> List[float]:
        """Return sorted unique jump times across the whole tree.

        A jump time is the *child* time of any parent→child transition where
        the child's time strictly exceeds the parent's time.
        """
        times: Set[float] = set()
        for node in self.nodes.values():
            if node.parent_id is not None:
                parent = self.nodes[node.parent_id]
                if node.time > parent.time:
                    times.add(node.time)
        return sorted(times)

    def get_ancestors(self, node_id: str) -> List[ScenarioNode]:
        """Return path from root to *node_id* (inclusive)."""
        return self.get_node_path(node_id)

    def get_descendants(self, node_id: str) -> List[ScenarioNode]:
        """Return all terminal nodes reachable from *node_id*."""
        results: List[ScenarioNode] = []
        stack = [node_id]
        while stack:
            nid = stack.pop()
            node = self.nodes[nid]
            if not node.children_ids:
                results.append(node)
            else:
                stack.extend(node.children_ids)
        return results

    def state_space(self) -> Set[Tuple]:
        """Return the set of all unique price states visited in the tree."""
        return {tuple(n.state.tolist()) for n in self.nodes.values()}


# ---------------------------------------------------------------------------
# Predictable trading strategy
# ---------------------------------------------------------------------------

class PredictableStrategy:
    """A predictable portfolio process represented on the scenario tree.

    Holdings are indexed by the *parent* node ID: ``holdings[node_id]`` is
    the vector of asset positions decided at *node_id* and held through any
    subsequent jump (i.e., for all transitions from *node_id* to its children).

    This encodes F_{t−}-measurability: the holding before the jump at time t
    is determined by the information at the preceding node.
    """

    def __init__(self) -> None:
        self.holdings: Dict[str, np.ndarray] = {}

    def get_holding(self, node_id: str) -> np.ndarray:
        """Return holding vector associated with *node_id*.

        Raises ``KeyError`` if no holding has been set for this node.
        """
        return self.holdings[node_id]

    def set_holding(self, node_id: str, h: np.ndarray) -> None:
        """Set the holding vector for *node_id* to *h*."""
        self.holdings[node_id] = np.asarray(h, dtype=float)


# ---------------------------------------------------------------------------
# Gain process  (H · S)
# ---------------------------------------------------------------------------

class GainProcess:
    """Computes the stochastic integral (H·S) along paths of the tree.

    For a path (n_0, n_1, …, n_k) from root to a node:
        (H·S)_{t_i} = Σ_{j=0}^{i-1} ⟨H_{n_j}, ΔS_{n_{j+1}}⟩
    where ΔS_{n_{j+1}} = n_{j+1}.state − n_j.state.
    """

    @staticmethod
    def compute_gain_path(
        tree: ScenarioTree,
        path: List[ScenarioNode],
        strategy: PredictableStrategy,
    ) -> np.ndarray:
        """Return cumulative gains at each node along *path*.

        Parameters
        ----------
        tree:     The scenario tree (unused except for type checking).
        path:     Ordered list of nodes from root to some node.
        strategy: Predictable strategy supplying holdings at each parent node.

        Returns
        -------
        gains : np.ndarray of shape (len(path),)
            ``gains[i]`` is (H·S) at ``path[i].time``.
            ``gains[0] = 0`` by convention (no jumps before the root).
        """
        gains = np.zeros(len(path))
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            delta_S = child.state - parent.state
            # H is set at the parent node (predictable, F_{t−}-measurable)
            h = strategy.get_holding(parent.node_id)
            gains[i + 1] = gains[i] + float(np.dot(h, delta_S))
        return gains

    @staticmethod
    def compute_terminal_gains(
        tree: ScenarioTree,
        strategy: PredictableStrategy,
    ) -> Dict[str, float]:
        """Return terminal gain (H·S)_T for every terminal node.

        Returns
        -------
        Dict mapping terminal node_id → scalar gain.
        """
        result: Dict[str, float] = {}
        for terminal in tree.get_terminal_nodes():
            path = tree.get_node_path(terminal.node_id)
            gains = GainProcess.compute_gain_path(tree, path, strategy)
            result[terminal.node_id] = float(gains[-1])
        return result


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------

class AdmissibilityChecker:
    """Checks a-admissibility of trading strategies on a scenario tree.

    A strategy H is *a-admissible* if the wealth process
        W_t = a + (H·S)_t ≥ 0   for all t, P-a.s.
    (with tolerance ``tol`` for numerical purposes).
    """

    @staticmethod
    def check_admissible(
        tree: ScenarioTree,
        strategy: PredictableStrategy,
        a: float,
        tol: float = 1e-10,
    ) -> Tuple[bool, Dict[str, np.ndarray]]:
        """Test a-admissibility of *strategy* with initial wealth *a*.

        Parameters
        ----------
        tree:     Scenario tree.
        strategy: Predictable strategy to test.
        a:        Initial wealth / admissibility lower-bound.
        tol:      Numerical tolerance for the non-negativity check.

        Returns
        -------
        is_admissible : bool
            ``True`` if W_t = a + (H·S)_t ≥ −tol on every path, at every t.
        wealth_paths : Dict[str, np.ndarray]
            Mapping terminal_id → wealth process W_t along that path.
        """
        is_admissible = True
        wealth_paths: Dict[str, np.ndarray] = {}

        for terminal in tree.get_terminal_nodes():
            path = tree.get_node_path(terminal.node_id)
            gains = GainProcess.compute_gain_path(tree, path, strategy)
            wealth = a + gains  # W_t at each node along path
            wealth_paths[terminal.node_id] = wealth
            if np.any(wealth < -tol):
                is_admissible = False

        return is_admissible, wealth_paths

    @staticmethod
    def find_arbitrage(
        tree: ScenarioTree,
        strategies_list: List[PredictableStrategy],
        a_vals: List[float],
        tol: float = 1e-10,
    ) -> Optional[Tuple[int, float]]:
        """Search *strategies_list* for a first a-admissible arbitrage.

        An arbitrage is a strategy that is a-admissible AND whose terminal
        gain satisfies:
            (H·S)_T ≥ −tol   on all paths,
            (H·S)_T >   tol   on at least one path.

        Parameters
        ----------
        strategies_list: Candidate strategies.
        a_vals:          Corresponding initial wealth values (same length).
        tol:             Numerical tolerance.

        Returns
        -------
        ``(index, a_value)`` of the first arbitrage found, or ``None``.
        """
        for idx, (strategy, a) in enumerate(zip(strategies_list, a_vals)):
            admissible, wealth_paths = AdmissibilityChecker.check_admissible(
                tree, strategy, a, tol
            )
            if not admissible:
                continue
            terminal_gains = GainProcess.compute_terminal_gains(tree, strategy)
            gains_arr = np.array(list(terminal_gains.values()))
            if np.all(gains_arr >= -tol) and np.any(gains_arr > tol):
                return (idx, a)
        return None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_binary_tree_1d(
    n_steps: int,
    up_factor: float,
    down_factor: float,
    p_up: float,
    S0: float,
    T: float,
) -> ScenarioTree:
    """Build a non-recombining binomial tree for a single risky asset.

    The price at each node is determined by the sequence of up/down moves:
        S_{path} = S0 · up_factor^{#up} · down_factor^{#down}

    Parameters
    ----------
    n_steps:     Number of time steps.
    up_factor:   Multiplicative factor for an up-move (e.g. 1.1).
    down_factor: Multiplicative factor for a down-move (e.g. 0.9).
    p_up:        Conditional probability of an up-move at each step.
    S0:          Initial asset price.
    T:           Terminal time; step length = T / n_steps.

    Returns
    -------
    ScenarioTree with 2^{n_steps + 1} − 1 nodes.
    """
    tree = ScenarioTree()
    dt = T / n_steps

    # Recursive builder; each call creates the node for a given path.
    def _build(path_str: str, step: int, price: float, parent_id: Optional[str],
               cond_prob: float) -> None:
        node_id = path_str if path_str else "root"
        time = step * dt
        state = np.array([price])
        children_ids: List[str] = []

        if step < n_steps:
            up_id = node_id + "_u"
            dn_id = node_id + "_d"
            children_ids = [up_id, dn_id]

        node = ScenarioNode(
            node_id=node_id,
            time=time,
            state=state,
            parent_id=parent_id,
            children_ids=children_ids,
            cond_prob=cond_prob,
            info={"step": step},
        )
        tree.add_node(node)

        if step < n_steps:
            _build(up_id, step + 1, price * up_factor, node_id, p_up)
            _build(dn_id, step + 1, price * down_factor, node_id, 1.0 - p_up)

    _build("root", 0, S0, None, 1.0)
    return tree


def build_simple_2d_tree(
    S0: np.ndarray,
    jumps_per_asset: List[List[Tuple[float, float]]],
    probs: List[List[float]],
    T: float,
) -> ScenarioTree:
    """Build a multi-step scenario tree for two risky assets.

    At each step the joint price can move to several outcomes, each
    characterised by an additive jump ``(Δs1, Δs2)`` and a conditional
    probability.

    Parameters
    ----------
    S0:              Initial price vector, shape (2,).
    jumps_per_asset: Length-n_steps list; entry k is a list of (Δs1, Δs2)
                     tuples giving the possible joint increments at step k+1.
    probs:           Length-n_steps list; entry k is a list of probabilities
                     matching jumps_per_asset[k].  Must sum to 1.
    T:               Terminal time.

    Returns
    -------
    ScenarioTree with d=2 state vectors.
    """
    S0 = np.asarray(S0, dtype=float)
    n_steps = len(jumps_per_asset)
    dt = T / n_steps
    tree = ScenarioTree()

    def _build(node_id: str, step: int, state: np.ndarray,
               parent_id: Optional[str], cond_prob: float) -> None:
        time = step * dt
        children_ids: List[str] = []

        if step < n_steps:
            n_outcomes = len(jumps_per_asset[step])
            children_ids = [f"{node_id}_o{k}" for k in range(n_outcomes)]

        node = ScenarioNode(
            node_id=node_id,
            time=time,
            state=state.copy(),
            parent_id=parent_id,
            children_ids=children_ids,
            cond_prob=cond_prob,
            info={"step": step},
        )
        tree.add_node(node)

        if step < n_steps:
            for k, (jump, prob) in enumerate(
                zip(jumps_per_asset[step], probs[step])
            ):
                child_state = state + np.array(jump)
                _build(
                    f"{node_id}_o{k}",
                    step + 1,
                    child_state,
                    node_id,
                    prob,
                )

    _build("root", 0, S0, None, 1.0)
    return tree


def build_arbitrage_tree(S0: float, T: float) -> ScenarioTree:
    """Build a one-jump, d=1 tree where an arbitrage exists.

    Both possible jumps are strictly positive (ΔS > 0 on every branch), so
    the buy-and-hold strategy (H = 1) starting from wealth a = 0 is an
    admissible arbitrage:
        (H·S)_T = ΔS > 0   in every scenario.

    Parameters
    ----------
    S0: Initial asset price.
    T:  Terminal time.

    Returns
    -------
    ScenarioTree with root + 2 terminal nodes.
    """
    tree = ScenarioTree()
    state0 = np.array([float(S0)])

    # Root node
    root = ScenarioNode(
        node_id="root",
        time=0.0,
        state=state0,
        parent_id=None,
        children_ids=["child_high", "child_low"],
        cond_prob=1.0,
        info={"description": "arbitrage_tree_root"},
    )
    tree.add_node(root)

    # Child "high": ΔS = +2.0
    child_high = ScenarioNode(
        node_id="child_high",
        time=T,
        state=state0 + np.array([2.0]),
        parent_id="root",
        children_ids=[],
        cond_prob=0.5,
        info={"description": "high_branch"},
    )
    tree.add_node(child_high)

    # Child "low": ΔS = +1.0  (still strictly positive → clear arbitrage)
    child_low = ScenarioNode(
        node_id="child_low",
        time=T,
        state=state0 + np.array([1.0]),
        parent_id="root",
        children_ids=[],
        cond_prob=0.5,
        info={"description": "low_branch"},
    )
    tree.add_node(child_low)

    return tree
