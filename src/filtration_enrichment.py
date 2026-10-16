"""
filtration_enrichment.py
========================
"exp_6" objects: an enriched filtration that carries auxiliary signals
*between* price jumps.

Key idea
--------
In the standard model only price movements generate information.  In the
enriched setting an additional signal I_t ∈ {0, 1} is revealed at some
time τ_s < τ_jump before the next price jump.  A strategy that is measurable
w.r.t. the *full* enriched filtration can condition on I_{τ_jump−} and choose
a holding that depends on the forthcoming jump direction.  A strategy that
only observes past prices cannot make this distinction.

Tree layout (example)
---------------------
root (t=0, price=S0, signal=None)
  ├── sig_0  (t=τ_s, event='signal_arrival', signal=0, prob=0.5)
  │     └── jump_down  (t=τ_j, event='price_jump', price=S0+down, prob=1.0)  [terminal]
  └── sig_1  (t=τ_s, event='signal_arrival', signal=1, prob=0.5)
        └── jump_up    (t=τ_j, event='price_jump', price=S0+up,   prob=1.0)  [terminal]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# EventNode
# ---------------------------------------------------------------------------

@dataclass
class EventNode:
    """A node in the enriched filtration tree.

    Each node records either a *signal arrival* or a *price jump* (or the
    initial state at t=0).

    Parameters
    ----------
    node_id:         Unique string identifier.
    time:            Calendar time of this event.
    event_type:      ``'signal_arrival'`` or ``'price_jump'``.
    price_state:     Current discounted price vector at this node.
    signal_state:    Auxiliary signal value (int) if revealed; ``None`` if not.
    parent_id:       Identifier of the parent node (``None`` for root).
    children_ids:    Identifiers of child nodes.
    cond_prob:       Conditional probability given parent.
    filtration_info: Auxiliary metadata.
    """

    node_id: str
    time: float
    event_type: str          # 'signal_arrival' or 'price_jump'
    price_state: np.ndarray
    signal_state: Optional[int]
    parent_id: Optional[str]
    children_ids: List[str] = field(default_factory=list)
    cond_prob: float = 1.0
    filtration_info: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EnrichedFiltrationTree
# ---------------------------------------------------------------------------

class EnrichedFiltrationTree:
    """Scenario tree with both price-jump and signal-arrival events.

    Attributes
    ----------
    nodes:   Mapping node_id → :class:`EventNode`.
    root_id: Identifier of the root node.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, EventNode] = {}
        self.root_id: Optional[str] = None

    def add_node(self, node: EventNode) -> None:
        """Register *node*; sets root_id if node has no parent."""
        self.nodes[node.node_id] = node
        if node.parent_id is None:
            self.root_id = node.node_id

    # ------------------------------------------------------------------
    # Node queries
    # ------------------------------------------------------------------

    def get_terminal_nodes(self) -> List[EventNode]:
        """Return all leaf nodes (no children)."""
        return [n for n in self.nodes.values() if not n.children_ids]

    def get_price_jump_nodes(self) -> List[EventNode]:
        """Return all nodes of type ``'price_jump'``."""
        return [n for n in self.nodes.values() if n.event_type == "price_jump"]

    def get_signal_arrival_nodes(self) -> List[EventNode]:
        """Return all nodes of type ``'signal_arrival'``."""
        return [n for n in self.nodes.values() if n.event_type == "signal_arrival"]

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------

    def get_path(self, terminal_id: str) -> List[EventNode]:
        """Return the path from root to *terminal_id* (inclusive)."""
        path: List[EventNode] = []
        node = self.nodes[terminal_id]
        while node is not None:
            path.append(node)
            if node.parent_id is None:
                break
            node = self.nodes[node.parent_id]
        path.reverse()
        return path

    def get_scenario_prob(self, terminal_id: str) -> float:
        """Return joint probability = product of cond_probs along path."""
        prob = 1.0
        node = self.nodes[terminal_id]
        while node.parent_id is not None:
            prob *= node.cond_prob
            node = self.nodes[node.parent_id]
        return prob

    # ------------------------------------------------------------------
    # Signal / price history analysis
    # ------------------------------------------------------------------

    def find_paths_same_price_different_signal(
        self,
        jump_time: float,
    ) -> List[Tuple[str, str, int, int]]:
        """Find terminal-path pairs sharing the same price history but differing signal.

        Two terminal paths are "same price history up to *jump_time*−" if
        every price-jump node on both paths before *jump_time* has the same
        price state.

        Parameters
        ----------
        jump_time: Reference time separating "before" (signal may differ) from
                   "at" (price jump occurs).

        Returns
        -------
        List of tuples ``(terminal_id_1, terminal_id_2, signal_1, signal_2)``
        for all pairs that share price history but carry different signal values.
        """
        terminals = self.get_terminal_nodes()

        def _price_history_key(term_id: str) -> Tuple:
            """Tuple of price states along price-jump nodes strictly before jump_time."""
            path = self.get_path(term_id)
            return tuple(
                tuple(n.price_state.tolist())
                for n in path
                if n.event_type == "price_jump" and n.time < jump_time
            )

        def _signal_before_jump(term_id: str) -> Optional[int]:
            """Most recent signal_state seen before jump_time."""
            path = self.get_path(term_id)
            sig = None
            for n in path:
                if n.time >= jump_time:
                    break
                if n.signal_state is not None:
                    sig = n.signal_state
            return sig

        # Group terminals by price-history key
        groups: Dict[Tuple, List[str]] = {}
        for t in terminals:
            key = _price_history_key(t.node_id)
            groups.setdefault(key, []).append(t.node_id)

        results: List[Tuple[str, str, int, int]] = []
        for members in groups.values():
            # Find pairs within each group that differ in signal
            sigs = {tid: _signal_before_jump(tid) for tid in members}
            for i, tid1 in enumerate(members):
                for tid2 in members[i + 1:]:
                    s1, s2 = sigs[tid1], sigs[tid2]
                    if s1 is not None and s2 is not None and s1 != s2:
                        results.append((tid1, tid2, int(s1), int(s2)))

        return results


# ---------------------------------------------------------------------------
# FullFiltrationStrategy
# ---------------------------------------------------------------------------

class FullFiltrationStrategy:
    """Trading strategy that exploits the full enriched filtration.

    The holding before a price jump can depend on the signal I_{τ_jump−} that
    was revealed at the preceding signal-arrival node.

    Holdings are stored with a composite key ``"{node_id}_s{signal}"`` to
    allow different positions for different realised signals.
    """

    def __init__(self) -> None:
        self.holdings: Dict[str, np.ndarray] = {}

    def set_holding_by_signal(
        self,
        node_id: str,
        signal: int,
        holding: np.ndarray,
    ) -> None:
        """Store holding for the node identified by (*node_id*, *signal*) pair.

        Parameters
        ----------
        node_id: Identifier of the decision node (e.g. the node just before
                 the price jump whose preceding signal determines the holding).
        signal:  Realised signal value.
        holding: Portfolio vector H ∈ ℝ^d.
        """
        key = f"{node_id}_s{signal}"
        self.holdings[key] = np.asarray(holding, dtype=float)

    def get_holding(
        self,
        price_jump_node: EventNode,
        tree: EnrichedFiltrationTree,
    ) -> np.ndarray:
        """Retrieve the holding applicable at *price_jump_node*.

        Traces the ancestry of *price_jump_node* to find the most recently
        revealed signal, then looks up the stored holding for that
        (parent_id, signal) combination.

        Falls back to zeros if no holding has been registered.

        Parameters
        ----------
        price_jump_node: The node at which the price jump occurs.
        tree:            The enriched filtration tree.

        Returns
        -------
        Holding vector H ∈ ℝ^d.
        """
        d = price_jump_node.price_state.shape[0]
        default = np.zeros(d)

        # Walk ancestry backward to find the nearest signal_arrival node
        parent_id = price_jump_node.parent_id
        while parent_id is not None:
            parent = tree.nodes[parent_id]
            if parent.signal_state is not None:
                # Found the signaling ancestor; look up composite key
                sig = parent.signal_state
                # Try keyed by the price_jump_node itself
                key_jump = f"{price_jump_node.node_id}_s{sig}"
                if key_jump in self.holdings:
                    return self.holdings[key_jump]
                # Try keyed by the signaling ancestor
                key_parent = f"{parent_id}_s{sig}"
                if key_parent in self.holdings:
                    return self.holdings[key_parent]
            parent_id = parent.parent_id

        # Also try a plain node_id lookup (no-signal case)
        if price_jump_node.node_id in self.holdings:
            return self.holdings[price_jump_node.node_id]

        return default


# ---------------------------------------------------------------------------
# PriceHistoryOnlyStrategy
# ---------------------------------------------------------------------------

class PriceHistoryOnlyStrategy:
    """Trading strategy restricted to measurability w.r.t. past prices only.

    Holdings are indexed by the *tuple* of observed price states along the
    path (ignoring any auxiliary signal).  Two paths that have identical price
    histories but different signals must use the same holding.
    """

    def __init__(self) -> None:
        self.holdings: Dict[tuple, np.ndarray] = {}

    def set_holding(self, price_history: tuple, holding: np.ndarray) -> None:
        """Store holding for a given price-history key.

        Parameters
        ----------
        price_history: Hashable representation of the price trajectory,
                       e.g. tuple of (price,) tuples.
        holding:       Portfolio vector H ∈ ℝ^d.
        """
        self.holdings[price_history] = np.asarray(holding, dtype=float)

    def get_holding(self, price_history: tuple) -> np.ndarray:
        """Retrieve the holding for *price_history*, or zeros if unknown."""
        return self.holdings.get(price_history, np.zeros(1))


# ---------------------------------------------------------------------------
# Factory: example enriched filtration tree
# ---------------------------------------------------------------------------

def build_enriched_filtration_example(
    S0: float = 1.0,
    T: float = 2.0,
    signal_time: float = 0.5,
    jump_time: float = 1.0,
    up_move: float = 0.5,
    down_move: float = -0.3,
) -> EnrichedFiltrationTree:
    """Build a simple two-path enriched filtration example.

    Timeline
    --------
    t=0          : Root (initial state, no signal, price=S0).
    t=signal_time: Signal I ∈ {0, 1} is revealed (each with prob 0.5).
    t=jump_time  : Price jump: +up_move if I=1, +down_move if I=0.
                   Both paths terminate after the jump.

    This illustrates that the holding before the price jump can depend on the
    signal (full filtration) but NOT on the price history alone (both paths
    share the same price history S_0, …, S_{jump_time−} = S0).

    Parameters
    ----------
    S0:          Initial price.
    T:           Terminal time (unused in payoffs here but recorded).
    signal_time: Time at which the auxiliary signal is revealed.
    jump_time:   Time at which the price jumps.
    up_move:     Additive price increment when I=1.
    down_move:   Additive price increment when I=0.

    Returns
    -------
    :class:`EnrichedFiltrationTree` with 5 nodes total.
    """
    tree = EnrichedFiltrationTree()
    s0_arr = np.array([float(S0)])

    # Root
    root = EventNode(
        node_id="root",
        time=0.0,
        event_type="price_jump",      # initial state, treated as a price node
        price_state=s0_arr.copy(),
        signal_state=None,
        parent_id=None,
        children_ids=["sig_0", "sig_1"],
        cond_prob=1.0,
        filtration_info={"description": "initial state"},
    )
    tree.add_node(root)

    # Signal arrival nodes (signal=0 and signal=1)
    for sig_val, child_id, jump_delta in [
        (0, "jump_down", down_move),
        (1, "jump_up",   up_move),
    ]:
        sig_node = EventNode(
            node_id=f"sig_{sig_val}",
            time=signal_time,
            event_type="signal_arrival",
            price_state=s0_arr.copy(),   # price unchanged at signal time
            signal_state=sig_val,
            parent_id="root",
            children_ids=[child_id],
            cond_prob=0.5,               # each signal equally likely
            filtration_info={"signal": sig_val},
        )
        tree.add_node(sig_node)

        # Price jump terminal node
        jump_price = s0_arr + np.array([jump_delta])
        jump_node = EventNode(
            node_id=child_id,
            time=jump_time,
            event_type="price_jump",
            price_state=jump_price.copy(),
            signal_state=sig_val,        # signal known at and after jump
            parent_id=f"sig_{sig_val}",
            children_ids=[],             # terminal
            cond_prob=1.0,
            filtration_info={"delta_S": jump_delta, "signal": sig_val},
        )
        tree.add_node(jump_node)

    return tree


# ---------------------------------------------------------------------------
# Gain computation for enriched strategies
# ---------------------------------------------------------------------------

def compute_gains_enriched(
    tree: EnrichedFiltrationTree,
    strategy: FullFiltrationStrategy,
) -> Dict[str, float]:
    """Compute terminal gain (H·S)_T under a full-filtration strategy.

    Only price-jump transitions contribute to the stochastic integral.

    Parameters
    ----------
    tree:     Enriched filtration tree.
    strategy: Full-filtration strategy.

    Returns
    -------
    Dict mapping terminal node_id → scalar terminal gain.
    """
    result: Dict[str, float] = {}

    for terminal in tree.get_terminal_nodes():
        path = tree.get_path(terminal.node_id)
        gain = 0.0

        for i in range(len(path) - 1):
            child = path[i + 1]
            parent = path[i]
            # Only price-jump transitions carry a payoff
            if child.event_type != "price_jump":
                continue
            delta_S = child.price_state - parent.price_state
            if np.linalg.norm(delta_S) < 1e-15:
                continue
            # Holding is set at the parent (predictable), using signal info
            h = strategy.get_holding(child, tree)
            gain += float(np.dot(h, delta_S))

        result[terminal.node_id] = gain

    return result


def compare_strategy_outcomes(
    tree: EnrichedFiltrationTree,
    full_strategy: FullFiltrationStrategy,
    price_only_strategy: PriceHistoryOnlyStrategy,
) -> Dict:
    """Compare terminal gains of full vs price-only strategies.

    Parameters
    ----------
    tree:                 Enriched filtration tree.
    full_strategy:        Strategy using full filtration (including signals).
    price_only_strategy:  Strategy using only observed price history.

    Returns
    -------
    dict with keys:
        ``full_gains``       — Dict[str, float] terminal gains under full strategy.
        ``price_only_gains`` — Dict[str, float] terminal gains under price-only strategy.
        ``differences``      — Dict[str, float] full_gain − price_only_gain per terminal.
        ``max_difference``   — float, maximum absolute difference.
        ``strategies_differ``— bool, True if any terminal gain differs.
    """
    full_gains = compute_gains_enriched(tree, full_strategy)

    # Compute price-only strategy gains: reconstruct price history tuple per terminal
    price_only_gains: Dict[str, float] = {}
    for terminal in tree.get_terminal_nodes():
        path = tree.get_path(terminal.node_id)
        gain = 0.0

        # Build price-history tuple from price-jump nodes seen so far
        price_history: List[tuple] = []

        for i in range(len(path) - 1):
            child = path[i + 1]
            parent = path[i]
            if child.event_type != "price_jump":
                continue
            delta_S = child.price_state - parent.price_state
            if np.linalg.norm(delta_S) < 1e-15:
                continue
            # The price-only strategy indexes by the history of prices observed *before* this jump
            ph_key = tuple(tuple(n.price_state.tolist())
                           for n in path[: path.index(child)]
                           if n.event_type == "price_jump")
            h = price_only_strategy.get_holding(ph_key)
            gain += float(np.dot(h, delta_S))
            price_history.append(tuple(child.price_state.tolist()))

        price_only_gains[terminal.node_id] = gain

    differences: Dict[str, float] = {
        tid: full_gains.get(tid, 0.0) - price_only_gains.get(tid, 0.0)
        for tid in {t.node_id for t in tree.get_terminal_nodes()}
    }
    max_diff = float(max(abs(v) for v in differences.values())) if differences else 0.0

    return {
        "full_gains": full_gains,
        "price_only_gains": price_only_gains,
        "differences": differences,
        "max_difference": max_diff,
        "strategies_differ": max_diff > 1e-10,
    }
