from itertools import combinations
from circuitree import CircuitGrammar, CircuiTree
from circuitree.models import SimpleNetworkGrammar
from functools import partial
from typing import Any, Container, Generator, Iterable, Mapping, Optional
from more_itertools import powerset
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from tqdm import tqdm


def make_complexity_graph(
    G: nx.DiGraph, grammar: CircuitGrammar, from_circuits: Iterable[Any]
) -> nx.DiGraph:
    """Given a search graph G and a root node, return the complexity graph.

    The complexity graph contains only the terminal nodes of G that are "successful"
    (i.e. have a mean reward above the cutoff). Nodes are connected if they differ
    by one action.

    This is implemented by first creating a subgraph of G containing only the
    successful terminal nodes. In that subgraph, each edge between two non-terminal nodes
    represents an edge in the complexity graph. Then for each edge between two
    """
    successful_circuits = set(from_circuits)

    # Create the complexity graph as a subgraph of G. Then for each edge between two
    # non-terminal nodes, add an edge between the corresponding terminal nodes.
    terminal_edges = {n: next(G.predecessors(n)) for n in successful_circuits}
    succesful_predecessors = set(terminal_edges.values())
    complexity_graph: nx.DiGraph = G.subgraph(
        succesful_predecessors | successful_circuits
    ).copy()
    edges_to_add = []
    for terminal_1, nonterminal_1 in terminal_edges.items():
        for node in complexity_graph.successors(nonterminal_1):
            if node == terminal_1:
                continue
            nonterminal_2 = node
            terminal_2 = grammar.do_action(nonterminal_2, "*terminate*")
            edges_to_add.append((terminal_1, terminal_2))

    for terminal_1, terminal_2 in edges_to_add:
        nonterminal_1 = terminal_edges[terminal_1]
        nonterminal_2 = terminal_edges[terminal_2]
        complexity_graph.add_edge(
            terminal_1, terminal_2, **G.edges[nonterminal_1, nonterminal_2]
        )
        complexity_graph.remove_edge(nonterminal_1, nonterminal_2)

    # Finally, remove nonterminal nodes
    complexity_graph.remove_nodes_from(succesful_predecessors)

    return complexity_graph


def _get_mean_reward(G: nx.DiGraph, n: Any) -> float:
    return G.nodes[n]["reward"] / max(1, G.nodes[n]["visits"])


def get_successful_states(G: nx.DiGraph, grammar: CircuitGrammar, cutoff: float):
    for n in G.nodes:
        if grammar.is_terminal(n):
            m = _get_mean_reward(G, n)
            if m >= cutoff:
                yield n


def compute_Q_tilde(
    G: nx.DiGraph, root_node: Any, inplace: bool = True, in_edges: bool = True
) -> nx.DiGraph:
    if inplace:
        graph = G
    else:
        graph = G.copy()

    # Iterate over nodes in DFS post-order
    for v in nx.dfs_postorder_nodes(graph, root_node):
        # Terminal node
        if graph.out_degree(v) == 0:
            Q_tilde = _get_mean_reward(graph, v)
        # Non-terminal node
        else:
            Q_tilde = np.mean([graph.nodes[w]["Q_tilde"] for w in graph.successors(v)])
        graph.nodes[v]["Q_tilde"] = Q_tilde
        if in_edges:
            for u in graph.predecessors(v):
                graph.edges[u, v]["Q_tilde"] = Q_tilde

    if not inplace:
        return graph


def compute_Q_tilde_loss(G, root, inplace=True):
    if inplace:
        graph = G
    else:
        graph = G.copy()
    compute_Q_tilde(graph, root, inplace=True, in_edges=True)
    for e in graph.edges:
        graph.edges[e]["Q_tilde_loss"] = 1 - G.edges[e]["Q_tilde"]


def get_minimum_spanning_arborescence(
    tree: Optional[CircuiTree] = None,
    graph: Optional[nx.DiGraph] = None,
    root: Optional[Any] = None,
    weight: str = "Q_tilde_loss",
) -> nx.DiGraph:
    G = tree.graph if tree is not None else graph
    root = tree.root if tree is not None else root

    if weight == "Q_tilde_loss" and weight not in G.edges[next(iter(G.edges))]:
        compute_Q_tilde_loss(G, root, inplace=True)

    # Compute the complexity tree as the minimum spanning tree of G
    msa: nx.DiGraph = nx.minimum_spanning_arborescence(
        G, attr=weight, preserve_attrs=True
    )

    # Copy node attributes from G
    nx.set_node_attributes(msa, G.nodes)

    return msa


def prune_branches_inplace(
    tree: nx.DiGraph,
    root_node: Any,
    keep_leaves: set[Any],
) -> set[Any]:
    """Given a rooted directed tree (arborescence) and a set of leaves to keep, remove all
    branches that do not lead to a good leaf."""
    # Recursively remove nodes that do not lead to a successful terminal node
    good_leaves = set(keep_leaves) | {root_node}
    bad_leaves = set(n for n in tree.nodes if tree.out_degree(n) == 0) - good_leaves
    while bad_leaves:
        tree.remove_nodes_from(bad_leaves)
        bad_leaves = set(n for n in tree.nodes if tree.out_degree(n) == 0) - good_leaves


def simplenetwork_n_interactions(genotype_or_pattern: str) -> int:
    """Returns the number of interactions in a SimpleNetworkGrammar genotype."""
    if "::" in genotype_or_pattern:
        pattern = genotype_or_pattern.split("::")[1]
    else:
        pattern = genotype_or_pattern

    if pattern:
        return genotype_or_pattern.count("_") + 1
    else:
        return 0


def simplenetwork_complexity_layout(
    complexity_graph: nx.DiGraph, grammar: SimpleNetworkGrammar
) -> dict[str, tuple[float, float]]:
    """Returns a layout for the complexity graph of a search.
    Accounts for multiple connected components by computing the number of interactions
    in each circuit."""

    if complexity_graph.number_of_edges() == 0:
        raise ValueError("The complexity graph has no edges.")

    pos: dict[str, tuple[float, float]] = graphviz_layout(complexity_graph, prog="dot")

    ### Deal with multiple connected components (clusters of circuits)
    # Find connected components
    components = list(nx.weakly_connected_components(complexity_graph))

    # Infer the y spacing between components
    edge0 = next(iter(complexity_graph.edges))
    dy = pos[edge0[0]][1] - pos[edge0[1]][1]

    # For each connected component, fix y values based on circuit complexity
    components = sorted(components, key=len, reverse=True)
    smallest_circuits = [min(c, key=len) for c in components]
    min_depth_per_component = np.array(
        [simplenetwork_n_interactions(n) for n in smallest_circuits]
    )
    depth_bias = dy * (min_depth_per_component - min_depth_per_component.min())
    for component, ybias in zip(components, depth_bias):
        for node in component:
            x, y = pos[node]
            pos[node] = (x, y - ybias)

    # Find the largest layer in any component and get the average x distance
    nodes, xyvals = zip(*pos.items())
    xvals, yvals = np.array(xyvals).T
    unique_yvals = np.unique(yvals)
    largest_layer_xvals = np.array([])
    for component in components:
        cmask = np.isin(nodes, list(component))
        for yval in unique_yvals:
            mask = cmask & np.isclose(yvals, yval)
            if mask.sum() > largest_layer_xvals.size:
                largest_layer_xvals = xvals[mask]
    dx = np.mean(np.diff(np.sort(largest_layer_xvals)))

    # Make sure components are not overlapping in the x dimension
    prev_x_max = xvals.min()
    for i, component in enumerate(components):
        cmask = np.isin(nodes, list(component))
        x_shift = prev_x_max - xvals[cmask].min() + 2 * dx
        xvals[cmask] += x_shift
        prev_x_max = xvals[cmask].max()
    pos = dict(zip(nodes, zip(xvals, yvals)))
    return pos


def simplenetwork_adjacency_matrix(
    state: str,
    grammar: SimpleNetworkGrammar,
    directed: bool = True,
    signed: bool = True,
) -> np.ndarray:
    """Returns an adjacency matrix representation of a SimpleNetwork circuit.
    Excludes self-loops."""

    # Get the interactions
    components, activations, inhibitions = grammar.parse_genotype(state)

    # Make a symmetric, unweighted adjacency matrix
    n_components = len(components)
    adjacency = np.zeros((n_components, n_components), dtype=int)
    for i, j in activations.tolist():
        if i != j:
            adjacency[i, j] = 1
            if not directed:
                adjacency[j, i] = 1

    inh_weight = -1 if signed else 1
    for i, j in inhibitions.tolist():
        if i != j:
            adjacency[i, j] = inh_weight
            if not directed:
                adjacency[j, i] = inh_weight
    return adjacency


def simplenetwork_as_undirected_graph(
    state: str, grammar: SimpleNetworkGrammar
) -> nx.DiGraph:
    """Returns an undirected graph representation of a SimpleNetwork circuit.
    Excludes self-loops."""

    adjacency = simplenetwork_adjacency_matrix(
        state, grammar, directed=False, signed=False
    )
    graph = nx.from_numpy_array(adjacency, create_using=nx.Graph)

    # Remove nodes with no connections
    graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph


def simplenetwork_negative_feedback_loops(genotype: str, grammar: SimpleNetworkGrammar):
    components, activations, inhibitions = grammar.parse_genotype(genotype)

    # Make a directed graph of the interactions, storing the +/- weights
    circuit = nx.DiGraph()
    circuit.add_nodes_from(components)
    for i, j in activations:
        if i != j:
            circuit.add_edge(i, j, weight=0)
    for i, j in inhibitions:
        if i != j:
            circuit.add_edge(i, j, weight=1)

    # Find all negative feedback loops by finding all cycles in the graph
    # and then checking if they are negative feedback loops
    edge_weights = nx.get_edge_attributes(circuit, "weight")
    negative_feedback_loops = []
    for cycle in nx.simple_cycles(circuit):
        edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        n_repressions = sum(edge_weights[e] for e in edges)

        # NFL will have an odd number of repressions
        if n_repressions % 2 == 1:
            negative_feedback_loops.append((n_repressions, cycle))

    return negative_feedback_loops


def partition_cheeger_contstant(G: nx.Graph, partition: set[Any]) -> float:
    return nx.algorithms.cuts.conductance(G, partition, weight=None)


def all_minor_subgraphs(G: nx.Graph) -> Generator[set[Any], None, None]:
    """Returns a generator that iterates through all subgraphs {S} of the
    connected graph G that are of size |S| <= |G| / 2"""
    if not nx.is_connected(G):
        raise ValueError("G must be connected.")
    if isinstance(G, nx.DiGraph):
        raise ValueError("G must be undirected.")

    nodes = list(G.nodes)
    max_S_vol = len(nodes) // 2
    node_combinations = powerset(nodes)
    _ = next(node_combinations)  # Skip the empty set
    for S in node_combinations:
        if len(S) <= max_S_vol:
            yield S


def cheeger_constant(G: nx.Graph) -> float:
    """Finds the conductance of the graph G by iterating through all possible cuts
    and returning the minimum conductance."""
    return min(partition_cheeger_contstant(G, S) for S in all_minor_subgraphs(G))


def simplenetwork_cheeger_constant(
    circuit: str, grammar: SimpleNetworkGrammar
) -> float:
    """Finds the conductance of the graph that describes the given circuit."""
    G = simplenetwork_as_undirected_graph(circuit, grammar)
    return cheeger_constant(G)


##########


def _edge_reward_loss(node1, node2, edge_attrs: Mapping) -> float:
    return 1 - edge_attrs["reward"] / max(1, edge_attrs["visits"])


def get_shortest_paths_from_root(
    G: nx.DiGraph,
    root: Any,
    to_states: Optional[Iterable[Any]] = None,
) -> Mapping[Any, list[Any]]:
    best_paths = {}
    for state in to_states:
        best_paths[state] = nx.shortest_path(
            G, source=root, target=state, weight=_edge_reward_loss
        )
    return best_paths


from scipy.stats import beta
from scipy.stats._continuous_distns import beta_gen
from scipy.special import logsumexp


class BetaMixtureModel:
    def __init__(
        self,
        rewards: Iterable[int | float],
        visits: Iterable[int | float],
        weights: Optional[Iterable[float]] = None,
    ):
        self.distributions = [Q_posterior(r, v) for r, v in zip(rewards, visits)]
        self.n_distributions = len(self.distributions)
        self.weights: np.ndarray = np.array(weights) or np.ones(self.n_distributions)
        self.weights /= self.weights.sum()

    def pdf(self, x):
        densities = np.array([dist.pdf(x) for dist in self.distributions])
        return np.dot(self.weights, densities)

    def cdf(self, x):
        cdfs = np.array([dist.cdf(x) for dist in self.distributions])
        return np.dot(self.weights, cdfs)

    def sf(self, x):
        sfs = np.array([dist.sf(x) for dist in self.distributions])
        return np.dot(self.weights, sfs)

    def logpdf(self, x):
        logdensities = np.array(
            [dist.logpdf(x) for dist in self.distributions], dtype=np.float64
        )
        return logsumexp(logdensities, b=self.weights)

    def logcdf(self, x):
        logcdfs = np.array([dist.logcdf(x) for dist in self.distributions])
        return logsumexp(logcdfs, b=self.weights)

    def logsf(self, x):
        logsfs = np.array([dist.logsf(x) for dist in self.distributions])
        return logsumexp(logsfs, b=self.weights)


def Q_posterior(n, N):
    """Returns the posterior probability of success given n successes in N trials.
    Uses a uniform Beta prior with alpha=1, beta=1.
    """
    return beta(n + 1, N - n + 1)


def get_posterior(
    G: nx.DiGraph, grammar: CircuitGrammar, node: Any, flat: bool = True
) -> beta_gen | BetaMixtureModel:
    """Returs the posterior distribution of success probability for a given node in the
    search graph G. For terminal nodes, this posterior is computed directly from the
    observed rewards and visits. For non-terminal nodes, the posterior is computed from
    the mean posterior of all reachable terminal states.
    Uses a conjugate uniform Beta prior.
    """
    if grammar.is_terminal(node):
        return Q_posterior(G.nodes[node]["reward"], G.nodes[node]["visits"])
    if flat:
        reachable_data = (
            (G.nodes[n]["reward"], G.nodes[n]["visits"])
            for n in nx.descendants(G, node)
            if grammar.is_terminal(n)
        )
        reachable_rewards, reachable_visits = zip(*reachable_data)
        return BetaMixtureModel(reachable_rewards, reachable_visits)
    else:
        raise NotImplementedError("Hierarchical posteriors not supported.")


def make_complexity_tree_mst(
    G: nx.DiGraph,
    root_node: Any,
    grammar: CircuitGrammar,
    success_cutoff: float,
    n_best: Optional[int] = None,
    **kwargs,
) -> tuple[nx.DiGraph, set[Any]]:
    """Given a search graph G (a DAG) and a root node, return the complexity tree.

    The complexity tree is a type of minimum spanning tree. For each edge in the search
    graph we compute Q_tilde, the empirical probability of success from that
    state-action pair.

    The search graph's minimum spanning arborescence (the directed version of a tree) is
    then computed using the complementary probability Q_tilde_loss = 1 - Q_tilde as the
    edge weight. Finally, we delete any branches that do not lead to a successful outcome
    and return the resulting tree.

    Algorithm for Q_tilde
    ---------------------
    Given a directed acyclic graph G = (V, E)
    For node v in PostOrder(V)
       If OutDegree(v) == 0
          Set Q_tilde_v = Q(v)
       Else
          W = Children(v)
          Set Q_tilde(v) = Sum(Q_tilde(w), for w in W) / |W|
       For u in Predecessors(v)
          Set Q_tilde_loss(u, v) = 1 - Q_tilde(v)
       End for
    End for

    """

    # Compute Q_tilde and its loss for each edge
    compute_Q_tilde(G, root_node, inplace=True)
    for e in G.edges:
        G.edges[e]["Q_tilde_loss"] = 1 - G.edges[e]["Q_tilde"]

    # Compute the complexity tree as the minimum spanning tree of G
    complexity_tree: nx.DiGraph = nx.minimum_spanning_arborescence(
        G,
        attr="Q_tilde_loss",
        preserve_attrs=True,
    )

    # Copy node attributes from G
    nx.set_node_attributes(complexity_tree, G.nodes)

    if n_best is not None:
        best_states = ...  # TODO: get the n_best best states
        prune_branches_inplace(complexity_tree, root_node, best_states)
        return complexity_tree, best_states
    else:
        terminal_nodes = set(n for n in complexity_tree.nodes if grammar.is_terminal(n))
        return complexity_tree, terminal_nodes


def merge_search_graphs(
    graphs: list[nx.DiGraph],
    merge_node_attrs: Optional[Container] = None,
    merge_edge_attrs: Optional[Container] = None,
    progress: bool = False,
) -> nx.DiGraph:
    """Given a list of search graphs, return a single merged search graph.
    Node and edge attributes that have numerical values (int or float) such as visits and rewards
    are summed. All other attributes are copied from the first graph.
    All nodes are assumed to have the same attributes, as are all edges (nodes and edges
    may have different attributes).
    """
    graph: nx.DiGraph = graphs.pop(0)

    if not graphs:
        return graph  # Only one graph

    n0 = next(iter(graph.nodes))
    e0 = next(iter(graph.edges))
    node_attrs = set(
        attr for attr, val in graph.nodes[n0].items() if isinstance(val, (int, float))
    )
    edge_attrs = set(
        attr for attr, val in graph.edges[e0].items() if isinstance(val, (int, float))
    )

    if merge_node_attrs is not None:
        node_attrs = node_attrs & merge_node_attrs
    if merge_edge_attrs is not None:
        edge_attrs = edge_attrs & merge_edge_attrs

    iterator = graphs
    if progress:
        iterator = tqdm(graphs, desc="Merging graphs", total=len(graphs) + 1)
        iterator.update(1)
    for other_graph in iterator:
        # Add nodes/edges not in the graph and accumulate attributes
        add_nodes = []
        for n, attrs in other_graph.nodes(data=True):
            if n not in graph.nodes:
                add_nodes.append((n, {k: attrs[k] for k in node_attrs}))
            else:
                for a in node_attrs:
                    graph.nodes[n][a] = sum(
                        [graph.nodes[n][a], other_graph.nodes[n][a]]
                    )
        graph.add_nodes_from(add_nodes)

        add_edges = []
        for *e, attrs in other_graph.edges(data=True):
            if e not in graph.edges:
                add_edges.append((*e, {k: attrs[k] for k in edge_attrs}))
            else:
                for a in edge_attrs:
                    graph.edges[e][a] = sum(
                        [graph.edges[e][a], other_graph.edges[e][a]]
                    )
        graph.add_edges_from(add_edges)

    return graph


def n_connected_components_in_circuit(
    genotype: Any, grammar: SimpleNetworkGrammar
) -> int:
    """Returns the number of connected components in the circuit represented by the
    genotype. Ignores circuit components that have no connections.

    This is computed by first parsing the genotype into a circuit graph and
    then computing the number of connected components in the graph."""
    components, activations, inhibitions = grammar.parse_genotype(
        genotype, nonterminal_ok=True
    )

    # Root circuit is a special case
    if activations.size == 0 and inhibitions.size == 0:
        return 1

    # Make a symmetric, unweighted adjacency matrix
    n_circuit_components = len(components)
    adjacency = np.zeros((n_circuit_components, n_circuit_components), dtype=int)
    for i, j in activations:
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    for i, j in inhibitions:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    # Remove any circuit nodes with no connections
    connected_nodes = np.where(adjacency.sum(axis=1) != 0)[0]
    adjacency = adjacency[np.ix_(connected_nodes, connected_nodes)]

    # The number of connected components is the number of zero eigenvalues of the
    # laplacian matrix
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    eigvals = np.linalg.eigvals(laplacian)
    n_connected_components = np.isclose(eigvals, 0).sum()

    return n_connected_components


##########################################################################


###### Compare the posterior distributions between reachable and non-reachable sets
# For each terminal, compute the posterior distribution of the mean reward Q
# For each node, perform a test by sampling from the posterior:
#  1. Split terminals into disjoint sets of reachable and non-reachable, X and Y resp.
#  2. For each set, sample from the posterior M times
#      a. Choose a random terminal
#      b. Sample from its posterior distribution
#      c. Repeat M times
#  3. Compute the probability p = P(x < y), where x ~ Post(X) and y ~ Post(Y)
#      a. Sort both samples
#      b. Iterate in a clever way to count the occurrences of x < y
#      c. Divide by M to get the probability
#  4. If p < alpha, then the node is significant

from collections import Counter
from numba import njit


def bayesian_p_value_of_motif(
    G: nx.DiGraph,
    grammar: CircuitGrammar,
    state: Any,
    sample_size: int,
    U: Optional[set[Any]] = None,
):
    """Given a search graph G and a nonterminal state, return the p-value of that state
    being a successful motif.

    The p-value is computed by sampling repeatedly from the posterior distribution of
    Q, the mean reward. We take `sample_size` samples from this posterior,
    each time randomly choosing a circuit from the set ``X`` of terminal circuits
    containing this motif and sampling from its posterior distribution. We then do the
    same for the set ``Y = U \ X`` where ``U`` is the set of all terminal circuits.
    Finally, we use the posterior samples to compute the p-value as the probability

            p = P(Qx < Qy),

    where

            Qx ~ Posterior(Q | states X, search data) and
            Qy ~ Posterior(Q | states U \ X, search data).

    This p-value can then be compared to a significance level alpha (i.e. ``state`` is a
    motif if p < alpha).
    """
    # Get the union set of reachable and non-reachable terminals
    if U is None:
        U = set(n for n in G.nodes if grammar.is_terminal(n))

    # Get the sets to compare
    X, Y = split_terminals_by_reachability(G, state, U=U, grammar=grammar)
    X = list(X)
    Y = list(Y)

    # Sample from the posterior of each set
    X_samples = sample_from_posteriors(G, X, sample_size)
    Y_samples = sample_from_posteriors(G, Y, sample_size)

    # Compute the p-value
    p_value = compute_proportion_less_than(X_samples, Y_samples)

    return p_value


def split_terminals_by_reachability(
    G: nx.DiGraph,
    from_node: Any,
    U: Optional[set[Any]] = None,
    grammar: Optional[CircuitGrammar] = None,
) -> tuple[set[Any], set[Any]]:
    """Given a node in the search graph G, return the disjoint sets X and Y of reachable
    and non-reachable terminal states, respectively (X ^ Y = {emptyset}).

    Optionally specify the union set of reachable and non-reachable. If specified, then
    the returned sets will satisfy X | Y = U.
    """
    if U is None:
        U = set(n for n in G.nodes if grammar.is_terminal(n))

    X = set()
    Y = U.copy()
    for n in nx.dfs_preorder_nodes(G, from_node):
        if n in U:
            X.add(n)
            Y.remove(n)

    return X, Y


def sample_from_posteriors(
    G: nx.DiGraph, terminal_states: list[Any], sample_size: int
) -> np.ndarray:
    """Given a search graph G and a set of terminal states, sample from the posterior
    distribution of each state n_samples times.

    NOTE: The ordering of the samples in the output array is not guaranteed to be random
    """
    # Get the number of times each state will be sampled
    n_states = len(terminal_states)
    n_samples_per_state = Counter(
        np.random.random_integers(0, n_states - 1, size=sample_size)
    )
    samples = -np.ones(sample_size, dtype=float)
    j = 0
    for idx, n_state in n_samples_per_state.items():
        state = terminal_states[idx]
        reward = G.nodes[state]["reward"]
        visits = G.nodes[state]["visits"]
        posterior = beta(reward + 1, visits - reward + 1)
        samples[j : j + n_state] = posterior.rvs(n_state)
        j += n_state
    return samples


@njit
def compute_proportion_less_than(x, y) -> float:
    """Given two samples x and y of the same size, compute the probability that x < y."""
    size = len(x)
    if len(y) != size:
        raise ValueError("x and y must have the same size.")

    # Find the ordering of the samples
    order = np.concatenate((x, y)).argsort()
    from_x = np.zeros(2 * size, dtype=np.bool_)
    from_x[:size] = True
    from_x = from_x[order]

    # Use the ordering to compute the proportion of times x < y
    cumulative_xs = np.cumsum(from_x)
    return cumulative_xs[~from_x].sum() / (size * size)


# def make_complexity_tree_dijkstra(
#     G: nx.DiGraph,
#     root_node: Any,
#     grammar: Any,
#     success_cutoff: float,
#     success_func: Optional[Callable] = None,
#     weight_func: Optional[Callable] = None,
#     n_best: Optional[int] = None,
# ) -> tuple[nx.DiGraph, Mapping[Any, list[Any]]]:
#     """Given a search graph G (a DAG) and a root node, return the complexity tree.

#     Here we compute the shortest paths from the root node to each terminal node. The
#     union of these paths is the complexity tree. Edges are weighted by the "reward loss"
#     which is defined as 1 - mean reward per sample (visit) of the edge.

#     If n_best is specified, only the n_best terminal nodes with the highest mean reward
#     are considered.
#     """
#     # success_func(G, node) should return a float used to determine whether a node is
#     # successful or not. By default, this is the mean reward per sample (visit)
#     success_func = success_func or _get_mean_reward

#     # weight_func(node1, node2, edge_attrs) should return a float used to determine the
#     # weight of an edge when computing the shortest path from root to terminal nodes.
#     # By default, this is the "reward loss" which is defined as 1 - mean reward
#     weight_func = weight_func or _edge_reward_loss

#     # Get the terminal nodes of G that are successful
#     best_states = get_successful_states(
#         G, grammar, success_cutoff, success_func=success_func
#     )
#     if n_best is not None:
#         key = partial(success_func, G)
#         best_states = sorted(best_states, key=key, reverse=True)[:n_best]
#     successful_states = set(best_states)

#     # Compute longest paths from the root node to each terminal node
#     successful_paths = get_shortest_paths_from_root(
#         G, root_node, successful_states, weight_func=weight_func
#     )

#     # Combine into a single tree
#     complexity_tree = nx.DiGraph()
#     for path in successful_paths.values():
#         nx.add_path(complexity_tree, path)

#     # Keep attributes from G
#     nx.set_node_attributes(complexity_tree, G.nodes)
#     nx.set_edge_attributes(complexity_tree, G.edges)

#     return complexity_tree, successful_paths
