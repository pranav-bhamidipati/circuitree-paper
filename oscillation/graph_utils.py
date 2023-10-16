from functools import partial
from typing import Any, Container, Iterable, Mapping, Optional
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from tqdm import tqdm


def _get_mean_reward(G: nx.DiGraph, n: Any) -> float:
    return G.nodes[n]["reward"] / max(1, G.nodes[n]["visits"])


def _edge_reward_loss(node1, node2, edge_attrs: Mapping) -> float:
    return 1 - edge_attrs["reward"] / max(1, edge_attrs["visits"])


def get_successful_states(G: nx.DiGraph, grammar: Any, cutoff: float):
    for n in G.nodes:
        if grammar.is_terminal(n):
            m = _get_mean_reward(G, n)
            if m >= cutoff:
                yield n


def complexity_layout(
    complexity_graph: nx.DiGraph,
) -> Mapping[Any, tuple]:
    pos = graphviz_layout(complexity_graph, prog="dot")
    return pos


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
    G: nx.DiGraph, grammar: Any, node: Any, flat: bool = True
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


def make_complexity_graph(G: nx.DiGraph, grammar: Any, cutoff: float) -> nx.DiGraph:
    """Given a search graph G and a root node, return the complexity graph.

    The complexity graph contains only the terminal nodes of G that are "successful"
    (i.e. have a mean reward above the cutoff). Nodes are connected if they differ
    by one action.

    This is implemented by first creating a subgraph of G containing only the
    successful terminal nodes. In that subgraph, each edge between two non-terminal nodes
    represents an edge in the complexity graph. Then for each edge between two
    """
    # Get the terminal nodes of G that are successful
    successful_states = set(get_successful_states(G, grammar, cutoff))

    # Create the complexity graph as a subgraph of G. Then for each edge between two
    # non-terminal nodes, add an edge between the corresponding terminal nodes.
    terminal_edges = {n: next(G.predecessors(n)) for n in successful_states}
    succesful_predecessors = set(terminal_edges.values())
    complexity_graph: nx.DiGraph = G.subgraph(
        succesful_predecessors | successful_states
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


def prune_complexity_tree(
    tree: nx.DiGraph, root_node: Any, grammar: Any, n_best: int, success_cutoff: float
) -> set[Any]:
    # Get the terminal nodes that are successful
    key = partial(_get_mean_reward, tree)
    best_states = get_successful_states(tree, grammar, success_cutoff)
    best_states = sorted(best_states, key=key, reverse=True)[:n_best]
    best_states = set(best_states)

    # Recursively remove nodes that do not lead to a successful terminal node
    good_leaves = best_states | {root_node}
    bad_leaves = set(n for n in tree.nodes if tree.out_degree(n) == 0) - good_leaves
    while bad_leaves:
        tree.remove_nodes_from(bad_leaves)
        bad_leaves = set(n for n in tree.nodes if tree.out_degree(n) == 0) - good_leaves

    return best_states


def make_complexity_tree_mst(
    G: nx.DiGraph,
    root_node: Any,
    grammar: Any,
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
        best_states = prune_complexity_tree(
            complexity_tree, root_node, grammar, n_best, success_cutoff
        )
        return complexity_tree, best_states
    else:
        terminal_nodes = set(n for n in complexity_tree.nodes if grammar.is_terminal(n))
        return complexity_tree, terminal_nodes


def merge_search_graphs(
    graphs: list[nx.DiGraph],
    merge_node_attrs: Optional[Container] = None,
    merge_edge_attrs: Optional[Container] = None,
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

    for other_graph in graphs:
        # Add nodes/edges not in the graph and accumulate attributes
        add_nodes = []
        for n, attrs in other_graph.nodes(data=True):
            if n not in graph.nodes:
                add_nodes.append((n, attrs))
            else:
                for a in node_attrs:
                    graph.nodes[n][a] = sum(
                        [graph.nodes[n][a], other_graph.nodes[n][a]]
                    )
        graph.add_nodes_from(add_nodes)

        add_edges = []
        for *e, attrs in other_graph.edges(data=True):
            if e not in graph.edges:
                add_edges.append((e, attrs))
            else:
                for a in edge_attrs:
                    graph.edges[e][a] = sum(
                        [graph.edges[e][a], other_graph.edges[e][a]]
                    )
        graph.add_edges_from(add_edges)

    return graph


def n_connected_components_in_circuit(genotype: Any, grammar: Any) -> int:
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
    grammar: Any,
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
    grammar: Optional[Any] = None,
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
