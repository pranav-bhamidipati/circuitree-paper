##### A frequentist method for identifying significant motifs from the results of an exhaustive search
# 1. Import the motif table/transposition table
# 2. For each terminal circuit i, compute Q_i = n_i/N and p_osc_i
#   - Each topology i is an oscillator or not based on the number of succcesses n_i in N trials
#   - Therefore, its number of successes is Binomially distributed
#   - The probability of being an oscillator is P(n > N * Q_thresh) = BinomialSF(N * Q_thresh),
#       where Q_thresh is the threshold for being an oscillator
#   - A frequentist approach is to say, "what if we repeated this experiment many times?"
#   - Then the observation that a given topology i is an oscillator is a random variable
#       that is Bernoulli distributed with success probability
#           p_osc_i = Binomial_SF(N * Q_thresh | Q_i)
#       where Q_i = n_i / N and Binomial_SF is the survival function of the Binomial distribution
# 3. Make the full graph of the 3-TF design space with grow_tree()
# 3. For each nonterminal, split the terminals into disjoint reachable and non-reachable sets
#   - The total number of oscillators in the set is
#           m = Sum_i[Bernoulli(p_osc_i)].
#       The sum of Bernoulli trials with different probabilities is described by the
#       *Poisson Binomial* distribution
#   - Therefore, the plugin estimator distribution for the number of oscillators in a set is
#           P(m | {topologies i} ) = PoissonBinomial(m | {p_osc_i})
# 4. Draw M paired bootstrap samples from the reachable and non-reachable sets and compute
#       the distribution of the difference m_reachable - m_nonreachable
# 5. Our null hypothesis is that m_reachable <= m_nonreachable, so if the p-value
#           p = (# times m_reachable - m_nonreachable <= 0) / M
#       is less than the significance level alpha, then we reject the null

from datetime import datetime
from functools import partial
from tqdm import tqdm
from oscillation import OscillationTree, OscillationGrammar
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binom
from typing import Iterable

from graph_utils import split_terminals_by_reachability


class PoissonBinomialRV:
    def __init__(self, p_successes: Iterable[float]):
        self.p_successes = np.array(p_successes)
        self.N_dists = len(self.p_successes)

    def sample(self, size: int = 1) -> np.ndarray:
        return np.sum(np.random.rand(size, self.N_dists) < self.p_successes, axis=1)


def compute_p_value(
    G: nx.DiGraph,
    terminals: list[str],
    p_oscs: np.ndarray,
    n_bootstrap: int,
    nonterminal: str,
):
    # Split terminals into reachable and non-reachable sets
    terminal_idx = {t: i for i, t in enumerate(terminals)}
    X, Y = split_terminals_by_reachability(G, nonterminal, U=set(terminals))

    # Draw paired bootstrap samples of # of oscillators
    p_oscs_X = [p_oscs[terminal_idx[t]] for t in X]
    p_oscs_Y = [p_oscs[terminal_idx[t]] for t in Y]
    m_reachable = PoissonBinomialRV(p_oscs_X).sample(n_bootstrap)
    m_nonreachable = PoissonBinomialRV(p_oscs_Y).sample(n_bootstrap)

    # Compute some summary statistics
    diff_m = m_reachable - m_nonreachable
    mean_diff_m = np.mean(diff_m)
    std_diff_m = np.std(diff_m)

    # Compute the p-value
    p = (diff_m <= 0).mean()
    data = {
        "motif": nonterminal,
        "bootstrap_samples": (m_reachable, m_nonreachable),
        "mean_diff": mean_diff_m,
        "std_diff": std_diff_m,
        "p_value": p,
    }

    return p, data


def main(
    results_csv: Path,
    Q_thresh: float = 0.01,
    n_bootstrap: int = 100_000,
    n_processes: int = 1,
    samples_per_topology: int = 10_000,
    components=["A", "B", "C"],
    interactions=["activates", "inhibits"],
    save_dir: Path = None,
    save: bool = False,
):
    """Computes the p-value of oscillation for each motif in a search graph. Assumes that
    the terminal states (completed circuits) have been sampled the same number of times.
    *NOT* used to analyze results of MCTS, which has biased sampling of terminals."""

    # Read in the results table and
    results_df = pd.read_csv(results_csv)

    # Get [Q_i], the percent of oscillations for each topology i
    terminals = results_df["state"].tolist()
    Qs = results_df["p_oscillation"].values

    # Calculate the probability of being an oscillator if we repeated this experiment:
    ### p_osc_i = Binomial_SF(N * Q_thresh | n_i, N)
    N = samples_per_topology
    p_oscs = np.array([binom(N, Q).sf(Q_thresh * N) for Q in Qs])

    # Make the full graph of the design space
    grammar = OscillationGrammar(components=components, interactions=interactions)
    tree = OscillationTree(
        components=components, interactions=interactions, root="ABC::"
    )
    tree.grow_tree()

    # For each nonterminal, compute a p-value for being a motif
    nonterminals = (
        n for n in tree.graph.nodes if not grammar.is_terminal(n) and n != tree.root
    )
    n_nonterminals = len(tree.graph.nodes) - len(terminals) - 1
    p_values = {}

    if n_processes == 1:
        pbar = tqdm(desc="Computing p-values", total=n_nonterminals)
        for n in nonterminals:
            pval, data = compute_p_value(tree.graph, terminals, p_oscs, n_bootstrap, n)
            p_values[n] = pval
            pbar.update(1)
        pbar.close()

    else:
        from multiprocessing import Pool

        compute_p_val_in_parallel = partial(
            compute_p_value, tree.graph, terminals, p_oscs, n_bootstrap
        )

        with Pool(n_processes) as pool:
            pbar = tqdm(desc="Computing p-values", total=n_nonterminals)
            for pval, data in pool.imap_unordered(
                compute_p_val_in_parallel, nonterminals
            ):
                p_values[data["motif"]] = pval
                pbar.update(1)
            pbar.close()

    # Save the results
    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_exhaustive_motifs_p_values.csv")
        print(f"Writing to: {fpath.resolve().absolute()}")
        pd.DataFrame.from_dict(p_values, orient="index").to_csv(fpath)


if __name__ == "__main__":
    results_csv = Path("data/oscillation/230717_motifs.csv")
    save_dir = results_csv.parent

    main(
        results_csv=results_csv,
        # n_bootstrap=1_000,
        n_processes=13,
        n_bootstrap=100_000,
        save=True,
        save_dir=save_dir,
    )
