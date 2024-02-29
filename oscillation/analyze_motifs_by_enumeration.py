from datetime import datetime
from functools import partial
from circuitree import CircuiTree, SimpleNetworkGrammar
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from graph_utils import simplenetwork_n_interactions


# def contingency_test(pattern: str, tree: OscillationTree, **kwargs):
#     df = tree._contingency_test(pattern, **kwargs)
#     df["pattern"] = pattern
#     return df


def compute_odds_ratio_and_ci(
    table: np.ndarray, confidence_level: float
) -> tuple[float, tuple[float, float]]:
    """Compute the odds ratio and confidence interval for a 2x2 contingency table.
    Each row [a, b, c, d] in the array corresponds to the 2x2 contingnecy table:

            [[a, b],
             [c, d]]
    """
    # Compute the odds ratio
    (a, b), (c, d) = table
    bc = b * c
    if bc == 0:
        odds_ratio = np.inf
    else:
        odds_ratio = (a * d) / bc

    # Compute the confidence interval
    if any(table.flatten() == 0):
        return odds_ratio, (np.nan, np.nan)
    else:
        upper_quantile = (1 + confidence_level) / 2
        log_odds_ratio = np.log(odds_ratio)
        std_err_log_OR = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        log_ci_width = stats.norm.ppf(upper_quantile) * std_err_log_OR
        ci_low = np.exp(log_odds_ratio - log_ci_width)
        ci_high = np.exp(log_odds_ratio + log_ci_width)
        return odds_ratio, (ci_low, ci_high)


def compute_odds_ratios(abcd: np.ndarray, progress: bool = False) -> np.ndarray:
    """Compute the odds ratio for multiple 2x2 contingency tables. Takes a 2D array of
    shape (n, 4) where n is the number of contingency tables.
    Each row [a, b, c, d] in the array corresponds to the 2x2 contingnecy table:

            [[a, b],
             [c, d]]

    """
    tables = abcd.reshape(-1, 2, 2)
    odds_ratios = np.zeros(len(tables))
    iterator = tables
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="Computing odds ratios")
    for i, table in enumerate(iterator):
        (a, b), (c, d) = table
        bc = b * c
        if bc == 0:
            odds_ratios[i] = np.inf
        else:
            odds_ratios[i] = (a * d) / (b * c)
    return odds_ratios


def compute_odds_ratios_with_ci(
    abcd: np.ndarray, confidence_level: float, progress: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the odds ratio and confidence intervals for multiple 2x2 contingency
    tables. Takes a 2D array of shape (n, 4) where n is the number of contingency
    tables.
    Each row [a, b, c, d] in the array corresponds to the 2x2 table:

            [[a, b],
             [c, d]]

    """
    tables = abcd.reshape(-1, 2, 2)
    odds_ratios = np.zeros(len(tables))
    cis = np.zeros((len(tables), 2))
    iterator = tables
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="Computing odds ratios +/- CI")
    for i, table in enumerate(iterator):
        odds_ratios[i], cis[i] = compute_odds_ratio_and_ci(table, confidence_level)
    return odds_ratios, *cis.T


def main(
    results_csv: Path,
    Q_threshold: float = 0.01,
    n_samples_exhaustive: int = 10_000,
    significance_threshold: int = 0.05,
    confidence: float = 0.95,
    barnard_ok: bool = True,
    nprocs: int = 1,
    save: bool = False,
    save_dir: Path = None,
    progress: bool = False,
):
    # Load the results of an enumeration and Identify successful circuits
    results_df = pd.read_csv(results_csv)
    all_circuits = set(results_df["state"].values)
    print(f"# total circuits in search tree: {len(all_circuits)}")

    components, *_ = SimpleNetworkGrammar.parse_genotype(next(iter(all_circuits)))
    grammar = SimpleNetworkGrammar(
        components=components,
        interactions=["activates", "inhibits"],
        root="".join(components) + "::",
    )

    Qs = dict(results_df[["state", "p_oscillation"]].values)
    Qs = {
        s: q for s, q in Qs.items() if s in all_circuits
    }  # keep only the fully connected circuits
    successful_circuits = set(n for n, q in Qs.items() if q >= Q_threshold)
    print(f"# oscillators identified by enumeration: {len(successful_circuits)}")

    testable_patterns = {
        g.split("::")[1]: g
        for g in successful_circuits
        if round(Qs[g] * n_samples_exhaustive) > 5
    }

    if not barnard_ok:
        # To run a chi-square test, there must be at least 5 successful circuits that
        # contain this pattern
        def _at_least_five_matches(pattern):
            n = 0
            for s in successful_circuits:
                if grammar.has_pattern(s, pattern):
                    n += 1
                if n > 5:
                    return True
            return False

        chisquare_ok = set()
        for pattern in testable_patterns:
            if _at_least_five_matches(pattern):
                chisquare_ok.add(pattern)

        print(
            f"Excluding {len(testable_patterns) - len(chisquare_ok)} based on the "
            "requirements to run a chi-squared test"
        )
        testable_patterns = {
            k: v for k, v in testable_patterns.items() if k in chisquare_ok
        }

    print(f"Testing {len(testable_patterns)} patterns for overrepresentation...")

    dfs = []
    do_one_test = partial(
        # contingency_test,
        # tree=tree,
        CircuiTree._contingency_test,
        grammar=grammar,
        null_samples=all_circuits,
        succ_samples=successful_circuits,
        correction=True,
        barnard_ok=True,
        exclude_self=True,
    )

    if nprocs > 1:
        from multiprocessing import Pool

        if progress:
            from tqdm import tqdm

            pbar = tqdm(desc="Running tests", total=len(testable_patterns))

        with Pool(nprocs) as pool:
            for df in pool.imap_unordered(do_one_test, testable_patterns.keys()):
                dfs.append(df)
                if progress:
                    pbar.update()

    else:
        iterator = testable_patterns.keys()
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Running tests")

        for pattern in iterator:
            df = do_one_test(pattern)
            dfs.append(df)

    ## Concatenate the results and wrangle the data
    # Pivot the 2x2 contingency table columns into 4 separate columns
    results_df = pd.concat(dfs).reset_index()
    pivoted = results_df.pivot_table(
        index="pattern",
        columns="has_pattern",
        values=["successful_paths", "overall_paths"],
    )
    pivoted = pivoted.astype(int)
    pivoted.columns = pivoted.columns.map(
        {
            ("overall_paths", False): "others_in_null",
            ("overall_paths", True): "pattern_in_null",
            ("successful_paths", False): "others_in_succ",
            ("successful_paths", True): "pattern_in_succ",
        }
    )

    # Drop the columns that are not needed anymore
    results_df = (
        results_df.loc[results_df["has_pattern"]]
        .set_index("pattern")
        .drop(columns=["has_pattern", "successful_paths", "overall_paths"])
    )
    results_df = pd.concat([results_df, pivoted], axis=1).reset_index()

    # Perform multiple test correction (Bonferroni)
    results_df["p_corrected"] = results_df["pvalue"] * len(results_df)

    # Compute confidence intervals for the odds ratio
    abcd = results_df[
        ["pattern_in_succ", "pattern_in_null", "others_in_succ", "others_in_null"]
    ].values

    if confidence is None:
        results_df["odds_ratio"] = compute_odds_ratios(abcd, progress=progress)
    else:
        odds_ratios, cis_low, cis_high = compute_odds_ratios_with_ci(
            abcd, confidence_level=confidence, progress=progress
        )
        ci_level = f"{int(confidence * 100)}%"
        results_df["odds_ratio"] = odds_ratios
        results_df[f"ci_{ci_level}_low"] = cis_low
        results_df[f"ci_{ci_level}_high"] = cis_high

    # Compute complexity
    results_df["complexity"] = results_df["pattern"].map(simplenetwork_n_interactions)

    # Store the p_oscillation column of the results df in the pattern df
    results_df["Q"] = results_df["pattern"].map(testable_patterns).map(Qs)

    # Find the patterns that are sufficient for oscillation and significantly
    # overrepresented (motifs)
    results_df["sufficient"] = True
    results_df["significant"] = results_df["p_corrected"] < significance_threshold
    results_df["overrepresented"] = results_df["odds_ratio"] > 1.0

    n_motifs = (
        results_df["sufficient"]
        & results_df["significant"]
        & results_df["overrepresented"]
    ).sum()
    print(f"Found {n_motifs} significantly overrepresented motifs")

    results_df = results_df.sort_values(["complexity", "p_corrected"])

    if save:
        today = datetime.now().strftime("%y%m%d")
        fname = Path(save_dir).joinpath(
            f"{today}_circuit_pattern_tests_exhaustive_search.csv"
        )
        print(f"Writing to: {fname.resolve().absolute()}")
        results_df.to_csv(fname)

    else:
        return results_df


if __name__ == "__main__":
    results_csv = Path("data/oscillation/231102_exhaustive_results.csv")
    save_dir = Path("data/oscillation/")
    main(
        results_csv=results_csv,
        n_samples_exhaustive=10_000,
        barnard_ok=False,
        nprocs=13,
        progress=True,
        save=True,
        save_dir=save_dir,
    )
