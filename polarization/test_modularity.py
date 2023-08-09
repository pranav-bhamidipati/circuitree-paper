from circuitree import OscillationTreeBase

if __name__ == "__main__":
    components = ["A", "B"]
    interactions = ["activates", "inhibits"]
    root = "AB::"

    SNT = OscillationTreeBase(
        components=components, interactions=interactions, root=root
    )
    SNT.is_success = lambda n: "AAa" in n
    SNT.grow_tree("AB::")

    ...

    Mt1 = SNT.modularity

    ...

    from polarization import PolarizationTree
    import pandas as pd

    polar_df = pd.read_csv("data/polarization2_data.csv")

    winners = polar_df.genotype.values
    win_probabilities = dict(zip(polar_df.genotype.values, polar_df.Q.values))

    dag = PolarizationTree(
        components=components,
        interactions=interactions,
        root=root,
        winners=winners,
        tree_shape="dag",
    )
    dag.grow_tree()

    ...

    Mt2 = dag.modularity

    ...
