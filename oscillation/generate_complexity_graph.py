from datetime import date
from pathlib import Path
from typing import Optional
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
from oscillation import OscillationTree


def main(
    graph_gml: Path,
    attrs_json: Optional[Path] = None,
    save_dir: Path = Path("./data/oscillation"),
    reward_threshold: float = 0.01,
    save: bool = False,
    accumulate: bool = False,
    **kwargs,
):
    print("Loading circuit search tree...")
    tree = OscillationTree.from_file(graph_gml, attrs_json)
    if accumulate:
        tree.accumulate_visits_and_rewards()

    print("Constructing complexity graph...")

    from circuitree.viz import complexity_layout

    ...
    C, pos, edge_weights = complexity_layout(tree, reward_threshold=reward_threshold)
    ...

    # C = tree.to_complexity_graph(reward_threshold=reward_threshold)
    # pos = graphviz_layout(C, prog="dot")

    pos_df = pd.DataFrame.from_dict(pos, orient="index", columns=("x", "y"))
    edges_df = pd.DataFrame(C.edges(), columns=["source", "target"])

    if save:
        today = date.today().strftime("%y%m%d")

        fname = (
            f"{today}_oscillation_complexity_graph_"
            f"thresh{reward_threshold:.3f}_nodes_xy.csv"
        )
        fpath = Path(save_dir).joinpath(fname)
        print(f"Writing to: {fpath.resolve().absolute()}")
        pos_df.to_csv(fpath)

        fpath = fpath.with_name(fpath.stem.replace("nodes", "edges"))
        print(f"Writing to: {fpath.resolve().absolute()}")
        edges_df.to_csv(fpath)


if __name__ == "__main__":
    gml = Path("data/oscillation/230726_3tf_master.gml")
    attrs_json = Path("data/oscillation/230726_3tf_master.json")
    save_dir = Path("data/oscillation")

    main(
        graph_gml=gml,
        attrs_json=attrs_json,
        save_dir=save_dir,
        save=True,
    )
