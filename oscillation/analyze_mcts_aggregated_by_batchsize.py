from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

from models.oscillation.oscillation import OscillationTree


def main(
    data_dir: Path,
    save_dir: Optional[Path] = None,
    save: bool = False,
):
    data_dir = Path(data_dir)
    gmls = sorted(data_dir.glob("*batchsize*.gml"))
    jsons = sorted(data_dir.glob("*batchsize*.json"))
    batch_sizes = defaultdict(list)
    for gml, json in zip(gmls, jsons):
        batch_size = int(gml.stem.split("_")[3].split("batchsize")[1])
        batch_sizes[batch_size].append((gml, json))

    # Compute the aggregate probability of oscillation across replicate search trees
    for batch_size, gmls_and_jsons in batch_sizes.items():
        gml0, json0 = gmls_and_jsons[0]
        tree = OscillationTree.from_file(gml0, json0)
        tree.accumulate_visits_and_rewards()
        for gml, json in gmls_and_jsons[1:]:
            _t = OscillationTree.from_file(gml, json)
            _t.accumulate_visits_and_rewards()
            for n, ndata in _t.graph.nodes(data=True):
                if n not in tree.graph.nodes:
                    tree.graph.add_node(n, **_t.graph.nodes[n])
                else:
                    tree.graph.nodes[n]["visits"] += ndata["visits"]
                    tree.graph.nodes[n]["reward"] += ndata["reward"]
            for *e, edata in _t.graph.edges(data=True):
                if e not in tree.graph.edges:
                    tree.graph.add_edge(*e, **edata)
                else:
                    tree.graph.edges[e]["visits"] += edata["visits"]
                    tree.graph.edges[e]["reward"] += edata["reward"]

        ...

        if save:
            date_str = date.today().strftime("%y%m%d")
            save_stem = Path(save_dir).joinpath(
                f"{date_str}_oscillation_aggregated_tree_batchsize{batch_size}"
            )
            save_gml = save_stem.with_suffix(".gml")
            save_json = save_stem.with_suffix(".json")
            print(f"Writing to: {save_gml.resolve().absolute()}")
            print(f"Writing to: {save_json.resolve().absolute()}")
            tree.to_file(
                gml_file=save_gml,
                json_file=save_json,
                save_attrs=["root", "components", "interactions"],
            )

    ...


if __name__ == "__main__":
    data_dir = Path("data/oscillation/230725_mcts_bootstrap_boolean2")
    save_dir = Path("data/oscillation")
    main(
        data_dir=data_dir,
        save_dir=save_dir,
        save=True,
    )
