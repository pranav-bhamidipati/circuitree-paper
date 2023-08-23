import warnings
from circuitree import SimpleNetworkTree
from functools import cache, cached_property
import h5py
from itertools import permutations
import numpy as np
from pathlib import Path
from typing import Optional, Iterable
from uuid import uuid4

from tf_network import TFNetworkModel


class OscillationTree(SimpleNetworkTree):
    def __init__(
        self,
        time_points: Optional[np.ndarray[np.float64]] = None,
        success_threshold: float = 0.005,
        autocorr_threshold: float = 0.4,
        init_mean: float = 10.0,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        batch_size: int = 1,
        max_iter_per_timestep: int = 100_000_000,
        save_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time_points = time_points
        self.autocorr_threshold = autocorr_threshold
        self.success_threshold = success_threshold

        self.dt = dt
        self.nt = nt
        self.init_mean = init_mean
        self.batch_size = batch_size

        self.max_iter_per_timestep = max_iter_per_timestep

        self.save_dir = save_dir

    @cached_property
    def _recolor(self):
        return [dict(zip(self.components, p)) for p in permutations(self.components)]

    @staticmethod
    def _recolor_string(mapping, string):
        return "".join([mapping.get(c, c) for c in string])

    @cache
    def get_interaction_recolorings(self, genotype: str) -> list[str]:
        if "::" in genotype:
            components, interactions = genotype.split("::")
        else:
            interactions = genotype

        interaction_recolorings = []
        for mapping in self._recolor:
            recolored_interactions = sorted(
                [self._recolor_string(mapping, ixn) for ixn in interactions.split("_")]
            )
            interaction_recolorings.append("_".join(recolored_interactions).strip("_"))

        return interaction_recolorings

    @cache
    def get_component_recolorings(self, genotype: str) -> list[str]:
        if "::" in genotype:
            components, interactions = genotype.split("::")
        else:
            components = genotype

        component_recolorings = []
        for mapping in self._recolor:
            recolored_components = "".join(
                sorted(self._recolor_string(mapping, components))
            )
            component_recolorings.append(recolored_components)

        return component_recolorings

    def get_recolorings(self, genotype: str) -> Iterable[str]:
        ris = self.get_interaction_recolorings(genotype)
        rcs = self.get_component_recolorings(genotype)
        recolorings = ["::".join([rc, ri]) for rc, ri in zip(rcs, ris)]

        return recolorings

    def get_unique_state(self, genotype: str) -> str:
        return min(self.get_recolorings(genotype))

    def has_motif(self, state, motif):
        if ("::" in motif) or ("*" in motif):
            raise ValueError(
                "Motif code should only contain interactions, no components"
            )
        if "::" not in state:
            raise ValueError(
                "State code should contain both components and interactions"
            )

        interaction_code = state.split("::")[1]
        if not interaction_code:
            return False
        state_interactions = set(interaction_code.split("_"))

        for recoloring in self.get_interaction_recolorings(motif):
            motif_interactions = set(recoloring.split("_"))
            if motif_interactions.issubset(state_interactions):
                return True
        return False

    def is_success(self, state: str) -> bool:
        payout = self.graph.nodes[state]["reward"]
        visits = self.graph.nodes[state]["visits"]
        return visits > 0 and payout / visits > self.success_threshold

    def get_reward(
        self,
        state: str,
        batch_size: Optional[int] = None,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        maxiter: Optional[int] = None,
    ) -> float:
        dt = dt if dt is not None else self.dt
        nt = nt if nt is not None else self.nt
        maxiter = maxiter if maxiter is not None else self.max_iter_per_timestep
        batch_size = batch_size if batch_size is not None else self.batch_size

        model = TFNetworkModel(
            state,
            initialize=True,
            dt=dt,
            nt=nt,
            max_iter_per_timestep=maxiter,
        )
        y_t, pop0s, param_sets, rewards = model.run_batch_job(batch_size, abs=True)

        nan_rewards = np.isnan(rewards)
        if nan_rewards.all():
            self.save_nan_data(
                rewards, model=model, y_t=y_t, pop0s=pop0s, param_sets=param_sets
            )
            reward = self.get_reward(state)
        elif nan_rewards.any():
            self.save_nan_data(
                rewards, model=model, y_t=y_t, pop0s=pop0s, param_sets=param_sets
            )
            reward = np.nanmean(rewards)
        else:
            reward = np.mean(rewards)
        return reward

    def save_nan_data(
        self,
        rewards: list[float],
        model: TFNetworkModel,
        pop0s,
        param_sets,
        seed: int | Iterable[int] | None = None,
        **kwargs,
    ) -> None:
        if self.save_dir is None:
            warnings.warn("No save directory specified, not saving nan data")
            return

        where_nans = np.isnan(rewards)
        if not where_nans.any():
            return

        warnings.warn(f"Saving data for {where_nans.sum()} runs with nan rewards")

        # Save the nan data to HDF
        state = model.genotype
        dt = model.dt
        nt = model.nt
        maxiter = model.max_iter_per_timestep
        if seed is None:
            save_seed = model.seed
        elif isinstance(seed, Iterable):
            save_seed = seed[where_nans]
        else:
            save_seed = seed

        fname = f"nans_state_{state}_uid_{uuid4()}.hdf5"
        fpath = Path(self.save_dir) / fname
        with h5py.File(fpath, "w") as f:
            f.create_dataset("seed", data=save_seed)
            f.create_dataset("pop0s", data=pop0s[where_nans])
            f.create_dataset("param_sets", data=param_sets[where_nans])
            f.attrs["state"] = state
            f.attrs["dt"] = dt
            f.attrs["nt"] = nt
            f.attrs["max_iter_per_timestep"] = maxiter
