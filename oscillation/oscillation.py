from functools import cached_property
from itertools import chain
import warnings
from circuitree import CircuiTree
from circuitree.models import SimpleNetworkGrammar
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Iterable
from uuid import uuid4

from tf_network import TFNetworkModel


class OscillationGrammar(SimpleNetworkGrammar):
    @cached_property
    def component_codes(self) -> set[str]:
        return set(c[0] for c in self.components)

    def get_actions(self, genotype: str) -> Iterable[str]:
        # If terminal already, no actions can be taken
        if self.is_terminal(genotype):
            return list()

        # Terminating assembly is always an option
        actions = ["*terminate*"]

        # Get the components and interactions in the current genotype
        components_joined, interactions_joined = genotype.strip("*").split("::")
        components = set(components_joined)
        interactions = set(ixn[:2] for ixn in interactions_joined.split("_") if ixn)
        n_interactions = len(interactions)

        # If we have reached the limit on interactions, only termination is an option
        if n_interactions >= self.max_interactions:
            return actions

        # We can add at most one more component not already in the genotype
        if len(components) < len(self.components):
            actions.append(next(c for c in self.component_codes if c not in components))

        # If we have no interactions yet, don't need to check for connectedness
        elif n_interactions == 0:
            possible_edge_options = [
                grp
                for grp in self.edge_options
                if grp and set(grp[0][:2]).issubset(components)
            ]
            return list(chain.from_iterable(possible_edge_options))

        # Otherwise, add all valid interactions 
        for action_group in self.edge_options:
            if action_group:
                c1_c2 = action_group[0][:2]
                c1, c2 = c1_c2

                # Add the interaction the necessary components are present, that edge
                # isn't already taken, and the added edge would be contiguous with the
                # existing edges (i.e. the circuit should be fully connected)
                has_necessary_components = c1 in components and c2 in components
                connected_to_current_edges = (
                    c1_c2[0] in interactions_joined or c1_c2[1] in interactions_joined
                )
                no_existing_edge = c1_c2 not in interactions
                if (
                    has_necessary_components
                    and connected_to_current_edges
                    and no_existing_edge
                ):
                    actions.extend(action_group)

        return actions

    def is_success(self, state: str) -> bool:
        reward = self.graph.nodes[state]["reward"]
        visits = self.graph.nodes[state]["visits"]
        return visits > 0 and reward / visits > self.success_threshold


class OscillationTree(OscillationGrammar, CircuiTree):
    def __init__(
        self,
        time_points: Optional[np.ndarray[np.float64]] = None,
        success_threshold: float = 0.01,
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
            reward = (rewards[~nan_rewards] > self.autocorr_threshold).mean()
        else:
            reward = np.mean(rewards > self.autocorr_threshold)
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
