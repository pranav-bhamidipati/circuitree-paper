from circuitree.models import SimpleNetworkTree
import numpy as np
from pathlib import Path
from typing import Optional, Iterable
from uuid import uuid4

from tf_network import TFNetworkModel


class OscillationTree(SimpleNetworkTree):
    def __init__(
        self,
        time_points: Optional[np.ndarray[np.float64]] = None,
        Q_threshold: float = 0.01,
        ACF_threshold: float = -0.4,
        init_mean: float = 10.0,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        save_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time_points = time_points
        self.ACF_threshold = ACF_threshold
        self.Q_threshold = Q_threshold

        self.dt = dt
        self.nt = nt
        self.init_mean = init_mean

        self.save_dir = save_dir

    def is_success(self, state: str) -> bool:
        reward = self.graph.nodes[state]["reward"]
        visits = self.graph.nodes[state]["visits"]
        return visits > 0 and reward / visits >= self.Q_threshold

    def get_reward(
        self,
        state: str,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        maxiter: Optional[int] = None,
    ) -> float:
        dt = dt if dt is not None else self.dt
        nt = nt if nt is not None else self.nt
        maxiter = maxiter if maxiter is not None else self.max_iter_per_timestep
        model = TFNetworkModel(
            state,
            initialize=True,
            dt=dt,
            nt=nt,
            max_iter_per_timestep=maxiter,
        )
        y_t, pop0, param_set, acf_min = model.run_job()

        reward = float(acf_min < self.ACF_threshold)
        return reward
