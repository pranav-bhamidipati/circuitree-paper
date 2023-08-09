from collections import Counter
from functools import cache, cached_property
from itertools import permutations, product
import numpy as np
from typing import Iterable
from circuitree import CircuiTree

__all__ = ["DimerNetworkTree"]


class DimerNetworkTree(CircuiTree):
    """
    DimerNetworkTree
    =================
    A CircuiTree for the design space of dimerizing TF networks.



    """

    def __init__(
        self,
        components: Iterable[str],
        regulators: Iterable[str],
        interactions: Iterable[str],
        n_binding_sites: int = 2,
        **kwargs,
    ):
        if len(set(c[0] for c in components)) < len(components):
            raise ValueError("First character of each component must be unique")
        if len(set(c[0] for c in regulators)) < len(regulators):
            raise ValueError("First character of each component must be unique")
        if len(set(c[0] for c in interactions)) < len(interactions):
            raise ValueError("First character of each interaction must be unique")

        super().__init__(**kwargs)

        self.components = components
        self.component_map = {c[0]: c for c in self.components}
        self.regulators = regulators
        self.regulator_map = {r[0]: r for r in self.regulators}
        self.interactions = interactions
        self.interaction_map = {ixn[0]: ixn for ixn in self.interactions}

        self.n_binding_sites = n_binding_sites

    @cached_property
    def dimer_options(self):
        dimers = set()
        monomers = self.components + self.regulators
        for c1, c2 in product(monomers, monomers):
            if c1 in self.regulators and c2 in self.regulators:
                continue
            dimers.add("".join(sorted([c1[0], c2[0]])))
        return list(dimers)

    @cached_property
    def edge_options(self):
        return [
            [d + ixn[0] + c[0] for ixn in self.interactions]
            for d, c in product(self.dimer_options, self.components)
        ]

    @staticmethod
    def is_terminal(genotype: str) -> bool:
        return genotype.startswith("*")

    def get_actions(self, genotype: str) -> Iterable[str]:
        if self.is_terminal(genotype):
            return list()

        # Terminating assembly is always an option
        actions = ["*terminate*"]

        components, regulators, interactions_joined = self.get_genotype_parts(genotype)
        n_bound = Counter()
        for ixn in interactions_joined.split("_"):
            if ixn:
                n_bound[ixn[3]] += 1

        for action_group in self.edge_options:
            if action_group:
                promoter = action_group[0][-1]
                if n_bound[promoter] < self.n_binding_sites:
                    actions.extend(action_group)

        return actions

    def do_action(self, genotype: str, action: str) -> str:
        if action == "*terminate*":
            new_genotype = "*" + genotype
        else:
            components_and_regulators, interactions = genotype.split("::")
            delim = ("", "_")[bool(interactions)]
            new_genotype = "::".join(
                [components_and_regulators, delim.join([interactions, action])]
            )
        return new_genotype

    @staticmethod
    @cache
    def get_unique_state(genotype: str) -> str:
        components_and_regulators, interactions = genotype.split("::")
        prefix = ""
        if components_and_regulators.startswith("*"):
            prefix = "*"
            components_and_regulators = components_and_regulators[1:]
        if "+" in components_and_regulators:
            components, regulators = components_and_regulators.split("+")
            unique_regulators = "".join(sorted(components))
            joiner = "+"
        else:
            components = components_and_regulators
            unique_regulators = ""
            joiner = ""

        unique_components = "".join(sorted(components))
        unique_components_and_regulators = joiner.join(
            [unique_components, unique_regulators]
        )

        unique_interaction_list = []
        for ixn in interactions.split("_"):
            m1, m2, logic, promoter = ixn
            unique_ixn = "".join(sorted([m1, m2])) + logic + promoter
            unique_interaction_list.append(unique_ixn)
        unique_interactions = "_".join(sorted(unique_interaction_list))

        return prefix + unique_components_and_regulators + "::" + unique_interactions

    @staticmethod
    def get_genotype_parts(genotype: str):
        components_and_regulators, interactions = genotype.strip("*").split("::")
        if "+" in components_and_regulators:
            components, regulators = components_and_regulators.split("+")
        else:
            components = components_and_regulators
            regulators = ""

        return components, regulators, interactions

    @staticmethod
    def parse_genotype(genotype: str, nonterminal_ok: bool = False):
        if not genotype.startswith("*") and not nonterminal_ok:
            raise ValueError(
                f"Assembly incomplete. Genotype {genotype} is not a terminal genotype."
            )

        components_and_regulators, interaction_codes = genotype.strip("*").split("::")
        if "+" in components_and_regulators:
            components, regulators = components_and_regulators.split("+")
        else:
            components = components_and_regulators
            regulators = ""
        component_indices = {c: i for i, c in enumerate(components + regulators)}

        interactions = interaction_codes.split("_") if interaction_codes else []

        activations = []
        inhbitions = []
        for monomer1, monomer2, ixn, promoter in interactions:
            m1, m2 = sorted([monomer1, monomer2])
            ixn_tuple = (component_indices[m1], component_indices[m2], promoter)
            if ixn.lower() == "a":
                activations.append(ixn_tuple)
            elif ixn.lower() == "i":
                inhbitions.append(ixn_tuple)
            else:
                raise ValueError(f"Unknown interaction type {ixn} in {genotype}")

        activations = np.array(activations, dtype=np.int_)
        inhbitions = np.array(inhbitions, dtype=np.int_)

        return components, activations, inhbitions

    @cached_property
    def _recolor_components(self):
        return [dict(zip(self.components, p)) for p in permutations(self.components)]

    @cached_property
    def _recolor_regulators(self):
        return [dict(zip(self.regulators, p)) for p in permutations(self.regulators)]

    @cached_property
    def _recolorings(self):
        return [
            rc | rr
            for rc, rr in product(self._recolor_components, self._recolor_regulators)
        ]

    @staticmethod
    def _recolor(mapping, code):
        return "".join([mapping.get(char, char) for char in code])

    def get_interaction_recolorings(self, genotype: str) -> list[str]:
        *_, interactions = self.get_genotype_parts(genotype)
        interaction_recolorings = (
            "_".join(
                sorted([self._recolor(mapping, ixn) for ixn in interactions.split("_")])
            ).strip("_")
            for mapping in self._recolorings
        )
        return interaction_recolorings

    def get_component_recolorings(self, genotype: str) -> list[str]:
        components, *_ = self.get_genotype_parts(genotype)
        component_recolorings = (
            "".join(sorted(self._recolor(mapping, components)))
            for mapping in self._recolor_components
        )
        return component_recolorings

    def get_regulator_recolorings(self, genotype: str) -> list[str]:
        _, regulators, *_ = self.get_genotype_parts(genotype)
        regulator_recolorings = (
            "".join(sorted(self._recolor(mapping, regulators)))
            for mapping in self._recolor_components
        )
        return regulator_recolorings

    def get_recolorings(self, genotype: str) -> Iterable[str]:
        prefix = "*" if self.is_terminal(genotype) else ""

        _, regulators, *_ = self.get_genotype_parts(genotype)
        if regulators:
            return (
                f"{prefix}{c}+{r}::{i}"
                for c, r, i in zip(
                    self.get_component_recolorings(genotype),
                    self.get_regulator_recolorings(genotype),
                    self.get_interaction_recolorings(genotype),
                )
            )
        else:
            return (
                f"{prefix}{c}::{i}"
                for c, i in zip(
                    self.get_component_recolorings(genotype),
                    self.get_interaction_recolorings(genotype),
                )
            )

    @cache
    def get_unique_state(self, genotype: str) -> str:
        return min(self.get_recolorings(genotype))

    @cache
    def _motif_recolorings(self, motif: str) -> list[set[str]]:
        if ("+" in motif) or ("::" in motif) or ("*" in motif):
            raise ValueError(
                "Motif code should only contain interactions, no components or "
                "regulators"
            )

        return [
            set(recoloring.split("_"))
            for recoloring in self.get_interaction_recolorings(motif)
        ]

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

        for motif_interactions_set in self._motif_recolorings(motif):
            if motif_interactions_set.issubset(state_interactions):
                return True
        return False
