# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint manager for computing Constraint signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ConstraintTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ConstrainedManagerBasedRLEnv


class ConstraintManager(ManagerBase):
    """Manager for computing Constraint signals for a given world.

    The Constraint manager computes the total constraint as a sum of the weighted constraint terms. The constraint
    terms are parsed from a nested config class containing the constraint manger's settings and constraint
    terms configuration.

    The constraint terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each constraint term should instantiate the :class:`constraintTermCfg` class.

    .. note::

        The constraint manager multiplies the constraint term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed constraint terms are balanced with
        respect to the chosen time-step interval in the environment.

    """

    _env: ConstrainedManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ConstrainedManagerBasedRLEnv):
        """Initialize the constraint manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ConstraintTermCfg]``).
            env: The environment instance.
        """
        super().__init__(cfg, env)
        # Initialize running probs for constraint
        self.running_maxes = dict() # Polyak averaging for max constraint violation
        self.running_mins = dict() # Polyak averaging for min constraint violation
        self.probs = dict() # Termination probabilities for each constraint
        self.max_p = dict() # Maximum termination probability for each constraint
        self.raw_constraints = dict()
        self.tau = 0.95 # Discount factor
        self.min_p = 0.0 # Minimum probability for termination

        # prepare extra info to store individual constraint term information
        self._episode_sums = dict()
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # create buffer for managing constraint per environment
        self._constraint_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def __str__(self) -> str:
        """Returns: A string representation for constraint manager."""
        msg = f"<ConstraintManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Constraint Terms"
        table.field_names = ["Index", "Name", "Init. Max P", "Final Max P"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Init. Max P"] = "r"
        table.align["Final Max Prob"] = "r"

        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.init_max_p, term_cfg.final_max_p])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active constraint terms."""
        return self._term_names

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual constraint terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual constraint terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual constraint terms.
        """
        # Reset the termination probabilities of the constraint manager
        self.probs.clear()
        self.raw_constraints.clear()
        
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            # store information
            # r_1 + r_2 + ... + r_n
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_constraint/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            # reset episodic sum
            self._episode_sums[key][env_ids] = 0.0
        # reset all the constraint terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras
        

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the constraint signal as a weighted sum of individual terms.

        This function calls each constraint term managed by the class and adds them to compute the net
        constraint signal. It also updates the episodic sums corresponding to individual constraint terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net constraint signal of shape (num_envs,).
        """
        # reset computation
        self._constraint_buf[:] = 0.0
        sqrt_func = lambda x: x.clamp(min=0.0).sqrt()
        # iterate over all the constraint terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # compute the constraint signal
            constraint = term_cfg.func(self._env, **term_cfg.params) #* dt
            # obtain the termination probability for the constraint
            self.add(name, sqrt_func(constraint), term_cfg.max_p)
        
        # Log the termination probabilities for each constraint
        self.log_all()

        return self.get_probs()
    
    def add(self, name, constraint, max_p=0.1):
        """Add a constraint violation to the constraint manager and compute the
        associated termination probability.

        Args:
            name (string): name of the constraint
            constraint (float tensor): value of constraint violations for this constraint
            max_p (float): maximum termination probability
        """

        # First, put constraint in the form Torch.FloatTensor((num_envs, n_constraints))
        # Convert constraints violation to float if they are not
        if not torch.is_floating_point(constraint):
            constraint = constraint.float()

        # Ensure constraint is 2-dimensional even with a single element
        if len(constraint.size()) == 1:
            constraint = constraint.unsqueeze(1)
        
        # Check that max_p is not 0
        if max_p == 0.0:
            max_p = self.get_term_cfg(name).init_max_p

        # Get the maximum constraint violation for the current step
        constraint_max = constraint.max(dim=1, keepdim=True)[0].clamp(min=1e-6)

        # Compute polyak average of the maximum constraint violation for this constraint
        if name not in self.running_maxes:
            self.running_maxes[name] = constraint_max
        else:
            self.running_maxes[name] = (
                self.tau * self.running_maxes[name] + (1.0 - self.tau) * constraint_max
            )
        
        # Store raw constraint value
        self.raw_constraints[name] = constraint

        # Get samples for which there is a constraint violation
        mask = constraint > 0.0

        # Compute the termination probability which scales between min_p and max_p with
        # increasing constraint violation. Remains at 0 when there is no violation.
        probs = torch.zeros_like(constraint)
        probs[mask] = self.min_p + torch.clamp(
            constraint[mask]
            / (self.running_maxes[name].expand(constraint.size())[mask]),
            min=0.0,
            max=1.0,
        ) * (max_p - self.min_p)
        self.probs[name] = probs
        self.max_p[name] = torch.tensor(max_p, device=self.device).repeat(constraint.shape[1])

    def get_probs(self) -> torch.Tensor:
        """Returns the termination probabilities due to constraint violations."""
        probs = torch.cat(list(self.probs.values()), dim=1)
        probs = probs.max(1).values
        return probs
    
    def get_raw_constraints(self) -> torch.Tensor:
        """Returns the raw constraint violations."""
        return torch.cat(list(self.raw_constraints.values()), dim=1)
    
    def get_running_maxes(self) -> torch.Tensor:
        """Returns the running maximum constraint violations."""
        return torch.cat(list(self.running_maxes.values()), dim=1)
    
    def get_max_p(self) -> torch.Tensor:
        """Returns the maximum termination probabilities for each constraint."""
        return torch.cat(list(self.max_p.values()), dim=1)
    
    def log_all(self):
        """Logs the termination probabilities for each constraint."""
        for name in list(self.probs.keys()):
            values = self.probs[name].max(1).values.gt(0.0).float()
            if name not in self._episode_sums:
                self._episode_sums[name] = torch.zeros_like(values)
            self._episode_sums[name] += values
    
    def get_names(self):
        """Returns the names of the constraints."""
        return list(self.probs.keys())
    
    def get_str(self, names=None):
        """Get a debug string with constraints names and their average termination probabilities"""
        if names is None:
            names = list(self.probs.keys())
        txt = ""
        for name in names:
            txt += " {}: {}".format(
                name,
                str(
                    100.0 * self.probs[name].max(1).values.gt(0.0).float().mean().item()
                )[:4],
            )
            txt += "\n"
            # txt += " {}: {}".format(name, str(100.0*self.probs[name].max(1).values.float().mean().item())[:4])

        return txt[1:]
    
    def get_vals(self):
        """Returns the values of the constraints."""
        return torch.cat(list(self.probs.values()), dim=1)
    
        
    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: ConstraintTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the constraint term.
            cfg: The configuration for the constraint term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"constraint term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> ConstraintTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the constraint term.

        Returns:
            The configuration of the constraint term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"constraint term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_names.index(term_name)]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of constraint functions."""
        # parse remaining constraint terms and decimate their information
        self._term_names: list[str] = list()
        self._term_cfgs: list[ConstraintTermCfg] = list()
        self._class_term_cfgs: list[ConstraintTermCfg] = list()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, ConstraintTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ConstraintTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # check for valid max_p type
            if not isinstance(term_cfg.max_p, (float)):
                raise TypeError(
                    f"Weight for the term '{term_name}' is not of type float"
                    f" Received: '{type(term_cfg.max_p)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
