"""Scaling methods for learned neural (state-action) Lyapunov functions.

Implements the three scaling methods of Chapter 3.1 so that clean_lsac, clean_lppo and clean_ly
can switch between them from the command line:

    none            Eq. (2.1): unscaled Lyapunov risk (equivalent to min_lie / regularization with mu = 0)
    min_lie         Method 1, Eq. (3.2)-(3.4): enforce a minimum Lie derivative, L_dot <= -mu
    regularization  Method 2, Eq. (3.6): add mu * (L(x, u) - ||x||^2)^2 to the Lyapunov risk
    structural      Method 3, Eq. (3.7)-(3.8): L(x, u) = x^T P x + (g(x, u) - g(0, u(0)))^2 with
                    P = A^T A + eps I, trained only on the decreasing condition

States are measured relative to the equilibrium state, so `x` above is `state - eq_state`. For the
state-only Lyapunov function used by clean_ly, pass `action_dim=0` and omit the actions.
"""
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ScalingMethod = Literal["none", "min_lie", "regularization", "structural"]


class LyapunovNet(nn.Module):
    def __init__(self, state_dim, action_dim, lr=3e-4, fc1_dims=256, fc2_dims=256, structural=False, eps=1e-3):
        super().__init__()
        self.structural = structural
        self.g = nn.Sequential(
            nn.Linear(state_dim + action_dim, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )
        if structural:
            # P = A^T A + eps I is positive definite, so the quadratic term cannot collapse to zero
            self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
            self.eps = eps

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _g(self, state, action):
        if action is None:
            return self.g(state)
        return self.g(torch.cat([state, action], dim=1))

    def forward(self, state, action=None, eq_state=None, eq_action=None):
        if not self.structural:
            return self._g(state, action)

        # Eq. (3.7): zero at (eq_state, eq_action) and positive everywhere else by construction
        dx = state - eq_state
        P = self.A.T @ self.A + self.eps * torch.eye(self.A.shape[0], dtype=dx.dtype, device=dx.device)
        quad = ((dx @ P) * dx).sum(dim=1, keepdim=True)
        g_eq = self._g(eq_state, eq_action)
        return quad + (self._g(state, action) - g_eq) ** 2


def lie_margin(method: ScalingMethod, mu: float) -> float:
    """Minimum decrease mu added to the Lie derivative in the Lyapunov and policy losses (method 1 only)."""
    return mu if method == "min_lie" else 0.0


def certification_loss(l_val, l_lie, l_eq):
    """Unscaled Lyapunov risk of Eq. (2.1); zero certifies stability regardless of the scaling method."""
    return F.relu(-l_val).mean() + F.relu(l_lie).mean() + (l_eq**2).mean()


def lyapunov_loss(method: ScalingMethod, mu: float, l_val, l_lie, l_eq, state, eq_state):
    if method == "structural":
        # Eq. (3.8): positivity and zero at the equilibrium hold by construction
        return F.relu(l_lie).mean()

    loss = F.relu(-l_val).mean() + F.relu(l_lie + lie_margin(method, mu)).mean() + (l_eq**2).mean()
    if method == "regularization":
        sq_norm = ((state - eq_state) ** 2).sum(dim=1, keepdim=True)
        loss = loss + mu * ((l_val - sq_norm) ** 2).mean()
    return loss


def run_tag(method: ScalingMethod, mu: float) -> str:
    """Short identifier used to keep results of different scaling configurations apart."""
    if method in ("none", "structural"):
        return method
    return f"{method}_mu{mu:g}"
