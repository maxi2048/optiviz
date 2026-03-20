from base import BaseOptimizer
import numpy as np


class Newton(BaseOptimizer):
    """
    Newton-Verfahren.
    Schritt: x_new = x - H(x)^{-1} * grad(x)
    """

    name = "Newton"

    def gradient(self, x, eps=1e-5):
        """Numerischer Gradient mit zentralen Differenzen."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = eps
            grad[i] = (self.f(x + e) - self.f(x - e)) / (2 * eps)
        return grad

    def hessian(self, x, eps=1e-4):
        """Berechnet die Hesse-Matrix numerisch."""
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ei = np.zeros(n)
                ej = np.zeros(n)
                ei[i] = eps
                ej[j] = eps
                H[i, j] = (
                    self.f(x + ei + ej)
                    - self.f(x + ei - ej)
                    - self.f(x - ei + ej)
                    + self.f(x - ei - ej)
                ) / (4 * eps ** 2)
        return H

    def step(self, x):
        grad = self.gradient(x)
        H = self.hessian(x)
        d = np.linalg.solve(H, grad)
        return x - d
