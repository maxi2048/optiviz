"""
optiviz/algorithms/base.py

Basisklasse fuer alle Optimierungsalgorithmen in optiviz.
Jeder Algorithmus erbt von BaseOptimizer und implementiert die Methode `step`.
"""

import numpy as np


class OptimizationResult:
    """Speichert das Ergebnis eines Optimierungslaufs."""

    def __init__(self, minimum, f_minimum, path, f_path, n_iter, success, algo_name):
        self.minimum = minimum        # bestes x das gefunden wurde
        self.f_minimum = f_minimum    # f(minimum)
        self.path = path              # alle besuchten Punkte, shape (n_iter+1, n_dims)
        self.f_path = f_path          # f-Werte entlang des Pfades
        self.n_iter = n_iter          # Anzahl Iterationen
        self.success = success        # hat der Algorithmus konvergiert?
        self.algo_name = algo_name    # z.B. "Newton"

    def __repr__(self):
        status = "konvergiert" if self.success else "nicht konvergiert"
        return (
            f"OptimizationResult({self.algo_name} | {status} | "
            f"Minimum: f({self.minimum}) = {self.f_minimum:.6f} | "
            f"{self.n_iter} Iterationen)"
        )


class BaseOptimizer:
    """
    Basisklasse fuer alle Optimierer.

    Unterklassen muessen nur die Methode `step` implementieren.
    Den Rest (Schleife, Pfad speichern, Konvergenz pruefen) erledigt
    diese Klasse automatisch.

    Parameter
    ----------
    f        : die zu minimierende Funktion, f(x) -> float
    tol      : Konvergenzschwelle (Schrittweite)
    max_iter : maximale Anzahl Iterationen
    """

    name = "BaseOptimizer"

    def __init__(self, f, tol=1e-6, max_iter=1000):
        self.f = f
        self.tol = tol
        self.max_iter = max_iter

    def run(self, x0):
        """Startet die Optimierung vom Startpunkt x0."""
        x = np.array(x0, dtype=float)

        path = [x.copy()]
        f_path = [self.f(x)]

        success = False

        for i in range(self.max_iter):
            x_new = self.step(x)

            path.append(x_new.copy())
            f_path.append(self.f(x_new))

            # Konvergenz: Schrittweite kleiner als Toleranz
            if np.linalg.norm(x_new - x) < self.tol:
                success = True
                x = x_new
                break

            x = x_new

        return OptimizationResult(
            minimum=x,
            f_minimum=self.f(x),
            path=np.array(path),
            f_path=np.array(f_path),
            n_iter=len(path) - 1,
            success=success,
            algo_name=self.name,
        )

    def step(self, x):
        """
        Berechnet den naechsten Punkt ausgehend von x.
        Muss in jeder Unterklasse ueberschrieben werden!
        """
        raise NotImplementedError(f"{self.name} muss die Methode 'step' implementieren.")

    def gradient(self, x, eps=1e-5):
        """
        Berechnet den Gradienten von f an der Stelle x
        mit zentralen Differenzen (numerisch).
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = eps
            grad[i] = (self.f(x + e) - self.f(x - e)) / (2 * eps)
        return grad