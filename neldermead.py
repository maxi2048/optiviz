import numpy as np
from base import BaseOptimizer, OptimizationResult


class NelderMead(BaseOptimizer):
    """

    Iterativ, aber nicht auf Basis von Punkten, sondern auf n+1 (Bei n dimensionen) Punkten
    Parameter
    ----------
    f          : zu minimierende Funktion
    bounds     : Suchraum, z.B. [(-5, 5), (-5, 5)]
    multistart : ob mehrere Startpunkte genutzt werden sollen
    n_starts   : Anzahl zufaelliger Startpunkte
    tol        : Konvergenzschwelle
    max_iter   : maximale Anzahl Iterationen
    step_size  : initiale Groesse des Simplex
    """

    name = "NelderMead"

    def __init__(self, f, bounds, multistart=False, n_starts=8,
                 tol=1e-6, max_iter=1000, step_size=0.5):
        super().__init__(f, bounds, multistart, n_starts, tol, max_iter)
        self.step_size = step_size

    def _run_single(self, x0):
        x0 = np.array(x0, dtype=float)
        n = len(x0)

        # Simplex initialisieren
        simplex = [x0.copy()]
        for i in range(n):
            punkt = x0.copy()
            punkt[i] += self.step_size
            simplex.append(punkt)
        simplex = np.array(simplex)

        path = [x0.copy()]
        f_path = [self.f(x0)]

        success = False

        for iteration in range(self.max_iter):

            # Simplex sortieren: bester Punkt zuerst
            werte = np.array([self.f(p) for p in simplex])
            reihenfolge = np.argsort(werte)
            simplex = simplex[reihenfolge]
            werte = werte[reihenfolge]

            bester = simplex[0]
            schlechtester = simplex[-1]
            schwerpunkt = np.mean(simplex[:-1], axis=0)

            # Besten Punkt im Pfad speichern
            path.append(bester.copy())
            f_path.append(werte[0])

            # Konvergenz: alle Punkte nah am besten
            if np.max(np.abs(simplex - bester)) < self.tol:
                success = True
                break

            # Reflexion
            reflexion = schwerpunkt + 1.0 * (schwerpunkt - schlechtester)
            f_reflexion = self.f(reflexion)

            if f_reflexion < werte[0]:
                # Besser als bester -> Expansion
                expansion = schwerpunkt + 2.0 * (reflexion - schwerpunkt)
                if self.f(expansion) < f_reflexion:
                    simplex[-1] = expansion
                else:
                    simplex[-1] = reflexion

            elif f_reflexion < werte[-2]:
                # Besser als zweitschlechtester -> akzeptieren
                simplex[-1] = reflexion

            else:
                # Schlechter -> Kontraktion
                kontraktion = schwerpunkt + 0.5 * (schlechtester - schwerpunkt)
                if self.f(kontraktion) < werte[-1]:
                    simplex[-1] = kontraktion
                else:
                    # Kontraktion auch schlecht -> Schrumpfung
                    for i in range(1, len(simplex)):
                        simplex[i] = bester + 0.5 * (simplex[i] - bester)

        return OptimizationResult(
            minimum=simplex[0],
            f_minimum=self.f(simplex[0]),
            path=np.array(path),
            f_path=np.array(f_path),
            n_iter=len(path) - 1,
            success=success,
            algo_name=self.name,
        )

    def step(self, x):
        raise NotImplementedError("NelderMead nutzt _run_single direkt.")