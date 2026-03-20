import numpy as np


# ===========================================================================
# Ergebnis-Container
# ===========================================================================

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


# ===========================================================================
# Basisklasse
# ===========================================================================

class BaseOptimizer:
    """
    Basisklasse fuer alle Optimierer.

    Unterklassen muessen nur die Methode `step` implementieren.

    Parameter
    ----------
    f          : die zu minimierende Funktion, f(x) -> float
    bounds     : Suchraum, z.B. [(-5, 5), (-5, 5)] fuer 2D
    multistart : ob mehrere Startpunkte genutzt werden sollen
    n_starts   : Anzahl zufaelliger Startpunkte (nur wenn multistart=True)
    tol        : Konvergenzschwelle (Schrittweite)
    max_iter   : maximale Anzahl Iterationen
    """

    name = "BaseOptimizer"

    def __init__(self, f, bounds, multistart=False, n_starts=8, tol=1e-6, max_iter=1000):
        self.f = f
        self.bounds = bounds
        self.multistart = multistart
        self.n_starts = n_starts
        self.tol = tol
        self.max_iter = max_iter

    def run(self, x0=None):
        """
        Startet die Optimierung.

        multistart=False:
            x0 = None          -> zufaelliger Startpunkt
            x0 = [0, 0]        -> genau dieser Startpunkt

        multistart=True:
            x0 = None                      -> n_starts zufaellige Punkte
            x0 = [[0,0], [1,1], [2,2]]     -> genau diese Startpunkte

        Gibt zurueck
        ------------
        multistart=False : ein einzelnes OptimizationResult
        multistart=True  : Liste von OptimizationResult (eines pro Start)
        """
        if self.multistart:
            if x0 is None:
                # keine Startpunkte angegeben -> zufaellig
                starts = self._random_starts(self.n_starts)
            else:
                # x0 ist eine Liste von Startpunkten
                starts = [np.array(s, dtype=float) for s in x0]
            return [self._run_single(s) for s in starts]
        else:
            if x0 is None:
                x0 = self._random_starts(1)[0]
            return self._run_single(x0)

    def _run_single(self, x0):
        """Fuehrt einen einzelnen Optimierungslauf durch."""
        x = np.array(x0, dtype=float)

        path = [x.copy()]
        f_path = [self.f(x)]

        success = False

        for i in range(self.max_iter):
            x_new = self.step(x)
            path.append(x_new.copy())
            f_path.append(self.f(x_new))

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

    def _random_starts(self, n):
        """Erzeugt n zufaellige Startpunkte innerhalb von bounds."""
        starts = []
        for _ in range(n):
            x0 = np.array([
                np.random.uniform(low, high)
                for (low, high) in self.bounds
            ])
            starts.append(x0)
        return starts

    def step(self, x):
        """Muss in jeder Unterklasse ueberschrieben werden!"""
        raise NotImplementedError(f"{self.name} muss die Methode 'step' implementieren.")
