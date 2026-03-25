import numpy as np
from base import BaseOptimizer, OptimizationResult
from newton import Newton
from neldermead import NelderMead
from class_Visualizer import Visualizer


METHODS = {
    "newton":     Newton,
    "neldermead": NelderMead,
}


# ===========================================================================
# Ergebnis-Klasse
# ===========================================================================

class OptimizeResult:
    """
    Ergebnis eines optimize()-Aufrufs.

    Attribute
    ----------
    results   : Liste von OptimizationResult (eines pro Start)
    best      : bestes OptimizationResult (kleinstes f_minimum)
    optima    : alle gefundenen Minima als Array, shape (n_starts, n_dims)
    f_optima  : f-Werte aller Optima, shape (n_starts,)
    paths     : alle Pfade als Liste von Arrays
    f_paths   : alle Konvergenzpfade als Liste von Arrays
    method    : genutzter Algorithmus als String
    """

    def __init__(self, results, f, bounds, method):
        # Einheitlich als Liste speichern
        if not isinstance(results, list):
            results = [results]

        self.results  = results
        self.f        = f
        self.bounds   = bounds
        self.method   = method

        # Bequeme Attribute
        self.best     = min(results, key=lambda r: r.f_minimum)
        self.optima   = np.array([r.minimum  for r in results])
        self.f_optima = np.array([r.f_minimum for r in results])
        self.paths    = [r.path   for r in results]
        self.f_paths  = [r.f_path for r in results]

    def __repr__(self):
        return (
            f"OptimizeResult({self.method} | "
            f"{len(self.results)} Lauf/Läufe | "
            f"bestes Minimum: f({self.best.minimum}) = {self.best.f_minimum:.6f})"
        )

    def plot_contour(self):
        """Konturplot mit allen Pfaden."""
        runs = [
            (r.path, r.f_path, f"{r.algo_name} Start {i+1}")
            for i, r in enumerate(self.results)
        ]
        viz = Visualizer(f=self.f, bounds=self.bounds)
        viz._plot_contour(runs)

    def plot_convergence(self):
        """Konvergenzplot fuer alle Laeufe."""
        runs = [
            (r.path, r.f_path, f"{r.algo_name} Start {i+1}")
            for i, r in enumerate(self.results)
        ]
        viz = Visualizer(f=self.f, bounds=self.bounds)
        viz._plot_convergence(runs)

    def plot(self):
        """Konturplot und Konvergenzplot zusammen."""
        runs = [
            (r.path, r.f_path, f"{r.algo_name} Start {i+1}")
            for i, r in enumerate(self.results)
        ]
        viz = Visualizer(f=self.f, bounds=self.bounds)
        viz.plot(runs)


# ===========================================================================
# Hauptfunktion
# ===========================================================================

def optimize(f, bounds, method="newton", x0=None, plot=False,
             multistart=False, n_starts=8, **kwargs):
    """
    Fuehrt eine Optimierung durch und gibt ein OptimizeResult zurueck.

    Parameter
    ----------
    f          : zu minimierende Funktion
    bounds     : Suchraum, z.B. [(-5, 5), (-5, 5)]
    method     : "newton" oder "neldermead"
    x0         : Startpunkt oder Liste von Startpunkten bei multistart
    plot       : ob automatisch geplottet werden soll
    multistart : ob mehrere Starts genutzt werden sollen
    n_starts   : Anzahl Starts bei multistart
    **kwargs   : weitere Parameter fuer den Algorithmus (tol, max_iter, ...)

    Beispiele
    ---------
    result = optiviz.optimize(f, bounds, method="newton", x0=[0, 0])
    print(result.best)
    print(result.optima)
    result.plot_contour()
    result.plot_convergence()
    """

    if method not in METHODS:
        raise ValueError(f"Unbekannte Methode '{method}'. Verfuegbar: {list(METHODS.keys())}")

    algo = METHODS[method](f, bounds, multistart=multistart, n_starts=n_starts, **kwargs)
    raw_results = algo.run(x0=x0)

    result = OptimizeResult(raw_results, f=f, bounds=bounds, method=method)

    if plot:
        result.plot()

    return result


# ===========================================================================
# Vergleichsfunktion
# ===========================================================================

def compare(f, bounds, x0=None, plot=True, **kwargs):
    """
    Fuehrt alle verfuegbaren Algorithmen auf derselben Funktion aus
    und plottet sie zusammen.
    """
    all_runs = []
    results = {}

    for name, algo_class in METHODS.items():
        algo = algo_class(f, bounds, **kwargs)
        raw = algo.run(x0=x0)
        results[name] = OptimizeResult(raw, f=f, bounds=bounds, method=name)

        for i, r in enumerate(results[name].results):
            all_runs.append((r.path, r.f_path, f"{name} Start {i+1}"))

    if plot:
        viz = Visualizer(f=f, bounds=bounds)
        viz.plot(all_runs)

    return results

Anleitung = {"""
1) OPTIMIERUNG STARTEN
   import optiviz
 
   Zu optimierende Objekte müssen immer als Funktion erstellt werden
   -> x ist hier mehrdimensional
 
   def meine_funktion(x):
       return x[0]**2 + x[1]**2
 
   bounds = [(-5, 5), (-5, 5)]
 
   result = optiviz.optimize(meine_funktion, bounds, method="newton", x0=[3, 4])
 
2) VERFUEGBARE METHODEN
   Für verschiedene Arten von Funktionen sind unterschiedliche Verfahren gut (Je nach Form)
   method="newton"     — Newton-Verfahren (schnell, braucht Gradient)
   method="neldermead" — Nelder-Mead (kein Gradient noetig)
 
3) MULTISTART
   Wir wollen sicher gehen das echte Optimum zu finden daher starten wir häufiger 
   result = optiviz.optimize(f, bounds, method="neldermead",
                              multistart=True, n_starts=8)
               
 
4) ERGEBNIS-ATTRIBUTE
   result.best         — bestes gefundenes Optimum
   result.optima       — alle gefundenen Minima (Array)
   result.f_optima     — f-Werte aller Optima
   result.paths        — alle Pfade
   result.f_paths      — alle Konvergenzpfade
 
5) PLOTS
   result.plot()            — Konturplot + Konvergenz zusammen
   result.plot_contour()    — nur Konturplot
   result.plot_convergence() — nur Konvergenzplot
 
6) ALGORITHMEN VERGLEICHEN
   optiviz.compare(f, bounds, x0=[0, 0])
"""}

def help():
    print(Anleitung)