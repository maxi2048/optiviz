import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, f, bounds):
        self.f = f
        self.bounds = bounds
        self.ndim = len(bounds)

    def plot(self, runs):
        """
        runs: Liste von Tupeln (path, values, label)
        path:   np.array shape (n_steps, ndim)
        values: np.array shape (n_steps,)
        label:  string z.B. "Newton"
        """
        if self.ndim == 1:
            self._plot_1d(runs)
        elif self.ndim == 2:
            self._plot_2d(runs)
        else:
            self._plot_pairplot(runs)

    def _plot_1d(self, runs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_f, ax_conv = axes

        # Funktion plotten
        lo, hi = self.bounds[0]
        xs = np.linspace(lo, hi, 300)
        ys = [self.f(np.array([x])) for x in xs]
        ax_f.plot(xs, ys, color='gray', lw=2, label='f(x)')
        ax_f.set_xlabel('x')
        ax_f.set_ylabel('f(x)')
        ax_f.set_title('Funktion & Pfad')

        # Pfade plotten
        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]
        for i, (path, values, label) in enumerate(runs):
            color = colors[i]
            px = path[:, 0]
            ax_f.scatter(px, values, color=color, s=30, zorder=3)
            ax_f.plot(px, values, color=color, lw=1, alpha=0.5)
            ax_f.scatter(px[0], values[0], color=color, s=100, marker='^', zorder=4, label=f'{label} Start')
            ax_f.scatter(px[-1], values[-1], color=color, s=100, marker='*', zorder=4, label=f'{label} Ende')

            # Konvergenzkurve
            ax_conv.plot(values, color=color, lw=2, label=label)

        ax_f.legend()
        ax_conv.set_xlabel('Iteration')
        ax_conv.set_ylabel('f(x)')
        ax_conv.set_title('Konvergenz')
        ax_conv.legend()
        plt.tight_layout()
        plt.show()

    def _plot_2d(self, runs):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_contour, ax_conv = axes

        # Konturplot erstellen
        lo0, hi0 = self.bounds[0]
        lo1, hi1 = self.bounds[1]
        xs = np.linspace(lo0, hi0, 300)
        ys = np.linspace(lo1, hi1, 300)
        X, Y = np.meshgrid(xs, ys)
        Z = np.vectorize(lambda u, v: self.f(np.array([u, v])))(X, Y)

        ax_contour.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        ax_contour.contour(X, Y, Z, levels=50, colors='white', linewidths=0.3, alpha=0.4)
        ax_contour.set_xlabel('x0')
        ax_contour.set_ylabel('x1')
        ax_contour.set_title('Konturplot & Pfad')

        # Pfade plotten
        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]
        for i, (path, values, label) in enumerate(runs):
            color = colors[i]
            px = path[:, 0]
            py = path[:, 1]
            ax_contour.plot(px, py, color=color, lw=1.5, alpha=0.8)
            ax_contour.scatter(px, py, color=color, s=15, alpha=0.6)
            ax_contour.scatter(px[0], py[0], color=color, s=120, marker='^', zorder=5, label=f'{label} Start')
            ax_contour.scatter(px[-1], py[-1], color=color, s=150, marker='*', zorder=5, label=f'{label} Ende')

            # Konvergenzkurve
            ax_conv.plot(values, color=color, lw=2, label=label)

        ax_contour.legend()
        ax_conv.set_xlabel('Iteration')
        ax_conv.set_ylabel('f(x)')
        ax_conv.set_title('Konvergenz')
        ax_conv.legend()
        plt.tight_layout()
        plt.show()

    def _plot_contour(self, runs):
        """Nur der Konturplot ohne Konvergenzplot."""
        lo0, hi0 = self.bounds[0]
        lo1, hi1 = self.bounds[1]
        xs = np.linspace(lo0, hi0, 300)
        ys = np.linspace(lo1, hi1, 300)
        X, Y = np.meshgrid(xs, ys)
        Z = np.vectorize(lambda u, v: self.f(np.array([u, v])))(X, Y)

        fig, ax = plt.subplots(figsize=(7, 6))
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, Z, levels=50, colors='white', linewidths=0.3, alpha=0.4)
        plt.colorbar(contour, ax=ax, label='f(x, y)')

        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]
        for i, (path, values, label) in enumerate(runs):
            color = colors[i]
            px = path[:, 0]
            py = path[:, 1]
            ax.plot(px, py, color=color, lw=1.5, alpha=0.8)
            ax.scatter(px[0], py[0], color=color, s=120, marker='^', zorder=5, label=f'{label} Start')
            ax.scatter(px[-1], py[-1], color=color, s=150, marker='*', zorder=5, label=f'{label} Ende')

        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_title('Konturplot & Pfad')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    def _plot_convergence(self, runs):
        """Nur der Konvergenzplot."""
        fig, ax = plt.subplots(figsize=(7, 4))

        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]
        for i, (path, values, label) in enumerate(runs):
            ax.plot(values, color=colors[i], lw=2, label=label)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('f(x)')
        ax.set_title('Konvergenzverlauf')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_pairplot(self, runs):
        print(f"Pairplot fuer {self.ndim}D noch nicht implementiert.")