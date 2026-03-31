import argparse
import sys
import numpy as np
from . import optiviz


# ============================================================
# Testfunktionen fuer die CLI
# ============================================================

def sphere(x):
    x = np.asarray(x, dtype=float)
    return np.sum(x ** 2)


def shifted_sphere(x):
    x = np.asarray(x, dtype=float)
    return np.sum((x - 1.0) ** 2)


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("rosenbrock braucht mindestens 2 Dimensionen.")
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def himmelblau(x):
    x = np.asarray(x, dtype=float)
    if len(x) != 2:
        raise ValueError("himmelblau ist nur fuer 2 Dimensionen definiert.")
    x0, x1 = x
    return (x0 ** 2 + x1 - 11) ** 2 + (x0 + x1 ** 2 - 7) ** 2


FUNCTIONS = {
    "sphere": sphere,
    "shifted_sphere": shifted_sphere,
    "rosenbrock": rosenbrock,
    "himmelblau": himmelblau,
}

METHODS = ["newton", "neldermead", "bfgs", "gradient_descent"]


# ============================================================
# Hilfetexte / Beispiele
# ============================================================

GENERAL_EXAMPLES = """
Beispiele:
  uv run -m optim_vizu optimize --function sphere --method newton --x0 3,4 --plot
  uv run -m optim_vizu optimize --function himmelblau --method neldermead --x0 0,0 --plot
  uv run -m optim_vizu optimize --function rosenbrock --dim 3 --method bfgs --x0 1.2,1.2,1.2
  uv run -m optim_vizu optimize --function sphere --dim 4 --method gradient_descent --x0 4,3,2,1 --lr 0.05
  uv run -m optim_vizu optimize --expr "(x[0]-2)**2 + (x[1]+1)**2" --x0 3,4 --method newton --plot
  uv run -m optim_vizu optimize --expr "np.sin(x[0]) + x[1]**2" --x0 1,2 --method neldermead --plot
  uv run -m optim_vizu compare --function sphere --dim 2 --x0 3,4 --plot
  uv run -m optim_vizu compare --expr "(x[0]-1)**2 + (x[1]-2)**2 + (x[2]+3)**2" --x0 4,0,1 --plot

Eingabeformate:
  --x0 "3,4"
  --x0 "1,-2,0.5"

  --bounds "-5,5"
    -> gleiche bounds fuer alle Dimensionen

  --bounds "-5,5;-2,3;0,10"
    -> eigene bounds pro Dimension

  --expr "(x[0]-2)**2 + (x[1]+1)**2"
  --expr "np.sin(x[0]) + x[1]**2"
"""

# ============================================================
# Eigener Parser: bei Fehler immer Usage + Beispiele
# ============================================================

class FriendlyArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"\nFehler: {message}\n\n")
        self.print_help(sys.stderr)
        sys.stderr.write("\n" + GENERAL_EXAMPLES + "\n")
        raise SystemExit(2)


# ============================================================
# Parser-Funktionen
# ============================================================

def parse_vector(text):
    """
    '3,4' -> np.array([3.0, 4.0])
    """
    try:
        parts = [p.strip() for p in text.split(",") if p.strip() != ""]
        if not parts:
            raise ValueError
        return np.array([float(p) for p in parts], dtype=float)
    except Exception:
        raise ValueError(
            f"x0 konnte nicht gelesen werden: '{text}'. "
            "Erwartet z.B. '3,4' oder '1,-2,0.5'."
        )


def parse_bounds(text, dim):
    """
    Erlaubt:
      '-5,5'
      '-5,5;-2,3;0,10'
    """
    try:
        text = text.strip()

        if ";" not in text:
            lo, hi = [float(v.strip()) for v in text.split(",")]
            if lo >= hi:
                raise ValueError("Bei bounds muss low < high gelten.")
            return [(lo, hi) for _ in range(dim)]

        bounds = []
        chunks = [chunk.strip() for chunk in text.split(";") if chunk.strip() != ""]
        for chunk in chunks:
            lo, hi = [float(v.strip()) for v in chunk.split(",")]
            if lo >= hi:
                raise ValueError("Bei bounds muss low < high gelten.")
            bounds.append((lo, hi))

        if len(bounds) != dim:
            raise ValueError(
                f"Anzahl der Bounds ({len(bounds)}) passt nicht zu dim={dim}."
            )

        return bounds

    except Exception as e:
        raise ValueError(
            f"Bounds konnten nicht gelesen werden: '{text}'. "
            "Erwartet z.B. '-5,5' oder '-5,5;-2,3;0,10'. "
            f"Details: {e}"
        )


def validate_function_and_dim(function_name, dim):
    if dim < 1:
        raise ValueError("dim muss mindestens 1 sein.")

    if function_name == "himmelblau" and dim != 2:
        raise ValueError("Die Funktion 'himmelblau' ist nur fuer dim=2 erlaubt.")

    if function_name == "rosenbrock" and dim < 2:
        raise ValueError("Die Funktion 'rosenbrock' braucht mindestens dim=2.")


def make_expr_function(expr):
    """
    Baut aus einem String-Ausdruck eine Funktion f(x).

    Beispiele:
        "(x[0]-2)**2 + (x[1]+1)**2"
        "np.sin(x[0]) + x[1]**2"
    """
    allowed_names = {
        "np": np,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
    }

    def f(x):
        local_names = {"x": np.asarray(x, dtype=float)}
        return eval(expr, {"__builtins__": {}}, {**allowed_names, **local_names})

    return f


def resolve_function(args):
    """
    Genau eine der beiden Optionen muss gesetzt sein:
    - --function
    - --expr
    """
    if args.function is None and args.expr is None:
        raise ValueError(
            "Bitte entweder --function oder --expr angeben.\n"
            'Beispiel: --function sphere\n'
            'Beispiel: --expr "(x[0]-2)**2 + (x[1]+1)**2"'
        )

    if args.function is not None and args.expr is not None:
        raise ValueError(
            "Bitte nur eine von beiden Optionen angeben: --function ODER --expr."
        )

    if args.expr is not None:
        return make_expr_function(args.expr), "expr"

    return FUNCTIONS[args.function], args.function


# ============================================================
# Parser bauen
# ============================================================

def build_parser():
    parser = FriendlyArgumentParser(
        prog="optiviz",
        description="CLI fuer optiviz: Optimierungsverfahren ueber die Konsole starten.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=GENERAL_EXAMPLES,
    )

    subparsers = parser.add_subparsers(dest="command")

    parser_opt = subparsers.add_parser(
        "optimize",
        help="Eine Methode auf einer Funktion ausfuehren.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=GENERAL_EXAMPLES,
    )
    add_common_arguments(parser_opt)
    parser_opt.add_argument(
        "--method",
        required=True,
        choices=METHODS,
        help="Optimierungsmethode."
    )
    parser_opt.add_argument(
        "--multistart",
        action="store_true",
        help="Mehrere zufaellige Startpunkte verwenden."
    )
    parser_opt.add_argument(
        "--n-starts",
        type=int,
        default=8,
        help="Anzahl Starts bei multistart."
    )
    parser_opt.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Toleranz."
    )
    parser_opt.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximale Iterationen."
    )
    parser_opt.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Lernrate fuer gradient_descent oder bfgs."
    )
    parser_opt.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Initiale Simplex-Groesse fuer neldermead."
    )

    parser_cmp = subparsers.add_parser(
        "compare",
        help="Alle Methoden auf derselben Funktion vergleichen.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=GENERAL_EXAMPLES,
    )
    add_common_arguments(parser_cmp)
    parser_cmp.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Toleranz."
    )
    parser_cmp.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximale Iterationen."
    )

    subparsers.add_parser(
        "help",
        help="Zeigt die Hilfe."
    )

    return parser


def add_common_arguments(parser):
    parser.add_argument(
        "--function",
        choices=FUNCTIONS.keys(),
        default=None,
        help="Zu optimierende vordefinierte Funktion."
    )
    parser.add_argument(
        "--expr",
        type=str,
        default=None,
        help='Funktionsausdruck als String, z.B. "(x[0]-2)**2 + (x[1]+1)**2"'
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Dimension. Falls --x0 gegeben ist, kann dim daraus bestimmt werden."
    )
    parser.add_argument(
        "--x0",
        type=str,
        default=None,
        help='Startpunkt, z.B. "3,4" oder "1,-2,0.5".'
    )
    parser.add_argument(
        "--bounds",
        type=str,
        default="-5,5",
        help='Bounds, z.B. "-5,5" oder "-5,5;-2,3;0,10".'
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot anzeigen."
    )


# ============================================================
# Logik
# ============================================================

def resolve_dimension(args):
    """
    dim bestimmen:
    1) wenn x0 vorhanden -> len(x0)
    2) sonst args.dim
    """
    if args.x0 is not None:
        x0 = parse_vector(args.x0)
        inferred_dim = len(x0)

        if args.dim is not None and args.dim != inferred_dim:
            raise ValueError(
                f"dim={args.dim} passt nicht zu x0 mit Laenge {inferred_dim}."
            )
        return inferred_dim, x0

    if args.dim is None:
        raise ValueError(
            "Bitte entweder --x0 oder --dim angeben. "
            "Ohne x0 kann die Dimension nicht automatisch bestimmt werden."
        )

    return args.dim, None


def run_optimize(args):
    dim, x0 = resolve_dimension(args)
    f, function_name = resolve_function(args)

    if args.function is not None:
        validate_function_and_dim(args.function, dim)

    bounds = parse_bounds(args.bounds, dim)

    if args.multistart and args.n_starts < 1:
        raise ValueError("--n-starts muss mindestens 1 sein.")

    kwargs = {
        "tol": args.tol,
        "max_iter": args.max_iter,
    }

    if args.lr is not None:
        kwargs["lr"] = args.lr

    if args.step_size is not None:
        kwargs["step_size"] = args.step_size

    result = optiviz.optimize(
        f=f,
        bounds=bounds,
        method=args.method,
        x0=x0,
        plot=args.plot,
        multistart=args.multistart,
        n_starts=args.n_starts,
        **kwargs,
    )

    print("\n=== OPTIMIERUNG ABGESCHLOSSEN ===")
    print(result)
    print("Methode:         ", args.method)
    print("Funktion:        ", function_name if args.expr is None else args.expr)
    print("Bestes Minimum:  ", result.best.minimum)
    print("Funktionswert:   ", result.best.f_minimum)
    print("Iterationen:     ", result.best.n_iter)
    print("Erfolg:          ", result.best.success)

    if args.multistart:
        print("\nAlle gefundenen Optima:")
        print(result.optima)
        print("\nAlle Funktionswerte:")
        print(result.f_optima)


def run_compare(args):
    dim, x0 = resolve_dimension(args)
    f, function_name = resolve_function(args)

    if args.function is not None:
        validate_function_and_dim(args.function, dim)

    bounds = parse_bounds(args.bounds, dim)

    results = optiviz.compare(
        f=f,
        bounds=bounds,
        x0=x0,
        plot=args.plot,
        tol=args.tol,
        max_iter=args.max_iter,
    )

    print("\n=== VERGLEICH ABGESCHLOSSEN ===")
    print("Funktion:", function_name if args.expr is None else args.expr)
    print("Dimension:", dim)

    for method_name, result in results.items():
        print(f"\n--- {method_name} ---")
        print(result)
        print("Bestes Minimum: ", result.best.minimum)
        print("Funktionswert:  ", result.best.f_minimum)
        print("Iterationen:    ", result.best.n_iter)
        print("Erfolg:         ", result.best.success)


def main():
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + GENERAL_EXAMPLES)
        raise SystemExit(0)

    args = parser.parse_args()

    try:
        if args.command == "help":
            parser.print_help()
            print("\n" + GENERAL_EXAMPLES)
            return

        if args.command == "optimize":
            run_optimize(args)
            return

        if args.command == "compare":
            run_compare(args)
            return

        parser.print_help()
        print("\n" + GENERAL_EXAMPLES)
        raise SystemExit(1)

    except ValueError as e:
        print(f"\nFehler: {e}\n")
        parser.print_help()
        print("\n" + GENERAL_EXAMPLES)
        raise SystemExit(2)

    except Exception as e:
        print(f"\nUnerwarteter Fehler: {e}\n")
        parser.print_help()
        print("\n" + GENERAL_EXAMPLES)
        raise SystemExit(2)


if __name__ == "__main__":
    main()