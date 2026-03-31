# optim_vizu

Visualisierung von verschiedenen Optimierungsverfahren  
Es wurden insgesamt 4 Verfahren implementiert  

3 Verfahren die anhand von Steigungsinformationen (Ableitung)  
ein Optimierungsproblem lösen  

- Newton  
- Gradient Descent  
- BFGS  

Und außerdem das Nelder Mead Verfahren welches ohne Ableitung auskommt  

Man kann zum einen Funktionen explorativ mit den verschiedenen Verfahren untersuchen  
Außerdem kann man aber auch die Algorithmen gegeneinander antreten lassen  
(z.B. in Bezug auf Geschwindigkeit, Konvergenz und Genauigkeit)  

---

## Installation

Im Projektordner:

git clone https://github.com/maxi2048/optim_vizu.git
cd optiviz
uv pip install -e .

---

## Starten des Pakets

Das Paket wird über die Konsole gestartet mit:

uv run -m optim_vizu <command> [optionen]

Verfügbare Commands:

- help  
- optimize  
- compare  

---

## Grundlegende Nutzung

Es gibt zwei Hauptmöglichkeiten:

1) Eine Funktion mit einem bestimmten Verfahren optimieren  
2) Mehrere Verfahren auf derselben Funktion vergleichen  

---

## Optimierung starten

Beispiel:

uv run -m optim_vizu optimize --function sphere --method newton --x0 3,4

Dabei bedeutet:

--function sphere  
vordefinierte Funktion wird verwendet  

--method newton  
Optimierungsverfahren  

--x0 3,4  
Startpunkt  

---

## Eigene Funktion verwenden

Statt einer vordefinierten Funktion kann auch ein eigener Ausdruck angegeben werden:

uv run -m optim_vizu optimize --expr "(x[0]-2)**2 + (x[1]+1)**2" --x0 3,4 --method newton

Es darf immer nur eine der beiden Optionen gesetzt sein:

- --function  
- --expr  

---

## Verfügbare Funktionen

- sphere  
- shifted_sphere  
- rosenbrock  
- himmelblau  

---

## Verfügbare Methoden

- newton  
- gradient_descent  
- bfgs  
- neldermead  

---

## Startpunkt

Der Startpunkt wird mit --x0 angegeben:

--x0 "3,4"  
--x0 "1,-2,0.5"  

Die Dimension ergibt sich automatisch aus der Länge von x0  

---

## Bounds

Bounds können optional angegeben werden:

--bounds "-5,5"  
gleiche Bounds für alle Dimensionen  

--bounds "-5,5;-2,3;0,10"  
eigene Bounds pro Dimension  

---

## Multistart

Mehrere zufällige Startpunkte:

uv run -m optim_vizu optimize --function sphere --method neldermead --multistart --n-starts 5

---

## Vergleich von Verfahren

Alle Verfahren auf derselben Funktion:

uv run -m optim_vizu compare --function sphere --x0 3,4

---

## Plotoptionen

Plots können mit --plot aktiviert werden:

uv run -m optim_vizu optimize --function sphere --method newton --x0 3,4 --plot  

Die Art der Visualisierung hängt von der Dimension ab:

1 Dimension  
- Funktionsgraph  
- Optimierungspfad  
- Konvergenzplot  

2 Dimensionen  
- Konturplot der Funktion  
- Optimierungspfad im Raum  
- Konvergenzplot  

3 Dimensionen  
- 3D Scatter der Funktion  
- Optimierungspfad im Raum  
- Konvergenzplot  

Mehr als 3 Dimensionen  
- Pairplot aller 2D Projektionen der Variablen  
- Darstellung der Pfade in den jeweiligen Projektionen  

---

## Nutzung in Python

Das Paket kann auch direkt in Python verwendet werden:

import optim_vizu as ov

def f(x):
    return x[0]**2 + x[1]**2

bounds = [(-5, 5), (-5, 5)]

result = ov.optimize(f, bounds, method="newton", x0=[3, 4])

print(result.best)
result.plot()

---

## Ergebnisobjekt

Das Ergebnis enthält:

result.best  
bestes gefundenes Minimum  

result.optima  
alle gefundenen Minima  

result.f_optima  
Funktionswerte der Minima  

result.paths  
Optimierungspfade  

result.f_paths  
Konvergenzverläufe  

---

## Plotmethoden im Python-Modus

Zusätzlich stehen folgende Methoden zur Verfügung:

result.plot()  
Konturplot + Konvergenz  

result.plot_contour()  
nur Konturplot (2D)  

result.plot_convergence()  
nur Konvergenzverlauf  