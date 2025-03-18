# Résolution d'équations différentielles par différences finies

Ce projet a été réalisé dans le cadre du cours d'analyse numérique du Master 1 Génie Informatique à l'École Polytechnique.

## Description

Ce code implémente la méthode des différences finies pour résoudre numériquement des équations différentielles ordinaires (EDO) de la forme :

```
-U''(x) = f(x)
```

avec les conditions aux limites de Dirichlet :
- U(0) = U₀
- U(1) = U₁

sur l'intervalle [0,1].

## Contenu du repository

- `resolution_edo.py` : Module principal contenant la fonction de résolution
- `tests.py` : Fichier contenant les tests de validation et d'analyse de convergence
- `rapport.pdf` : Rapport d'analyse des résultats et de la convergence

## Méthode utilisée

La méthode des différences finies sur un maillage uniforme est utilisée pour discrétiser l'équation différentielle. Le schéma de discrétisation conduit à un système linéaire de la forme :

```
-U(i-1) + 2U(i) - U(i+1) = h²·f(xᵢ)
```

pour i allant de 1 à N-1, où h = 1/N est le pas de discrétisation.

## Analyse de convergence

Le code permet d'étudier la vitesse numérique de convergence en calculant l'erreur en norme L∞ pour différentes tailles de maillage (N = 10, 20, 40, 80, 160, 320) sur deux cas test :
- u(x) = sin(πx)
- u(x) = x³

## Utilisation

Pour utiliser ce code, importez le module `resolution_edo.py` et appelez la fonction `resoudre_equation_diff` avec les paramètres appropriés.

### Prérequis

- Python 3.x
- NumPy
- Matplotlib

### Installation

```bash
pip install numpy matplotlib
```

## Licence

Ce projet est réalisé à des fins éducatives dans le cadre du cours d'analyse numérique à l'École Polytechnique.