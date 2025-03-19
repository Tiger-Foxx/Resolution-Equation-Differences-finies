# Résolution d'Équations Différentielles par Différences Finies

Ce projet implémente une méthode numérique pour résoudre des équations différentielles ordinaires (EDO) du type `-U''(x) = f(x)` avec conditions aux limites de Dirichlet. Il a été réalisé dans le cadre du cours d'analyse numérique du Master 1 Génie Informatique à l'École Polytechnique.

## 📝 Description du problème

Nous résolvons numériquement l'équation différentielle suivante :
```
-U''(x) = f(x)
```
avec les conditions aux limites de Dirichlet :
- U(0) = U₀
- U(1) = U₁

sur l'intervalle [0,1].

## 🧮 Méthode des différences finies

La méthode utilisée discrétise l'intervalle [0,1] en N+1 points équidistants. Pour chaque point intérieur xᵢ, l'équation différentielle est approchée par :

```
-U(i-1) + 2U(i) - U(i+1) = h²·f(xᵢ)
```

où h = 1/N est le pas de discrétisation. Ceci conduit à un système linéaire AX = b qui est résolu par le code.

## 📁 Organisation du code

Le projet est composé de trois fichiers principaux :

- `main.py` : Contient la fonction `resoudre_equation_diff()` qui implémente la méthode des différences finies
- `analyse_convergence.py` : Module d'analyse de la convergence qui génère des graphiques et calcule l'ordre de convergence
- `utilisation.py` : Fichier de démonstration avec des exemples simples (à utiliser pour la présentation au professeur)

## 📊 Fonctionnalités implémentées

1. **Résolution de l'équation différentielle** avec la méthode des différences finies
2. **Visualisation** de la solution numérique vs solution exacte
3. **Analyse de convergence** avec calcul d'erreur en norme L∞
4. **Export des résultats** sous format graphique, CSV et TXT

## 📈 Analyse de convergence

Le code étudie la vitesse de convergence en calculant l'erreur pour différentes tailles de maillage (N = 10, 20, 40, 80, 160, 320) sur plusieurs cas tests :
- u(x) = sin(πx)
- u(x) = x³
- u(x) = log(x+1)

Les résultats sont visualisés sous forme de graphiques et l'ordre de convergence est calculé. L'ordre théorique attendu est 2.

## 🚀 Utilisation

### Prérequis

- Python 3.x
- NumPy
- Matplotlib

### Installation des dépendances

```bash
pip install numpy matplotlib
```

### Exécution

Pour une démonstration simple avec visualisation des solutions :
```bash
python utilisation.py
```

Pour lancer l'analyse complète de convergence :
```bash
python analyse_convergence.py
```

## 👥 Tâches pour les membres de l'équipe

### Pour tous :
1. Comprendre la méthode des différences finies et le code implémenté
2. Être capable d'expliquer l'organisation du projet et les résultats obtenus


## 📝 Résultats attendus

Pour la présentation au professeur, nous devons être capables de :
1. Expliquer la méthode mathématique utilisée
2. Montrer les graphiques de convergence pour chaque cas test
3. Analyser l'ordre de convergence numérique et le comparer à la théorie (ordre 2)
4. Discuter des avantages et limitations de la méthode

## 📌 Points importants pour la présentation

- La méthode converge-t-elle à l'ordre attendu ? (oui pour sinus mais pour une raison inconnue x^3 presente des etrangetes)
- Comment évolue l'erreur quand on raffine le maillage ? (elle dinimue)


## 📄 Licence

Ce projet est réalisé à des fins éducatives dans le cadre du cours d'analyse numérique à l'École Polytechnique.