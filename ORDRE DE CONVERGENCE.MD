# Explication de la vitesse numérique de convergence

## Qu'est-ce que la vitesse numérique de convergence ?

La vitesse numérique de convergence mesure à quelle rapidité une méthode numérique se rapproche de la solution exacte lorsqu'on raffine le maillage (en augmentant le nombre de points N).

Pour les méthodes de différences finies, on s'intéresse généralement à comment l'erreur diminue quand on augmente N. Dans le cas de notre schéma, nous utilisons des différences finies centrées de second ordre pour approximer la dérivée seconde, ce qui théoriquement devrait donner une convergence d'ordre 2.

## Erreur en norme L∞

La norme L∞ (ou norme du maximum) est définie comme la valeur maximale absolue de l'erreur sur tout le domaine :

```
||e||_∞ = max |u_exacte(x_i) - u_numerique(x_i)|
```

où l'erreur est calculée sur tous les points de discrétisation x_i.

Cette norme est particulièrement utile car elle nous donne la pire erreur commise en n'importe quel point du domaine.

## Ordre numérique de convergence

L'ordre numérique de convergence, noté p, caractérise la relation entre l'erreur et le pas de discrétisation h = 1/N. Pour une méthode d'ordre p, on a théoriquement :

```
||e||_∞ ≈ C·h^p
```

où C est une constante.

En pratique, on détermine p expérimentalement en calculant l'erreur pour différentes valeurs de N. Si l'on prend le logarithme de l'équation précédente :

```
log(||e||_∞) ≈ log(C) + p·log(h)
```

Donc, pour deux maillages consécutifs de pas h₁ et h₂, l'ordre p peut être estimé par :

```
p ≈ log(||e₁||_∞/||e₂||_∞) / log(h₁/h₂)
```

## Interprétation des résultats

Pour notre schéma de différences finies centrées, nous nous attendons théoriquement à un ordre de convergence p ≈ 2. Cela signifie que si l'on double le nombre de points (N → 2N), l'erreur devrait être divisée par 4.

L'analyse de convergence dans le code permet de vérifier si cette prédiction théorique se confirme en pratique. Si l'ordre calculé est proche de 2, cela valide notre implémentation et confirme que la méthode converge comme attendu.

Des différences entre l'ordre théorique et calculé peuvent s'expliquer par :
- Des erreurs d'arrondi numérique
- Des particularités de la solution exacte (régularité, singularités)
- Des effets de bord aux frontières du domaine

Les graphiques de convergence en échelle logarithmique nous aident à visualiser cette relation : une pente de -2 dans le graphe log(erreur) vs log(N) confirme un ordre de convergence de 2.