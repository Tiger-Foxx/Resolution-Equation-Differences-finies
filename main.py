import numpy as np
import matplotlib.pyplot as plt


def resoudre_equation_diff(f, N, U0, U1, tracer_graphe=False):
    """
    Résout l'équation différentielle -U''(x) = f(x) avec les conditions aux limites
    U(0) = U0 et U(1) = U1 par la méthode des différences finies.

    Paramètres:
    -----------
    f : fonction
        Terme source f(x) dans l'équation -U''(x) = f(x)
    N : int
        Nombre de subdivisions de l'intervalle [0,1]
    U0 : float
        Condition à la limite en x=0
    U1 : float
        Condition à la limite en x=1
    tracer_graphe : bool, optionnel
        Si True, trace le graphe de la solution

    Retourne:
    ---------
    U : array
        Tableau contenant la solution complète (y compris les conditions limites)
    x : array
        Points de discrétisation correspondants
    """
    if N <= 1:
        raise ValueError("N doit être supérieur à 1")

    h = 1 / N  #pas
    x_interieur = np.linspace(0, 1, N + 1)[1:-1]  #(de x1 à x_{N-1})

    A = np.zeros((N - 1, N - 1))
    b = np.zeros(N - 1)

    for i in range(N - 1):
        if i == 0:
            A[i, i] = 2
            A[i, i + 1] = -1
            b[i] = h ** 2 * f(x_interieur[i]) + U0
        elif i == N - 2:
            A[i, i] = 2
            A[i, i - 1] = -1
            b[i] = h ** 2 * f(x_interieur[i]) + U1
        else:
            A[i, i - 1] = -1
            A[i, i] = 2
            A[i, i + 1] = -1
            b[i] = h ** 2 * f(x_interieur[i])

    try:
        U_interieur = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise RuntimeError("Impossible de résoudre le système linéaire. La matrice est peut-être singulière.")

    #On construit la soluce complete
    x = np.linspace(0, 1, N + 1)
    U = np.zeros(N + 1)
    U[0] = U0
    U[1:-1] = U_interieur
    U[-1] = U1
    if tracer_graphe:
        plt.figure(figsize=(10, 6))
        plt.plot(x, U, 'b-', linewidth=2)
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('U(x)')
        plt.title('Solution de l\'équation -U\'\'(x) = f(x)')
        plt.show()

    return U, x