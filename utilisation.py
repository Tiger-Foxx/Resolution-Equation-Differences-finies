import numpy as np
import matplotlib.pyplot as plt
from main import resoudre_equation_diff


def main():
    # Exemple 1: Solution analytique u(x) = sin(π*x)
    # Pour cette solution, f(x) = π²*sin(π*x)
    def f1(x):
        return np.pi ** 2 * np.sin(np.pi * x)

    # Conditions aux limites pour sin(π*x)
    U0 = 0  # sin(0) = 0
    U1 = 0  # sin(π) = 0

    # Résolution avec N = 40 points
    N = 10
    U_solution, x = resoudre_equation_diff(f1, N, U0, U1, tracer_graphe=True)

    # Comparaison avec la solution exacte
    x_exact = np.linspace(0, 1, 1000)
    u_exact = np.sin(np.pi * x_exact)

    plt.figure(figsize=(10, 6))
    plt.plot(x, U_solution, 'bo-', label='Solution numérique')
    plt.plot(x_exact, u_exact, 'r-', label='Solution exacte')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'Comparaison des solutions pour u(x) = sin(πx) avec N = {N}')
    plt.legend()
    plt.show()

    # Exemple 2: Solution analytique u(x) = x³
    # Pour cette solution, f(x) = 6x
    def f2(x):
        return -1/(1+x)**2

    # Conditions aux limites pour x³
    U0 = 0  # 0³ = 0
    U1 = 1  # 1³ = 1

    # Résolution avec N = 40 points
    U_solution, x = resoudre_equation_diff(f2, N, U0, U1, tracer_graphe=True)

    # Comparaison avec la solution exacte
    u_exact = np.log(x_exact+1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, U_solution, 'bo-', label='Solution numérique')
    plt.plot(x_exact, u_exact, 'r-', label='Solution exacte')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'Comparaison des solutions pour u(x) = x³ avec N = {N}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()