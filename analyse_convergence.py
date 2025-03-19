import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv

from main import resoudre_equation_diff


def solution_exacte_sin(x):
    """Solution exacte u(x) = sin(πx)"""
    return np.sin(np.pi * x)


def solution_exacte_cube(x):
    """Solution exacte u(x) = x³"""
    return x ** 3
def solution_exacte_log(x):
    """Solution exacte u(x) = log(x+1)"""
    return np.log(x+1)


def terme_source_sin(x):
    """Terme source f(x) pour u(x) = sin(πx) dans -u''(x) = f(x)"""
    return np.pi ** 2 * np.sin(np.pi * x)


def terme_source_cube(x):
    """Terme source f(x) pour u(x) = x³ dans -u''(x) = f(x)"""
    return 6 * x
def terme_source_log(x):
    """Terme source f(x) pour u(x) = log(x+1) dans -u''(x) = f(x)"""
    return -1/(1+x)**2

def erreur_Linfini(u_numerique, u_exacte, x):
    """Calcule l'erreur en norme L∞ entre la solution numérique et la solution exacte"""
    return np.max(np.abs(u_numerique - u_exacte(x)))


def calculer_ordre_convergence(N_values, erreurs):
    """Calcule l'ordre numérique de convergence à partir des erreurs pour différentes tailles de maillage"""
    log_N = np.log(np.array(N_values))
    log_erreurs = np.log(np.array(erreurs))

    ordres = []
    for i in range(1, len(N_values)):
        ordre = -(log_erreurs[i] - log_erreurs[i - 1]) / (log_N[i] - log_N[i - 1])
        ordres.append(ordre)

    ordre_moyen = np.mean(ordres)

    return ordres, ordre_moyen


def analyser_convergence(solution_exacte, terme_source, u0, u1, N_values, nom_cas, dossier_figures):
    """Analyse complète de la convergence pour un cas test donné"""
    erreurs = []

    for N in N_values:

        u_numerique, x = resoudre_equation_diff(terme_source, N, u0, u1, tracer_graphe=False)

        erreur = erreur_Linfini(u_numerique, solution_exacte, x)
        erreurs.append(erreur)

        x_exact = np.linspace(0, 1, 1000)
        u_exact = solution_exacte(x_exact)

        plt.figure(figsize=(10, 6))
        plt.plot(x, u_numerique, 'bo-', label=f'Solution numérique (N={N})')
        plt.plot(x_exact, u_exact, 'r-', label='Solution exacte')
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title(f'Comparaison des solutions pour {nom_cas} avec N = {N}')
        plt.legend()
        plt.tight_layout()

        fichier_figure = os.path.join(dossier_figures, f"{nom_cas.replace('(', '').replace(')', '')}_N{N}.png")
        plt.savefig(fichier_figure, dpi=300)
        plt.close()

    ordres, ordre_moyen = calculer_ordre_convergence(N_values, erreurs)

    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, erreurs, 'bo-', linewidth=2)
    plt.loglog(N_values, [erreurs[0] * (N_values[0] / N) ** 2 for N in N_values], 'r--',
               label=f'Ordre 2 théorique (pente -2)')
    plt.grid(True)
    plt.xlabel('N (échelle log)')
    plt.ylabel('Erreur L-infini (échelle log)')
    plt.title(f'Convergence pour {nom_cas}')
    plt.legend()
    plt.tight_layout()

    fichier_figure = os.path.join(dossier_figures, f"{nom_cas.replace('(', '').replace(')', '')}_convergence.png")
    plt.savefig(fichier_figure, dpi=300)
    plt.close()

    return erreurs, ordres, ordre_moyen


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier_figures = f"figures_{timestamp}"
    os.makedirs(dossier_figures, exist_ok=True)

    N_values = [10, 20, 40, 80, 160, 320]

    # Cas test 1: u(x) = sin(πx)
    u0_sin = 0  # sin(0) = 0
    u1_sin = 0  # sin(1π) = 0
    erreurs_sin, ordres_sin, ordre_moyen_sin = analyser_convergence(
        solution_exacte_sin, terme_source_sin, u0_sin, u1_sin,
        N_values, "u(x) = sin(πx)", dossier_figures
    )

    # Cas test 2: u(x) = x³
    u0_cube = 0  # 0³ = 0
    u1_cube = 1  # 1³ = 1
    erreurs_cube, ordres_cube, ordre_moyen_cube = analyser_convergence(
        solution_exacte_cube, terme_source_cube, u0_cube, u1_cube,
        N_values, "u(x) = x³", dossier_figures
    )

    # Afficher les résultats dans la console
    print("\n" + "=" * 80)
    print("ANALYSE DE CONVERGENCE - MÉTHODE DES DIFFÉRENCES FINIES")
    print("=" * 80)

    print("\nCas 1: u(x) = sin(πx)")
    print("-" * 60)
    print(f"{'N':<8} {'Erreur L-infini':<20} {'Ordre de conv.':<15}")
    print("-" * 60)
    for i, N in enumerate(N_values):
        ordre_str = f"{ordres_sin[i - 1]:.4f}" if i > 0 else "N/A"
        print(f"{N:<8} {erreurs_sin[i]:<20.10e} {ordre_str:<15}")
    print("-" * 60)
    print(f"Ordre moyen de convergence: {ordre_moyen_sin:.4f}")

    print("\nCas 2: u(x) = x³")
    print("-" * 60)
    print(f"{'N':<8} {'Erreur L-infini':<20} {'Ordre de conv.':<15}")
    print("-" * 60)
    for i, N in enumerate(N_values):
        ordre_str = f"{ordres_cube[i - 1]:.4f}" if i > 0 else "N/A"
        print(f"{N:<8} {erreurs_cube[i]:<20.10e} {ordre_str:<15}")
    print("-" * 60)
    print(f"Ordre moyen de convergence: {ordre_moyen_cube:.4f}")

    # Sauvegarder les résultats dans un fichier CSV avec encodage UTF-8
    fichier_resultats = f"resultats_convergence_{timestamp}.csv"
    with open(fichier_resultats, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cas', 'N', 'Erreur L-infini', 'Ordre de convergence'])

        # Cas sin(πx)
        for i, N in enumerate(N_values):
            ordre = ordres_sin[i - 1] if i > 0 else None
            writer.writerow(['sin(πx)', N, erreurs_sin[i], ordre])
        writer.writerow(['sin(πx)', 'Ordre moyen', '', ordre_moyen_sin])
        writer.writerow([])

        # Cas x³
        for i, N in enumerate(N_values):
            ordre = ordres_cube[i - 1] if i > 0 else None
            writer.writerow(['x³', N, erreurs_cube[i], ordre])
        writer.writerow(['x³', 'Ordre moyen', '', ordre_moyen_cube])

    # Sauvegarder les résultats dans un fichier texte formaté avec encodage UTF-8
    fichier_txt = f"resultats_convergence_{timestamp}.txt"
    with open(fichier_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("ANALYSE DE CONVERGENCE - MÉTHODE DES DIFFÉRENCES FINIES\n")
        f.write("=" * 120 + "\n\n")

        f.write("Cas 1: u(x) = sin(πx)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'N':<8} {'Erreur L-infini':<20} {'Ordre de conv.':<15}\n")
        f.write("-" * 100 + "\n")
        for i, N in enumerate(N_values):
            ordre_str = f"{ordres_sin[i - 1]:.4f}" if i > 0 else "N/A"
            f.write(f"{N:<8} {erreurs_sin[i]:<20.10e} {ordre_str:<15}\n")
        f.write("-" * 100 + "\n")
        f.write(f"Ordre moyen de convergence: {ordre_moyen_sin:.4f}\n\n")

        f.write("Cas 2: u(x) = x³\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'N':<8} {'Erreur L-infini':<20} {'Ordre de conv.':<15}\n")
        f.write("-" * 100 + "\n")
        for i, N in enumerate(N_values):
            ordre_str = f"{ordres_cube[i - 1]:.4f}" if i > 0 else "N/A"
            f.write(f"{N:<8} {erreurs_cube[i]:<20.10e} {ordre_str:<15}\n")
        f.write("-" * 100 + "\n")
        f.write(f"Ordre moyen de convergence: {ordre_moyen_cube:.4f}\n")

        f.write("\n\nAnalyse sauvegardée le: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("\nRésultats sauvegardés dans:")
    print(f" - Figures: {dossier_figures}/")
    print(f" - Résultats CSV: {fichier_resultats}")
    print(f" - Résultats TXT: {fichier_txt}")

    print("\nAnalyse terminée!")


if __name__ == "__main__":
    main()