import csv
import statistics
from typing import List

def saveCSV(filepath: str, iteration: int, nb_pred: int,nb_th: int,
            nb_found: int,f1_scores: List[float], quartiles: List[float],
             low: float, high: float, pvalue: float, write_header: bool = True ):
    """
    Paramètres
    ----------
    filepath : str  Chemin vers le fichier CSV.
    iteration : int  Numéro de l'itération.
    nb_pred : int    Nombre de modules prédits.
    nb_th : int      Nombre de "true hits".
    nb_found : int   Nombre de "true hits" retrouvés.
    f1_scores : List[float]  Liste des scores F1.
    quartiles : List[float]  Les quartiles Q25, Q50, Q75.
    pvalue : float  p-value associée à l’itération.
    write_header : bool   Si True, écrit l'en-tête (utile au début).
    """

    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["iteration", "nb_pred", "true_hits", "found", "f1_score",
                "mean_f1", "variance_f1", "Q25", "Q50", "Q75", "low", "high", "pvalue" ])

        writer.writerow([iteration, nb_pred, nb_th, nb_found, round(f1_scores[-1], 5),
            round(statistics.mean(f1_scores), 5), round(statistics.pstdev(f1_scores), 5),
            round(quartiles[0], 5),round(quartiles[1], 5),round(quartiles[2], 5),
            round(low, 5),round(high, 5),round(pvalue, 5) ])
