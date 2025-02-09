from pyswip import Prolog
from itertools import islice
import pandas as pd


# Inizializza Prolog
prolog = Prolog()
prolog.consult("suggerimenti.pl")

def query_to_csv(brano_preferito, output_csv):
    # Esegui la query in Prolog
    risultati = list(islice(prolog.query(
        f"suggerisci_simile_energia('{brano_preferito}', Suggerito, Artista, Anno2, Energia2, Valence2, Dance2, Tempo2, DistanzaTotale)"), 200))
    # Controlla se ci sono risultati
    if risultati:
        # Crea un DataFrame da risultati
        df = pd.DataFrame(risultati)

        # Specifica le colonne corrispondenti ai campi di output di Prolog
        df.columns = [
            "name", "artists", "year", "energy", "valence", "danceability", "tempo", "distance"
        ]

        # Ordina i risultati per distanza crescente (più simili in alto)
        df = df.sort_values(by="distance")

        # Salva il DataFrame in un file CSV
        df.to_csv(output_csv, index=False)
        print(f"Risultati salvati in {output_csv}")
    else:
        print("Nessun risultato trovato.")


#Inserire nel primo argomento il nome del brano che si vuole cercare
query_to_csv("Wicked Game","risultato.csv")

