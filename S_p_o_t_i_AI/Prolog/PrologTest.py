from pyswip import Prolog
from itertools import islice
import pandas as pd


# Inizializza Prolog
prolog = Prolog()
prolog.consult("suggerimenti.pl")


def format_title(title):
    return " ".join(word.capitalize() for word in title.split())


def query_to_csv(brano_preferito, output_csv, retry=True):
    # Esegui la query in Prolog
    risultati = list(islice(prolog.query(
        f"suggerisci_simile_energia('{brano_preferito}', Suggerito, Artista2, Anno2, Energia2, Valence2, Dance2, Tempo2, DistanzaTotale)"), 200))
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

    elif retry:  # Se non ci sono risultati e il retry è True, riprova con la formattazione del brano
        print("Nessun risultato trovato. Faccio un secondo tentativo cambiando formattazione brano:")
        brano_preferito = format_title(brano_preferito)
        query_to_csv(brano_preferito, output_csv,retry=False)  # Disabilita il retry per evitare loop infinito

    else:
        print("Nessun risultato trovato dopo il secondo tentativo.")


def queryart_to_csv(brano_preferito,artista, output_csv, retry=True):
    artista = format_title(artista)
    # Esegui la query in Prolog
    risultati = list(islice(prolog.query(
        f"suggerisci2_simile_energia('{brano_preferito}','{artista}', Suggerito, Artista2, Anno2, Energia2, Valence2, Dance2, Tempo2, DistanzaTotale)"), 200))
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

    elif retry:  # Se non ci sono risultati e il retry è True, riprova con la formattazione del brano
        print("Nessun risultato trovato. Faccio un secondo tentativo cambiando formattazione brano:")
        brano_preferito = format_title(brano_preferito)
        queryart_to_csv(brano_preferito, artista, output_csv,retry=False)  # Disabilita il retry per evitare loop infinito

    else:
        print("Nessun risultato trovato dopo il secondo tentativo.")


input_data = input("Inserisci il nome del brano e l'artista separati da una virgola (ex. Da Funk, Daft Punk): ").strip()
brano, artista = (input_data.split(",", 1) + [""])[:2]  #Assicura che artista sia opzionale
brano, artista = brano.strip(), artista.strip()

if artista=="":
    query_to_csv(brano, "risultato.csv")
else:
    queryart_to_csv(brano, artista, "risultato.csv")


