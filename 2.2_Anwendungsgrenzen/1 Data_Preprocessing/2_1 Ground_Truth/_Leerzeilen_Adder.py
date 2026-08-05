import csv
import os


def fuege_leerzeilen_ein(csv_filename):
    print(f"Starte Skript mit {csv_filename}...")

    # Pfade definieren
    base_dir = os.path.dirname(__file__) # dirname(os.path.abspath(__file__))
    # csv_dir = os.path.normpath(os.path.join(base_dir, "..", "0 Model_Database"))
    csv_path = os.path.join(base_dir, f"{csv_filename}.csv")
    # print(csv_path) # Debug


    with open(csv_path, mode="r", newline="", encoding="utf-8") as f_in:
        # Erste Zeile kurz roh lesen, um das Trennzeichen (; oder ,) zu bestimmen
        erste_zeile = f_in.readline()
        trennzeichen = ";" if ";" in erste_zeile else ","

        # Dateizeiger wieder an den Anfang setzen
        f_in.seek(0)

        reader = csv.reader(f_in, delimiter=trennzeichen)

        # Header einlesen und merken
        header = next(reader)

        # Index der Spalte "GUID_A" ermitteln
        try:
            guid_index = header.index("GUID_A")
        except ValueError:
            print("Fehler: Spalte 'GUID_A' wurde nicht gefunden.")
            return

        zeilen = list(reader)

    neue_zeilen = [header]
    vorherige_guid = None

    for i, zeile in enumerate(zeilen):
        # Falls die Zeile leer ist, überspringen
        if not zeile:
            continue

        aktuelle_guid = zeile[guid_index]

        # Wenn sich der Wert ändert (aber nicht bei der allerersten Datenzeile)
        if vorherige_guid is not None and aktuelle_guid != vorherige_guid:
            # Fügt eine Leerzeile mit 7 leeren Feldern ein
            neue_zeilen.append(["", "", "", "", "", "", ""])

        neue_zeilen.append(zeile)
        vorherige_guid = aktuelle_guid

    # Ergebnis in neue Datei schreiben
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out, delimiter=trennzeichen)
        writer.writerows(neue_zeilen)
    
    print(f"...Skript mit {csv_filename} erfolgreich überarbeitet.")


# Dateinamen anpassen
# fuege_leerzeilen_ein("model_1 GT_Labels")
# fuege_leerzeilen_ein("model_7 GT_Labels")
# fuege_leerzeilen_ein("model_9 GT_Labels")
