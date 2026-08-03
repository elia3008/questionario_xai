# -*- coding: utf-8 -*-
"""
Studio Streamlit sull'usabilita' dei metodi XAI
===============================================
Confronto tra una condizione BASELINE (nessuna spiegazione) e tre metodi di
spiegazione (SHAP, DiCE, Anchors) su uno stesso modello black-box
(Random Forest, dataset Heart Disease / Cleveland).

FLUSSO
    consenso -> demografiche -> istruzioni
    -> BLOCCO 1: Baseline (fisso, sempre per primo)
    -> BLOCCHI 2-4: SHAP / DiCE / Anchors in ordine bilanciato tra partecipanti
    -> pagina finale

STRUTTURA DI OGNI BLOCCO
    fase 1 "esposizione": 2 pazienti esempio (1 malato + 1 sano) con il verdetto
                          del modello; nei blocchi 2-4 anche la spiegazione
    fase 2 "domande":     5 pazienti di test con verdetto NASCOSTO
                          -> previsione + confidenza
                          -> (solo blocchi con spiegazione) scala ESS

Nella baseline non viene mostrata alcuna probabilita': l'output "nudo" del
modello e' solo Malato/Sano. Nella baseline non viene somministrata la ESS,
perche' non c'e' alcuna spiegazione da valutare.

STATO DI AVANZAMENTO
    [OK] baseline: 2 esempi + 5 test, dati definitivi
    [OK] SHAP    : 2 esempi + spiegazione locale, dati definitivi
    [OK] SHAP    : 5 pazienti di test
    [OK] DiCE    : 2 esempi + controfattuali + 5 test, dati definitivi
    [OK] Anchors : 2 esempi + regole + 5 test  (vedi note su A_T5 e A_EX_sano)

I dati sono salvati in forma anonima in responses.csv (formato lungo).

--------------------------------------------------------------------------
COME ESEGUIRLO
--------------------------------------------------------------------------
    pip install streamlit pandas matplotlib
    streamlit run xai_study_app.py
"""

import itertools
import os
import re
import time
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Polygon
import pandas as pd
import streamlit as st

# =========================================================================
# CONFIGURAZIONE
# =========================================================================
st.set_page_config(page_title="Studio XAI", page_icon="🫀", layout="centered")

RESULTS_FILE = "responses.csv"
COUNTER_FILE = "participant_counter.txt"

# --- Google Sheets -------------------------------------------------------
# Nomi dei due fogli dentro il documento Google
SHEET_RISPOSTE = "risposte"
SHEET_ASSEGNAZIONI = "assegnazioni"

# Colonne del foglio risposte, nell'ordine in cui vengono scritte
COLONNE = ["participant_id", "group", "block_sequence", "block_position",
           "method", "qtype", "item", "response", "correct", "confidence",
           "rt_seconds", "timestamp"]

BASELINE = "Baseline"
METHODS = ["SHAP", "DiCE", "Anchors"]
ORDERS = list(itertools.permutations(METHODS))   # 6 ordini possibili (3!)
N_BLOCKS = 1 + len(METHODS)                      # baseline + 3 metodi

# In quale blocco (1..4) compare l'attention check.
# Default 2 = primo blocco CON spiegazione: cosi' non "sporca" la baseline,
# che e' la condizione di controllo.
ATTENTION_CHECK_AT_BLOCK = 2

# Mostrare il grafico di importanza globale SHAP nel blocco SHAP?
# Per ora disattivato. Se lo riattivi, popola prima SHAP_GLOBAL con i valori
# reali del notebook.
SHOW_SHAP_GLOBAL = False

# Colori delle barre SHAP (stessa palette della libreria shap)
SHAP_RED = "#ff0d57"    # contributo verso MALATO
SHAP_BLUE = "#1e88e5"   # contributo verso SANO

# Stile del grafico SHAP locale:
#   "waterfall" = come la libreria shap: parte da E[f(X)] e accumula fino a f(x)
#   "barre"     = barre semplici centrate sullo zero, senza probabilita'
SHAP_PLOT_STYLE = "waterfall"

# Valore atteso del modello, E[f(X)]: punto di partenza del waterfall.
SHAP_BASE_VALUE = 0.465

# ------------------------------------------------------------------
# MODALITA' SVILUPPO
# ------------------------------------------------------------------
# Si attiva SOLO aggiungendo ?dev=<chiave> all'URL, per esempio:
#     http://localhost:8501/?dev=sblocca
# Non e' una costante da ricordarsi di rimettere a False: l'app pubblicata
# resta normale.
#
# La chiave si legge da secrets.toml, sezione [app] -> dev_key. Tenerla li'
# invece che nel codice e' importante se il repository GitHub e' pubblico:
# altrimenti chiunque legga i sorgenti la troverebbe scritta. Se nei secrets
# non c'e', si usa il valore di ripiego qui sotto.
#
# In modalita' sviluppo:
#   - compare un pannello laterale per saltare a qualsiasi blocco/fase
#   - le risposte possono essere precompilate con un click
#   - NULLA viene scritto in responses.csv (le prove non sporcano i dati)
#   - il contatore del bilanciamento non viene incrementato
DEV_KEY_RIPIEGO = "sblocca"

# =========================================================================
# SCHEMA DELLE FEATURE  (13 variabili, codifica Cleveland grezza)
# =========================================================================
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

FEATURE_LABEL = {
    "age":      "Età (age)",
    "sex":      "Sesso (sex)",
    "cp":       "Tipo dolore toracico (cp)",
    "trestbps": "Pressione riposo (trestbps)",
    "chol":     "Colesterolo (chol)",
    "fbs":      "Glicemia (fbs)",
    "restecg":  "ECG riposo (restecg)",
    "thalach":  "Freq. cardiaca max (thalach)",
    "exang":    "Angina da sforzo (exang)",
    "oldpeak":  "Depressione ST (oldpeak)",
    "slope":    "Pendenza ST (slope)",
    "ca":       "Vasi colorati (ca)",
    "thal":     "Test al tallio (thal)",
}

# Nome breve senza il codice tra parentesi, usato nei grafici SHAP
FEATURE_SHORT = {k: v.split(" (")[0] for k, v in FEATURE_LABEL.items()}

# UNICA fonte di verita' per le etichette: cambiando qui cambiano ovunque
# (tabelle, grafici SHAP, controfattuali, regole). Evita disallineamenti.
CATEGORICAL_MEANING = {
    "sex":     {0: "Femmina", 1: "Maschio"},
    "cp":      {1: "Angina tipica", 2: "Angina atipica",
                3: "Dolore non anginoso", 4: "Asintomatico"},
    "fbs":     {0: "Normale (<=120 mg/dL)", 1: "Alta (>120 mg/dL)"},
    "restecg": {0: "Normale", 1: "Anomalia onda ST-T", 2: "Ipertrofia ventric."},
    "exang":   {0: "No", 1: "Sì"},
    "slope":   {1: "Ascendente", 2: "Piatta", 3: "Discendente"},
    "thal":    {3: "Normale", 6: "Difetto fisso", 7: "Difetto reversibile"},
}


# Variabili omesse di default da TUTTI i profili, per alleggerire la lettura.
# Sono le tre che nei nostri esempi non compaiono mai nelle spiegazioni e che
# hanno il contributo SHAP piu' vicino a zero.
# Un singolo paziente puo' sovrascrivere l'elenco con la chiave "hide".
HIDE_DEFAULT = ["fbs", "trestbps", "restecg"]


def _feature_citate(paziente):
    """Variabili nominate dalla spiegazione di questo paziente.

    Non vanno mai nascoste, altrimenti la spiegazione parlerebbe di un valore
    che il partecipante non trova da nessuna parte nella tabella.
    """
    eid = paziente.get("id")
    citate = set()
    for f, _, _ in DICE_CF.get(eid, ()):        # controfattuali DiCE
        citate.add(f)
    regola = ANCHORS_RULE.get(eid)              # regole Anchors
    if regola:
        # la variabile e' dichiarata nella coppia, non dedotta dal testo:
        # riscrivere le condizioni non puo' rompere la corrispondenza
        for feat, _ in regola["rule"]:
            citate.add(feat)
    return citate


def visible_features(paziente):
    """Le variabili da mostrare per un paziente, nell'ordine standard.

    Parte da HIDE_DEFAULT, salvo che il paziente indichi un proprio "hide", e
    rimette comunque in vista le variabili citate dalla sua spiegazione.

    ATTENZIONE nel blocco SHAP: il waterfall parte da E[f(X)] e somma i
    contributi fino a f(x). Nascondendo variabili la somma cambia, e f(x) non
    corrisponde piu' alla probabilita' reale del modello — a meno che i
    contributi omessi si annullino. Se ne occupa _verifica_shap_nascoste().
    """
    nascoste = set(paziente.get("hide", HIDE_DEFAULT)) - _feature_citate(paziente)
    return [f for f in FEATURES if f not in nascoste]


def _num(v):
    """Formatta un numero senza zeri inutili: 0.0 -> '0', 1.90 -> '1.9'."""
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:g}"


def meaning(feat, val):
    """Testo leggibile della colonna 'Valore' (con unita' di misura)."""
    if feat in CATEGORICAL_MEANING:
        return CATEGORICAL_MEANING[feat].get(int(val), f"⚠ codice {val} non mappato")
    if feat == "age":
        return f"{_num(val)} anni"
    if feat == "trestbps":
        return f"{_num(val)} mmHg"
    if feat == "chol":
        return f"{_num(val)} mg/dl"
    if feat == "thalach":
        return f"{_num(val)} bpm"
    if feat == "oldpeak":
        return f"{_num(val)} mm"
    if feat == "ca":
        n = int(val)
        return "0 vasi" if n == 0 else ("1 vaso" if n == 1 else f"{n} vasi")
    return _num(val)


def short_value(feat, val):
    """Valore compatto per le etichette del grafico SHAP (senza unita')."""
    if feat in CATEGORICAL_MEANING:
        return CATEGORICAL_MEANING[feat].get(int(val), f"?{val}")
    return _num(val)


# =========================================================================
# BLOCCO 1 - BASELINE          (dati definitivi)
# =========================================================================
# --- 2 esempi mostrati CON il verdetto del modello (nessuna probabilita')
BASELINE_EXAMPLES = [
    {"id": "B_EX_malato", "pred": "Malato", "values": {   # id24
        "age": 55, "sex": 1, "cp": 4, "trestbps": 160, "chol": 289, "fbs": 0,
        "restecg": 2, "thalach": 145, "exang": 1, "oldpeak": 0.8,
        "slope": 2, "ca": 1, "thal": 7}},
    {"id": "B_EX_sano", "pred": "Sano", "values": {       # id46
        "age": 61, "sex": 0, "cp": 4, "trestbps": 130, "chol": 330, "fbs": 0,
        "restecg": 2, "thalach": 169, "exang": 0, "oldpeak": 0,
        "slope": 1, "ca": 0, "thal": 3}},
]

# --- 5 pazienti di TEST, verdetto NASCOSTO al partecipante.
#     'truth' = previsione reale del modello, usata solo per il punteggio.
BASELINE_TESTS = [
    {"id": "B_T1", "truth": "Malato", "values": {         # id25
        "age": 66, "sex": 1, "cp": 4, "trestbps": 112, "chol": 212, "fbs": 0,
        "restecg": 2, "thalach": 132, "exang": 1, "oldpeak": 0.1,
        "slope": 1, "ca": 1, "thal": 3}},
    {"id": "B_T2", "truth": "Malato", "values": {         # id26
        "age": 62, "sex": 0, "cp": 4, "trestbps": 138, "chol": 294, "fbs": 1,
        "restecg": 0, "thalach": 106, "exang": 0, "oldpeak": 1.9,
        "slope": 2, "ca": 3, "thal": 3}},
    {"id": "B_T3", "truth": "Malato", "values": {         # id0
        "age": 58, "sex": 1, "cp": 3, "trestbps": 112, "chol": 230, "fbs": 0,
        "restecg": 2, "thalach": 165, "exang": 0, "oldpeak": 2.5,
        "slope": 2, "ca": 1, "thal": 7}},
    {"id": "B_T4", "truth": "Sano", "values": {           # id41
        "age": 43, "sex": 1, "cp": 3, "trestbps": 130, "chol": 315, "fbs": 0,
        "restecg": 0, "thalach": 162, "exang": 0, "oldpeak": 1.9,
        "slope": 1, "ca": 1, "thal": 3}},
    {"id": "B_T5", "truth": "Sano", "values": {           # id14
        "age": 51, "sex": 1, "cp": 3, "trestbps": 100, "chol": 222, "fbs": 0,
        "restecg": 0, "thalach": 143, "exang": 1, "oldpeak": 1.2,
        "slope": 2, "ca": 0, "thal": 3}},
]

# =========================================================================
# BLOCCO SHAP - ESEMPI         (dati definitivi)
# =========================================================================
# I profili omettono le variabili elencate in HIDE_DEFAULT (glicemia, pressione
# a riposo, ECG a riposo), per alleggerire la lettura. Sul waterfall questo
# sposta leggermente il valore finale:
#   id18 -> contributi omessi +0.02  =>  f(x) mostrato 0.895 invece di 0.915
#   id10 -> contributi omessi  0.00  =>  f(x) mostrato 0.255, invariato
# Lo scarto e' impercettibile per chi non conosce il valore vero, ma va
# dichiarato tra le scelte di disegno. _verifica_shap_nascoste() lo segnala in
# modalita' sviluppo.
SHAP_EXAMPLES = [
    {"id": "S_EX_malato", "pred": "Malato",               # id18,  f(x) = 0.92
     "values": {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 140, "chol": 187, "fbs": 0,
        "restecg": 2, "thalach": 144, "exang": 1, "oldpeak": 4,
        "slope": 1, "ca": 2, "thal": 7}},
    {"id": "S_EX_sano", "pred": "Sano",                   # id10,  f(x) = 0.25
     "values": {
        "age": 68, "sex": 0, "cp": 3, "trestbps": 120, "chol": 211, "fbs": 0,
        "restecg": 2, "thalach": 115, "exang": 0, "oldpeak": 1.5,
        "slope": 2, "ca": 0, "thal": 3}},
]

# Contributi SHAP locali: positivo = spinge verso MALATO, negativo = verso SANO.
# Valore atteso del modello E[f(X)] = 0.465.
SHAP_LOCAL = {
    "S_EX_malato": {                       # somma = +0.45  ->  f(x) = 0.92
        "cp": 0.11, "ca": 0.11, "oldpeak": 0.11, "thal": 0.08,
        "slope": -0.05, "thalach": 0.03, "sex": 0.02, "exang": 0.02,
        "age": 0.02, "chol": -0.02, "restecg": 0.01, "trestbps": 0.01,
        "fbs": 0.00,
    },
    "S_EX_sano": {                         # somma = -0.21  ->  f(x) = 0.25
        "thalach": 0.11, "cp": -0.10, "ca": -0.08, "thal": -0.08,
        "sex": -0.08, "chol": -0.05, "slope": 0.05, "age": 0.03,
        "exang": -0.02, "oldpeak": 0.01, "trestbps": -0.01,
        "restecg": 0.01, "fbs": 0.00,
    },
}

# Importanza globale SHAP: usata solo se SHOW_SHAP_GLOBAL = True.
# >>> DA POPOLARE con i valori reali prima di riattivarla.
SHAP_GLOBAL = {}

# --- 5 pazienti di TEST del blocco SHAP, verdetto NASCOSTO
SHAP_TESTS = [
    {"id": "S_T1", "truth": "Sano", "values": {           # id3
        "age": 53, "sex": 0, "cp": 4, "trestbps": 138, "chol": 234, "fbs": 0,
        "restecg": 2, "thalach": 160, "exang": 0, "oldpeak": 0,
        "slope": 1, "ca": 0, "thal": 3}},
    {"id": "S_T2", "truth": "Malato", "values": {         # id13
        "age": 52, "sex": 1, "cp": 4, "trestbps": 112, "chol": 230, "fbs": 0,
        "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 0,
        "slope": 1, "ca": 1, "thal": 3}},
    {"id": "S_T3", "truth": "Malato", "values": {         # id44
        "age": 45, "sex": 1, "cp": 4, "trestbps": 142, "chol": 309, "fbs": 0,
        "restecg": 2, "thalach": 147, "exang": 1, "oldpeak": 0,
        "slope": 2, "ca": 3, "thal": 7}},
    {"id": "S_T4", "truth": "Sano", "values": {           # id47
        "age": 51, "sex": 1, "cp": 4, "trestbps": 140, "chol": 261, "fbs": 0,
        "restecg": 2, "thalach": 186, "exang": 1, "oldpeak": 0,
        "slope": 1, "ca": 0, "thal": 3}},
    {"id": "S_T5", "truth": "Malato", "values": {         # id38
        "age": 63, "sex": 1, "cp": 4, "trestbps": 130, "chol": 254, "fbs": 0,
        "restecg": 2, "thalach": 147, "exang": 0, "oldpeak": 1.4,
        "slope": 2, "ca": 1, "thal": 7}},
]

# =========================================================================
# =========================================================================
# BLOCCO DiCE - ESEMPI E CONTROFATTUALI      (dati definitivi)
# =========================================================================
DICE_EXAMPLES = [
    {"id": "D_EX_malato", "pred": "Malato", "values": {   # id60
        "age": 57, "sex": 1, "cp": 4, "trestbps": 130, "chol": 131, "fbs": 0,
        "restecg": 0, "thalach": 115, "exang": 1, "oldpeak": 1.2,
        "slope": 2, "ca": 1, "thal": 7}},
    {"id": "D_EX_sano", "pred": "Sano", "values": {       # id16
        "age": 39, "sex": 0, "cp": 3, "trestbps": 138, "chol": 220, "fbs": 0,
        "restecg": 0, "thalach": 152, "exang": 0, "oldpeak": 0,
        "slope": 2, "ca": 0, "thal": 3}},
]

# Controfattuali DiCE: (feature, valore attuale, valore alternativo).
# Valori GREZZI: le etichette le genera meaning(), cosi' restano allineate
# alle tabelle dei profili.
DICE_CF = {
    # id60: da "malattia presente" a "malattia assente"
    "D_EX_malato": [("age", 57, 64), ("trestbps", 130, 128),
                    ("chol", 131, 263), ("thalach", 115, 105),
                    ("oldpeak", 1.2, 0.2)],
    # id16: da "malattia assente" a "malattia presente"
    "D_EX_sano":   [("age", 39, 47), ("sex", 0, 1),
                    ("trestbps", 138, 108), ("chol", 220, 243),
                    ("slope", 2, 1)],
}

# --- 5 pazienti di TEST del blocco DiCE, verdetto NASCOSTO
DICE_TESTS = [
    {"id": "D_T1", "truth": "Sano", "values": {           # id2
        "age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0,
        "restecg": 2, "thalach": 172, "exang": 0, "oldpeak": 1.4,
        "slope": 1, "ca": 0, "thal": 3}},
    {"id": "D_T2", "truth": "Malato", "values": {         # id5
        "age": 60, "sex": 1, "cp": 4, "trestbps": 117, "chol": 230, "fbs": 1,
        "restecg": 0, "thalach": 160, "exang": 1, "oldpeak": 1.4,
        "slope": 1, "ca": 2, "thal": 7}},
    {"id": "D_T3", "truth": "Malato", "values": {         # id6
        "age": 44, "sex": 1, "cp": 4, "trestbps": 120, "chol": 169, "fbs": 0,
        "restecg": 0, "thalach": 144, "exang": 1, "oldpeak": 2.8,
        "slope": 3, "ca": 0, "thal": 6}},
    {"id": "D_T4", "truth": "Sano", "values": {           # id32
        "age": 54, "sex": 0, "cp": 3, "trestbps": 108, "chol": 267, "fbs": 0,
        "restecg": 2, "thalach": 167, "exang": 0, "oldpeak": 0,
        "slope": 1, "ca": 0, "thal": 3}},
    {"id": "D_T5", "truth": "Malato", "values": {         # id59
        "age": 40, "sex": 1, "cp": 4, "trestbps": 110, "chol": 167, "fbs": 0,
        "restecg": 2, "thalach": 114, "exang": 1, "oldpeak": 2,
        "slope": 2, "ca": 0, "thal": 7}},
]

# =========================================================================
# BLOCCO Anchors - ESEMPI, REGOLE E TEST     (dati definitivi)
# =========================================================================
ANCHORS_EXAMPLES = [
    {"id": "A_EX_malato", "pred": "Malato", "values": {   # id12
        "age": 56, "sex": 0, "cp": 4, "trestbps": 200, "chol": 288, "fbs": 1,
        "restecg": 2, "thalach": 133, "exang": 1, "oldpeak": 4,
        "slope": 3, "ca": 2, "thal": 7}},
    {"id": "A_EX_sano", "pred": "Sano", "values": {       # id41
        "age": 43, "sex": 1, "cp": 3, "trestbps": 130, "chol": 315, "fbs": 0,
        "restecg": 0, "thalach": 162, "exang": 0, "oldpeak": 1.9,
        "slope": 1, "ca": 1, "thal": 3}},
]

# Regole Anchors. 'precision' = quota di pazienti simili in cui la regola e'
# corretta; 'coverage' = quota di pazienti a cui la regola si applica.
#
# Ogni condizione e' una coppia (variabile, testo): la variabile serve al
# codice — cosi' visible_features() sa che quella riga non va nascosta dal
# profilo — mentre il testo e' quello che legge il partecipante e puo' essere
# riscritto liberamente senza rompere nulla.
ANCHORS_RULE = {
    "A_EX_malato": {"rule": [
                        ("thalach", "Frequenza cardiaca massima minore o uguale a 133,25 bpm"),
                        ("oldpeak", "Depressione del tratto ST maggiore di 1,6 mm"),
                        ("thal",    "Test al tallio: difetto reversibile")],
                    "pred": "Malato", "precision": 1.00, "coverage": 0.25},
    "A_EX_sano":   {"rule": [
                        ("cp",    "Tipo di dolore toracico: dolore non anginoso"),
                        ("age",   "Età minore o uguale a 48 anni"),
                        ("slope", "Pendenza del tratto ST: ascendente")],
                    "pred": "Sano", "precision": 0.97, "coverage": 0.26},
}

# --- 5 pazienti di TEST del blocco Anchors, verdetto NASCOSTO
ANCHORS_TESTS = [
    {"id": "A_T1", "truth": "Sano", "values": {           # id4
        "age": 39, "sex": 0, "cp": 3, "trestbps": 94, "chol": 199, "fbs": 0,
        "restecg": 0, "thalach": 179, "exang": 0, "oldpeak": 0,
        "slope": 1, "ca": 0, "thal": 3}},
    {"id": "A_T2", "truth": "Malato", "values": {         # id21
        "age": 54, "sex": 1, "cp": 4, "trestbps": 110, "chol": 239, "fbs": 0,
        "restecg": 0, "thalach": 126, "exang": 1, "oldpeak": 2.8,
        "slope": 2, "ca": 1, "thal": 7}},
    {"id": "A_T3", "truth": "Sano", "values": {           # id27
        "age": 71, "sex": 0, "cp": 4, "trestbps": 112, "chol": 149, "fbs": 0,
        "restecg": 0, "thalach": 125, "exang": 0, "oldpeak": 1.6,
        "slope": 2, "ca": 0, "thal": 3}},
    {"id": "A_T4", "truth": "Malato", "values": {         # id58, prob 58% -> da confermare
        "age": 57, "sex": 1, "cp": 3, "trestbps": 128, "chol": 229, "fbs": 0,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 0.4,
        "slope": 2, "ca": 1, "thal": 7}},
    {"id": "A_T5", "truth": "Sano", "values": {           # id14  <-- vedi nota
        "age": 51, "sex": 1, "cp": 3, "trestbps": 100, "chol": 222, "fbs": 0,
        "restecg": 0, "thalach": 143, "exang": 1, "oldpeak": 1.2,
        "slope": 2, "ca": 0, "thal": 3}},
]
# ATTENZIONE: A_T5 (id14) e' lo STESSO paziente di B_T5 nella baseline, dove
# e' gia' un item di test. Il partecipante lo valuterebbe due volte, quindi la
# risposta in Anchors non e' indipendente da quella in baseline. Va sostituito.
# Nota anche: A_EX_sano (id41) coincide con B_T4, item di test della baseline.

# =========================================================================
# VARIABILI OMESSE IN PIU' NEI BLOCCHI DiCE E ANCHORS
# =========================================================================
# Oltre a HIDE_DEFAULT, questi due blocchi omettono anche l'angina da sforzo:
# non compare in nessuno dei loro controfattuali ne' nelle loro regole, quindi
# toglierla accorcia il profilo senza rendere incomprensibile la spiegazione.
# Resta invece visibile in SHAP, dove il grafico le assegna un contributo.
#
# DiCE tiene in piu' la pressione a riposo: i suoi controfattuali la usano, e i
# pazienti di test devono mostrare le stesse variabili degli esempi, altrimenti
# il partecipante non ritrova nel profilo cio' su cui ha imparato a ragionare.
#
# Le chiavi "hide" sono assegnate qui, in un punto solo, invece di ripeterle su
# ognuno dei quattordici pazienti.
for _p in DICE_EXAMPLES + DICE_TESTS:
    _p["hide"] = [f for f in HIDE_DEFAULT if f != "trestbps"] + ["exang"]

for _p in ANCHORS_EXAMPLES + ANCHORS_TESTS:
    _p["hide"] = HIDE_DEFAULT + ["exang"]

# =========================================================================
# MAPPE BLOCCO -> MATERIALE
# =========================================================================
EXAMPLES = {
    BASELINE:  BASELINE_EXAMPLES,
    "SHAP":    SHAP_EXAMPLES,
    "DiCE":    DICE_EXAMPLES,
    "Anchors": ANCHORS_EXAMPLES,
}
TESTS = {
    BASELINE:  BASELINE_TESTS,
    "SHAP":    SHAP_TESTS,
    "DiCE":    DICE_TESTS,
    "Anchors": ANCHORS_TESTS,
}

# =========================================================================
# TITOLI E DEFINIZIONI DEI TRE METODI
# =========================================================================
# Mostrati in cima alla fase di esposizione, prima dei due pazienti esempio.
# Il valore atteso citato nel testo di SHAP e' preso da SHAP_BASE_VALUE, cosi'
# la definizione non puo' andare in disaccordo con il grafico.
INTRO_METODO = {
    "SHAP": {
        "titolo": "Spiegazione SHAP",
        "testo": (
            "**Come leggere il grafico:** se la variabile è associata a una "
            "barra ROSSA, allora quella variabile spinge il modello a "
            "classificare il paziente come MALATO. Se è associata a una barra "
            "BLU, verso SANO. Si parte da E[f(X)] = "
            f"{SHAP_BASE_VALUE:g}"
            " (probabilità media di presentare una malattia cardiaca) e i "
            "contributi di ciascuna variabile portano alla predizione finale "
            "f(x)."

        ),
    },
    "DiCE": {
        "titolo": "Spiegazione DiCE",
        "testo": (
            "In questa sezione il modello mostra le sue previsioni insieme a "
            "una spiegazione di tipo \"controfattuale\": una tabella che indica "
            "quali valori dovrebbero cambiare, e in che modo, per far cambiare "
            "la previsione del modello."
        ),
    },
    "Anchors": {
        "titolo": "Spiegazione Anchors",
        "testo": (
            "In questa sezione il modello mostra le sue previsioni insieme a "
            "una spiegazione sotto forma di regola: un insieme di condizioni "
            "che, quando sono tutte vere, determinano la previsione del modello."
        ),
    },
}

# =========================================================================
# SCALE
# =========================================================================
SATISFACTION_ITEMS = [
    "Da questa spiegazione capisco come il modello ha preso la decisione.",
    "Nel complesso la spiegazione è soddisfacente.",
    "La spiegazione contiene un livello di dettaglio adeguato.",
    "Questa spiegazione mi aiuterebbe a fidarmi del modello.",
    "La spiegazione mi è utile per prevedere le decisioni del modello.",
]
LIKERT = {1: "1 - Per niente d'accordo", 2: "2", 3: "3 - Neutrale",
          4: "4", 5: "5 - Del tutto d'accordo"}

# La scala ESS e' mostrata come cursore. st.select_slider parte SEMPRE da un
# valore: senza un segnaposto iniziale non si distinguerebbe chi ha scelto "3"
# da chi non ha toccato nulla, e la ESS e' una misura principale dello studio.
# Lo 0 e' quindi lo stato "non ancora risposto" e viene bloccato in validazione.
NON_RISPOSTO = 0
LIKERT_SLIDER = {NON_RISPOSTO: "(nessuna risposta)", **LIKERT}

CONFIDENCE = {1: "1 - Per niente sicuro/a", 2: "2", 3: "3", 4: "4",
              5: "5 - Del tutto sicuro/a"}

# =========================================================================
# CLASSIFICAZIONE TECNICI / NON TECNICI
# =========================================================================
BG_STAT = "Statistica / Informatica / Data science / Machine learning"
BG_STEM = "Altro ambito STEM (scienza, tecnologia, ingegneria o matematica)"
BG_ALTRO = "Altro"
BACKGROUNDS = [BG_STAT, BG_STEM, BG_ALTRO]

FAMILIARITA = ["Nessuna", "Poca", "Media", "Buona", "Molta"]


def classifica_gruppo(bg, fam):
    """Assegna il partecipante al gruppo 'tecnico' o 'non_tecnico'.

    La soglia di familiarita' richiesta si abbassa man mano che il background
    si avvicina all'ambito:
      - background statistico/informatico -> sempre tecnico;
      - altro ambito STEM                 -> tecnico da 'Poca' in su;
      - altro                             -> tecnico solo con 'Buona' o 'Molta'.

    Nel foglio finiscono comunque le risposte grezze (background e
    familiarita'), quindi in fase di analisi si puo' riclassificare con un
    criterio diverso senza rifare la raccolta.
    """
    def almeno(livello):
        return FAMILIARITA.index(fam) >= FAMILIARITA.index(livello)

    if bg == BG_STAT:
        tecnico = True
    elif bg == BG_STEM:
        tecnico = almeno("Poca")
    else:
        tecnico = almeno("Buona")
    return "tecnico" if tecnico else "non_tecnico"


# =========================================================================
# FUNZIONI DI SUPPORTO
# =========================================================================
# =========================================================================
# SALVATAGGIO DEI DATI
# =========================================================================
# Due modalita', scelte automaticamente:
#
#   GOOGLE SHEETS  se in .streamlit/secrets.toml ci sono le credenziali di un
#                  service account. E' la modalita' da usare online.
#   CSV LOCALE     altrimenti. Comoda per lavorare sul proprio computer.
#
# ATTENZIONE: su Streamlit Community Cloud il filesystem viene azzerato a ogni
# riavvio, sospensione o aggiornamento del codice. Il CSV locale NON e' quindi
# utilizzabile per la raccolta vera: online serve Google Sheets.

class _ErroreCollegamento(Exception):
    """Sollevata quando non si riesce ad aprire il foglio Google."""


@st.cache_resource(show_spinner=False)
def _documento_o_errore():
    """Apre il documento Google, sollevando _ErroreCollegamento se fallisce.

    IMPORTANTE: solleva invece di restituire None perche' st.cache_resource NON
    mette in cache le eccezioni. Cosi' un collegamento riuscito viene
    memorizzato una volta sola, mentre un fallimento viene ritentato a ogni
    esecuzione: correggendo secrets.toml l'app riparte da sola, senza dover
    riavviare il processo.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        raise _ErroreCollegamento(
            "librerie mancanti in QUESTO interprete Python. Se le hai gia' "
            "installate, l'app sta girando in un ambiente diverso: attiva il "
            "venv e reinstalla li' dentro (streamlit compreso). Il percorso "
            "dell'interprete e' scritto qui sopra.")
    try:
        presente = "gcp_service_account" in st.secrets
    except Exception:
        presente = False
    if not presente:
        raise _ErroreCollegamento(
            "nessun secrets.toml trovato, oppure manca [gcp_service_account]. "
            "Streamlit lo cerca nella cartella da cui lanci il comando.")
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(creds)
    except Exception as e:
        raise _ErroreCollegamento(f"credenziali non valide ({type(e).__name__})")

    conf = {}
    try:
        conf = st.secrets.get("sheets", {})
    except Exception:
        pass
    if not (conf.get("spreadsheet_id") or conf.get("spreadsheet_url")):
        raise _ErroreCollegamento("manca spreadsheet_id nella sezione [sheets]")
    try:
        if conf.get("spreadsheet_id"):
            return client.open_by_key(conf["spreadsheet_id"])
        return client.open_by_url(conf["spreadsheet_url"])
    except Exception as e:
        testo = f"{type(e).__name__}: {e}"
        if "SpreadsheetNotFound" in testo or "NOT_FOUND" in testo or "404" in testo:
            raise _ErroreCollegamento(
                "il service account non vede il foglio: controlla di averlo "
                "condiviso con lui come Editor e che l'ID sia di un foglio di "
                "calcolo (/spreadsheets/), non di un documento (/document/)")
        if "PERMISSION_DENIED" in testo or "403" in testo:
            raise _ErroreCollegamento(
                "permesso negato: il foglio e' condiviso in sola lettura")
        if "has not been used" in testo or "SERVICE_DISABLED" in testo:
            raise _ErroreCollegamento(
                "Google Sheets API non abilitata sul progetto Google Cloud")
        raise _ErroreCollegamento(testo[:140])


def _apri_documento():
    """Restituisce (documento, motivo). Il motivo e' None se e' tutto a posto."""
    try:
        return _documento_o_errore(), None
    except _ErroreCollegamento as e:
        return None, str(e)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:140]}"


def _spreadsheet():
    """Solo il documento, senza il motivo dell'eventuale fallimento."""
    return _apri_documento()[0]


def _worksheet(nome, intestazione):
    """Restituisce il foglio richiesto, creandolo con l'intestazione se manca."""
    doc = _spreadsheet()
    if doc is None:
        return None
    try:
        try:
            return doc.worksheet(nome)
        except Exception:
            ws = doc.add_worksheet(title=nome, rows=2000,
                                   cols=max(len(intestazione), 4))
            ws.append_row(intestazione, value_input_option="RAW")
            return ws
    except Exception:
        return None


def _numero_riga(risposta_append):
    """Estrae il numero di riga dalla risposta di append_row di gspread."""
    try:
        rng = risposta_append["updates"]["updatedRange"]   # es. "'foglio'!A5:D5"
        m = re.search(r"![A-Z]+(\d+)", rng)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _cella(v):
    """Converte un valore Python in qualcosa che Google Sheets accetta."""
    return "" if v is None else v


def claim_balanced_order():
    """Assegna un ordine bilanciato dei 3 metodi (rotazione sui 6 ordini).

    Su Google Sheets aggiunge una riga al foglio 'assegnazioni' e usa il numero
    di riga restituito da Google come indice del partecipante. L'append e' una
    singola chiamata API, quindi due partecipanti simultanei ottengono righe
    diverse: niente race condition, a differenza del contatore su file.

    In modalita' sviluppo restituisce sempre il primo ordine senza consumare
    un'assegnazione.
    """
    if dev_mode():
        return list(ORDERS[0])

    ws = _worksheet(SHEET_ASSEGNAZIONI,
                    ["riga", "participant_id", "ordine", "timestamp"])
    if ws is not None:
        try:
            ora = datetime.now().isoformat(timespec="seconds")
            ris = ws.append_row(["", st.session_state.pid, "", ora],
                                value_input_option="RAW")
            riga = _numero_riga(ris)
            if riga is not None:
                # riga 1 = intestazione, quindi il primo partecipante e' riga 2
                ordine = list(ORDERS[(riga - 2) % len(ORDERS)])
                try:   # rispecchio riga e ordine, utile per il monitoraggio
                    ws.update([[riga]], f"A{riga}", value_input_option="RAW")
                    ws.update([[">".join(ordine)]], f"C{riga}",
                              value_input_option="RAW")
                except Exception:
                    pass
                return ordine
        except Exception as e:
            st.session_state.storage_error = f"assegnazione: {str(e)[:150]}"

    # --- fallback locale (sviluppo, oppure Sheets non raggiungibile)
    count = 0
    if os.path.exists(COUNTER_FILE):
        try:
            count = int((open(COUNTER_FILE).read().strip() or "0"))
        except ValueError:
            count = 0
    ordine = list(ORDERS[count % len(ORDERS)])
    try:
        with open(COUNTER_FILE, "w") as f:
            f.write(str(count + 1))
    except Exception:
        pass
    return ordine


def log(qtype, method, item, response, correct=None, confidence=None, rt=None):
    """Accumula una riga di risposta (formato lungo)."""
    st.session_state.rows.append({
        "participant_id": st.session_state.pid,
        "group": st.session_state.group,
        "block_sequence": ">".join(st.session_state.blocks or []),
        "block_position": st.session_state.block_idx + 1,
        "method": method,
        "qtype": qtype,
        "item": item,
        "response": response,
        "correct": correct,
        "confidence": confidence,
        "rt_seconds": rt,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


def _dev_key():
    """La chiave che sblocca il pannello: dai secrets, o quella di ripiego."""
    try:
        return st.secrets.get("app", {}).get("dev_key", DEV_KEY_RIPIEGO)
    except Exception:
        return DEV_KEY_RIPIEGO


def dev_mode():
    """True se l'app e' aperta con ?dev=<chiave> nell'URL."""
    try:
        return st.query_params.get("dev") == _dev_key()
    except Exception:
        return False


def autofill():
    """True se in modalita' sviluppo e' attiva la precompilazione."""
    return dev_mode() and st.session_state.get("dev_autofill", False)


def default_index():
    """Indice iniziale dei radio: 0 se precompilo, altrimenti nessuna scelta."""
    return 0 if autofill() else None


def salva_tutto():
    """Scrive TUTTE le risposte sul foglio, una volta sola, a fine questionario.

    Il salvataggio avviene solo al termine: chi interrompe a meta' non lascia
    righe parziali, e il foglio contiene esclusivamente questionari completi.

    Tre protezioni contro le righe doppie:
      - il flag 'saved' impedisce una seconda scrittura nella stessa sessione,
        anche se Streamlit riesegue lo script (cosa che accade di continuo);
      - la scrittura avviene in un'unica chiamata append_rows;
      - se la scrittura fallisce il flag NON viene alzato, cosi' un tentativo
        successivo puo' ancora riuscire.

    Restituisce True se i dati sono stati scritti (o lo erano gia').
    """
    if dev_mode():
        return False
    if st.session_state.get("saved"):
        return True                      # gia' salvato: non riscrivere
    righe = st.session_state.rows
    if not righe:
        return False

    ws = _worksheet(SHEET_RISPOSTE, COLONNE)
    if ws is not None:
        try:
            ws.append_rows([[_cella(r.get(c)) for c in COLONNE] for r in righe],
                           value_input_option="RAW")
            st.session_state.saved = True
            st.session_state.storage = "sheets"
            st.session_state.storage_error = None
            return True
        except Exception as e:
            st.session_state.storage_error = f"scrittura: {str(e)[:150]}"

    try:
        pd.DataFrame(righe).to_csv(
            RESULTS_FILE, mode="a",
            header=not os.path.exists(RESULTS_FILE), index=False)
        st.session_state.saved = True
        st.session_state.storage = "local"
        return True
    except Exception as e:
        st.session_state.storage_error = f"csv: {str(e)[:150]}"
        return False


def _radio_index(chiave, opzioni):
    """Indice iniziale di un radio, ripescando l'eventuale risposta salvata.

    Serve perche' Streamlit scarta lo stato dei widget che non vengono
    ridisegnati: uscendo dal form per rivedere le spiegazioni, le risposte
    andrebbero perse. Le conserviamo quindi in st.session_state.answers, che
    non e' legato al ciclo di vita dei widget.
    """
    val = _answers().get(chiave)
    if val in opzioni:
        return opzioni.index(val)
    return default_index()


def _slider_value(chiave, opzioni, default):
    """Valore iniziale di un cursore, ripescando l'eventuale risposta salvata.

    Stesso ruolo di _radio_index, ma i cursori vogliono il valore e non
    l'indice.
    """
    val = _answers().get(chiave)
    return val if val in opzioni else default


def _answers():
    """Dizionario delle risposte salvate, creandolo se non esiste."""
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    return st.session_state.answers


def _salva_risposte(mappa):
    """Memorizza le risposte correnti del form (chiave widget -> valore)."""
    salvate = _answers()
    for chiave, valore in mappa.items():
        if valore is not None:
            salvate[chiave] = valore


def _scroll_in_cima():
    """Riporta la finestra in cima alla pagina.

    Streamlit ridisegna la pagina ma il browser conserva la posizione dello
    scroll: cambiando schermata ci si ritrova a meta' o in fondo. Non esiste
    un'API nativa, quindi si inietta un frammento di JavaScript che agisce sul
    documento genitore (il frammento vive dentro un iframe).

    Il contatore nel commento serve a rendere l'HTML diverso a ogni chiamata:
    altrimenti Streamlit riutilizza lo stesso iframe e lo script non viene
    rieseguito.
    """
    n = st.session_state.get("_scroll_n", 0) + 1
    st.session_state["_scroll_n"] = n
    html = f"""
    <script>
      /* chiamata {n} */
      (function () {{
        const doc = window.parent && window.parent.document;
        if (!doc) return;
        const bersagli = [
          doc.querySelector('section.main'),
          doc.querySelector('[data-testid="stMain"]'),
          doc.querySelector('[data-testid="stAppViewContainer"]'),
          doc.querySelector('.main'),
          doc.scrollingElement, doc.documentElement, doc.body
        ];
        const suvvia = () => {{
          for (const el of bersagli) {{
            if (!el) continue;
            try {{ el.scrollTo({{top: 0, left: 0, behavior: 'instant'}}); }}
            catch (e) {{ el.scrollTop = 0; }}
          }}
          try {{ window.parent.scrollTo(0, 0); }} catch (e) {{}}
        }};
        suvvia();
        // ripetuto: al primo giro il contenuto nuovo puo' non essere ancora
        // stato disegnato, e il browser rimetterebbe lo scroll dov'era
        requestAnimationFrame(suvvia);
        setTimeout(suvvia, 60);
        setTimeout(suvvia, 250);
      }})();
    </script>
    """
    # st.components.v1.html e' deprecato dal 2026 in favore di st.iframe:
    # si usa il nuovo se c'e', altrimenti si ripiega sul vecchio.
    try:
        st.iframe(html, height=1)
    except AttributeError:
        import streamlit.components.v1 as components
        components.html(html, height=0)


def _pulsante_rivedi_flottante(attivo, testo):
    """Pulsante fisso in basso a destra che riporta alle spiegazioni.

    Il pulsante vero e' un form_submit_button in cima al modulo: scorrendo tra
    i cinque pazienti sparisce dalla vista, e per rivedere la spiegazione
    bisognerebbe risalire tutta la pagina. Qui se ne crea un gemello ancorato
    allo schermo, che al click preme quello vero.

    L'elemento viene inserito nel documento genitore (non nell'iframe dello
    script, che e' alto un pixel e lo ritaglierebbe) e rimosso quando non
    serve piu'. L'id univoco impedisce che se ne accumulino piu' copie.
    """
    stato = f"{attivo}|{testo}"
    if st.session_state.get("_flottante") == stato:
        return                       # gia' nello stato giusto: non reiniettare
    st.session_state["_flottante"] = stato

    etichetta = testo.replace("'", "\\'")
    html = f"""
    <script>
      (function () {{
        const d = window.parent && window.parent.document;
        if (!d) return;
        const vecchio = d.getElementById('rivedi-flottante');
        if (vecchio) vecchio.remove();
        if (!{str(bool(attivo)).lower()}) return;

        // stile in un foglio a parte: serve la media query per alzare il
        // pulsante sui telefoni, dove il badge di Streamlit e' piu' ingombrante
        let css = d.getElementById('rivedi-flottante-css');
        if (!css) {{
          css = d.createElement('style');
          css.id = 'rivedi-flottante-css';
          css.textContent = `
            #rivedi-flottante {{
              position: fixed; right: 18px; bottom: 88px; z-index: 2147483000;
              padding: 11px 18px; border-radius: 24px;
              border: 1px solid #c8cdd4; background: #ffffff; color: #1f2933;
              font-size: 15px; font-weight: 600; cursor: pointer;
              font-family: inherit; max-width: min(92vw, 340px);
              box-shadow: 0 3px 14px rgba(0,0,0,.22);
            }}
            #rivedi-flottante:hover {{ background: #f1f3f5; }}
            @media (max-width: 640px) {{
              #rivedi-flottante {{
                right: 12px; bottom: 90px;
                padding: 10px 15px; font-size: 14px;
              }}
            }}`;
          d.head.appendChild(css);
        }}

        const b = d.createElement('button');
        b.id = 'rivedi-flottante';
        b.type = 'button';
        b.textContent = '{etichetta}';

        b.onclick = function () {{
          // cerca il vero pulsante del modulo e lo preme
          const tutti = d.querySelectorAll('button');
          for (const x of tutti) {{
            if (x.id === 'rivedi-flottante') continue;
            const t = (x.innerText || '').trim();
            if (t.indexOf('Rivedi') !== -1) {{ x.click(); return; }}
          }}
        }};
        d.body.appendChild(b);
      }})();
    </script>
    """
    try:
        st.iframe(html, height=1)
    except AttributeError:
        import streamlit.components.v1 as components
        components.html(html, height=0)


def _avvisa_prima_di_uscire(attivo):
    """Fa comparire l'avviso del browser se si prova a ricaricare o chiudere.

    Streamlit perde tutto lo stato al ricaricamento della pagina: il
    partecipante ripartirebbe da capo. Il messaggio nativo del browser
    ("Vuoi davvero abbandonare il sito?") intercetta i casi accidentali, che
    sono la maggioranza. Il testo non e' personalizzabile: lo decidono i
    browser per evitare abusi.

    Va disattivato sulla pagina finale, altrimenti l'avviso comparirebbe anche
    a chi ha legittimamente finito.
    """
    stato = "on" if attivo else "off"
    if st.session_state.get("_avviso_uscita") == stato:
        return                      # gia' impostato, non ripetere l'iniezione
    st.session_state["_avviso_uscita"] = stato
    corpo = ("w.onbeforeunload = function (e) { e.preventDefault(); "
             "e.returnValue = ''; return ''; };" if attivo
             else "w.onbeforeunload = null;")
    html = f"""
    <script>
      (function () {{
        const w = window.parent;
        if (!w) return;
        {corpo}
      }})();
    </script>
    """
    try:
        st.iframe(html, height=1)
    except AttributeError:
        import streamlit.components.v1 as components
        components.html(html, height=0)


def _scroll_se_cambio_pagina():
    """Chiama _scroll_in_cima() solo quando si cambia davvero schermata.

    Un rerun causato dal click su un radio non deve far saltare la pagina in
    cima: si confronta quindi l'identita' della schermata corrente con quella
    disegnata l'ultima volta.
    """
    chiave = (st.session_state.get("step"),
              st.session_state.get("block_idx"),
              st.session_state.get("block_phase"))
    if st.session_state.get("_pagina_corrente") != chiave:
        st.session_state["_pagina_corrente"] = chiave
        _scroll_in_cima()


def _accumula_tempo(fase):
    """Somma il tempo della visita appena conclusa e riavvia il cronometro.

    Serve perche' il partecipante puo' passare piu' volte tra esposizione e
    domande: senza accumulo si registrerebbe solo la durata dell'ultima visita.
    """
    ora = time.time()
    trascorso = ora - st.session_state.t_phase
    if fase == "exposure":
        st.session_state.t_exposure += trascorso
    else:
        st.session_state.t_questions += trascorso
    st.session_state.t_phase = ora


def _inizia_blocco():
    """Azzera fase, cronometri, contatore dei ritorni e risposte salvate."""
    st.session_state.block_phase = "exposure"
    st.session_state.answers = {}
    st.session_state.revisits = 0
    st.session_state.t_exposure = 0.0
    st.session_state.t_questions = 0.0
    st.session_state.t_phase = time.time()


def features_table(values, feats=None):
    """Tabella a 2 colonne: Variabile | Valore (testo leggibile).

    feats: elenco delle variabili da mostrare. Se None le mostra tutte.
    """
    feats = FEATURES if feats is None else feats
    df = pd.DataFrame({
        "Variabile": [FEATURE_LABEL[f] for f in feats],
        "Valore":    [meaning(f, values[f]) for f in feats],
    }).set_index("Variabile")
    st.table(df)


def _shap_sorted(example):
    """Contributi ordinati per |valore| decrescente (il piu' grande in alto).

    Esclude le variabili nascoste per questo paziente, cosi' il grafico mostra
    esattamente le stesse righe della tabella del profilo.

    sorted() e' stabile: a parita' di valore assoluto vince l'ordine con cui
    le feature sono scritte in SHAP_LOCAL.
    """
    mostrate = set(visible_features(example))
    contributi = {f: v for f, v in SHAP_LOCAL[example["id"]].items()
                  if f in mostrate}
    return sorted(contributi.items(), key=lambda kv: -abs(kv[1]))


def _left_labels(ax, rows, values):
    """Etichette 'Nome variabile = valore' a sinistra dell'asse.

    Nome in nero grassetto, valore in grigio. Usa HPacker: e' matplotlib a
    calcolare gli ingombri, quindi i due pezzi non possono sovrapporsi
    qualunque sia il font o la risoluzione.
    """
    for y, f in enumerate(rows):
        nome = TextArea(FEATURE_SHORT[f], textprops=dict(
            color="#111111", fontsize=10.5, fontweight="bold"))
        val = TextArea(" = " + short_value(f, values[f]), textprops=dict(
            color="#999999", fontsize=10.5))
        pacco = HPacker(children=[nome, val], align="baseline", pad=0, sep=0)
        ax.add_artist(AnnotationBbox(
            pacco, (0.0, y), xybox=(-10, 0),
            xycoords=("axes fraction", "data"), boxcoords="offset points",
            box_alignment=(1.0, 0.5), frameon=False, annotation_clip=False))


def _verifica_shap_nascoste(example):
    """Avvisa se le variabili nascoste alterano il valore finale f(x).

    Nascondere variabili nel waterfall e' innocuo solo se i loro contributi si
    annullano: altrimenti la somma delle barre non arriva piu' alla probabilita'
    vera del modello, e il grafico direbbe una cosa falsa. L'avviso compare solo
    in modalita' sviluppo, cosi' i partecipanti non lo vedono mai.
    """
    nascoste = set(example.get("hide", ()))
    if not nascoste or not dev_mode():
        return
    scarto = sum(v for f, v in SHAP_LOCAL[example["id"]].items() if f in nascoste)
    if abs(scarto) > 0.005:
        st.warning(
            f"⚠️ {example['id']}: i contributi nascosti sommano {scarto:+.2f}, "
            f"quindi f(x) mostrato differisce di {scarto:+.2f} dalla probabilita' "
            "reale del modello. Scegli variabili i cui contributi si annullino.")


def plot_shap_waterfall(example):
    """Waterfall in stile libreria shap: da E[f(X)] fino a f(x)."""
    values = example["values"]
    items = _shap_sorted(example)
    n = len(items)
    bottom_up = list(reversed(items))          # y=0 in basso

    # accumulo dal basso: ogni barra riparte dove finisce la precedente
    starts, x = [], SHAP_BASE_VALUE
    for _, v in bottom_up:
        starts.append(x)
        x += v
    fx = x

    punti = [SHAP_BASE_VALUE] + [s + v for s, (_, v) in zip(starts, bottom_up)]
    lo, hi = min(punti), max(punti)
    span = (hi - lo) or 0.1
    xlim = (lo - span * 0.55, hi + span * 0.28)
    width = xlim[1] - xlim[0]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    tip = width * 0.012        # punta della freccia
    minw = width * 0.008       # larghezza minima per restare visibile
    h = 0.30                   # semi-altezza della barra

    for y, ((f, v), x0) in enumerate(zip(bottom_up, starts)):
        col = SHAP_RED if v >= 0 else SHAP_BLUE
        sgn = 1 if v >= 0 else -1
        w = max(abs(v), minw)
        x1 = x0 + sgn * w
        t = min(tip, w)
        ax.add_patch(Polygon(
            [(x0, y - h), (x1 - sgn * t, y - h), (x1, y),
             (x1 - sgn * t, y + h), (x0, y + h), (x0 + sgn * t, y)],
            closed=True, facecolor=col, edgecolor="white",
            linewidth=0.8, zorder=3))

        txt = f"{v:+.2f}".replace("+0.00", "+0").replace("-0.00", "+0")
        if w > width * 0.075:                       # etichetta dentro la barra
            ax.text(x0 + sgn * w / 2, y, txt, ha="center", va="center",
                    color="white", fontsize=10, zorder=4)
        else:                                       # etichetta fuori
            ax.text(x1 + sgn * width * 0.012, y, txt,
                    ha="left" if v >= 0 else "right", va="center",
                    color=col, fontsize=10, zorder=4)

        ax.axhline(y, color="#dddddd", linewidth=0.6,
                   linestyle=(0, (2, 3)), zorder=1)

    ax.axvline(SHAP_BASE_VALUE, color="#bbbbbb", linewidth=0.9, zorder=2)
    ax.axvline(fx, color="#bbbbbb", linewidth=0.9,
               linestyle=(0, (2, 2)), zorder=2)

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([""] * n)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#666")

    _left_labels(ax, [f for f, _ in bottom_up], values)

    ax.annotate(f"$f(x) = {fx:.3g}$", xy=(fx, 1.015),
                xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=11, color="#333")
    ax.annotate(f"$E[f(X)] = {SHAP_BASE_VALUE:.3g}$",
                xy=(SHAP_BASE_VALUE, -0.085),
                xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=11, color="#999")

    fig.subplots_adjust(left=0.46, right=0.97, top=0.90, bottom=0.16)
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)


def plot_shap_bars(example):
    """Barre semplici centrate sullo zero (non mostra la probabilita')."""
    values = example["values"]
    items = list(reversed(_shap_sorted(example)))   # max in alto con barh
    labels = [f"{short_value(f, values[f])} = {FEATURE_SHORT[f]}"
              for f, _ in items]
    vals = [v for _, v in items]
    colors = [SHAP_RED if v >= 0 else SHAP_BLUE for v in vals]

    fig, ax = plt.subplots(figsize=(7, 5.4))
    bars = ax.barh(labels, vals, color=colors, height=0.62)
    ax.axvline(0, color="#999", linewidth=0.9)
    span = max(abs(v) for v in vals) or 1.0
    for b, v in zip(bars, vals):
        off = span * 0.03
        ax.text(v + (off if v >= 0 else -off), b.get_y() + b.get_height() / 2,
                f"{v:+.2f}".replace("+0.00", "0.00"), va="center",
                ha="left" if v >= 0 else "right", fontsize=9,
                color=SHAP_RED if v >= 0 else SHAP_BLUE)
    ax.set_xlim(-span * 1.45, span * 1.45)
    ax.set_xlabel("Contributo alla decisione del modello\n"
                  "(rosso = spinge verso MALATO, blu = spinge verso SANO)",
                  fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#eee", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_shap_local(example):
    """Sceglie lo stile in base a SHAP_PLOT_STYLE."""
    if SHAP_PLOT_STYLE == "waterfall":
        plot_shap_waterfall(example)
    else:
        plot_shap_bars(example)


def plot_shap_global():
    """Importanza globale: attiva solo se SHOW_SHAP_GLOBAL = True."""
    if not SHAP_GLOBAL:
        st.warning("SHAP_GLOBAL non è ancora popolato.")
        return
    items = sorted(SHAP_GLOBAL.items(), key=lambda kv: kv[1])
    labels = [FEATURE_SHORT[k] for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 5.0))
    ax.barh(labels, vals, color="#7f8c8d")
    ax.set_xlabel("Importanza media (|SHAP|)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def verdetto(pred):
    """Previsione del modello evidenziata: Malato in rosso, Sano in verde."""
    colore = "red" if pred == "Malato" else "green"
    return f":{colore}-background[**{pred}**]"

def render_example(method, example):
    """UN paziente esempio: verdetto + tabella + (eventuale) spiegazione."""
    eid = example["id"]
    st.markdown(f"**Previsione del modello:** {verdetto(example['pred'])}")
    features_table(example["values"], visible_features(example))

    if method == BASELINE:
        return  # nessuna spiegazione: solo il verdetto nudo

    if method == "SHAP":
        st.markdown("**Quanto ha pesato ogni variabile su questa decisione:**")
        _verifica_shap_nascoste(example)
        plot_shap_local(example)

    elif method == "DiCE":
        st.markdown("##### Cosa cambierebbe la previsione (scenario \"what-if\")")
        rows = [(FEATURE_LABEL[f], meaning(f, cur), meaning(f, alt))
                for f, cur, alt in DICE_CF[eid]]
        st.table(pd.DataFrame(
            rows, columns=["Variabile", "Valore attuale", "Valore alternativo"]
        ).set_index("Variabile"))
        target = "Sano" if example["pred"] == "Malato" else "Malato"
        # testo pieno e non st.caption: la conclusione e' la parte piu'
        # importante della spiegazione controfattuale e deve risaltare
        st.markdown(f"#### Con queste modifiche la previsione diventerebbe: "
                    f"{verdetto(target)}")

    elif method == "Anchors":
        a = ANCHORS_RULE[eid]
        st.markdown("##### Regola usata dal modello per questo paziente")
        st.markdown("**SE** valgono tutte queste condizioni:")
        st.markdown("\n".join(f"- **{testo}**" for _, testo in a["rule"]))
        st.markdown(f"#### ALLORA la previsione è: {verdetto(a['pred'])}")
        st.caption(f"La regola è corretta nel {a['precision']:.0%} dei pazienti "
                   f"simili e si applica al {a['coverage']:.0%} dei pazienti.")


# =========================================================================
# INIZIALIZZAZIONE DELLO STATO
# =========================================================================
if "step" not in st.session_state:
    st.session_state.step = "consent"
    st.session_state.pid = uuid.uuid4().hex[:8]
    st.session_state.rows = []
    st.session_state.flushed = 0
    st.session_state.group = None
    st.session_state.blocks = None       # [Baseline, m1, m2, m3]
    st.session_state.block_idx = 0
    st.session_state.block_phase = "exposure"
    st.session_state.saved = False       # True dopo la scrittura finale
    st.session_state.storage = None      # "sheets" | "local"
    st.session_state.storage_error = None
    st.session_state.answers = {}        # risposte salvate (sopravvivono ai widget)
    st.session_state.revisits = 0        # quante volte e' tornato a rivedere
    st.session_state.t_exposure = 0.0    # tempo accumulato in esposizione
    st.session_state.t_questions = 0.0   # tempo accumulato sulle domande
    st.session_state.t_start = time.time()
    st.session_state.t_phase = time.time()


# =========================================================================
# PAGINE
# =========================================================================
def page_consent():
    st.title("Indagine sulla comprensione delle spiegazioni di modelli XAI (Explainable AI)")
    st.write(
        "In questo studio lavorerai con un **modello di machine "
        "learning** addestrato su dati clinici di pazienti reali. A partire "
        "da alcuni valori — età, pressione, colesterolo, esiti di esami del "
        "cuore — il modello stima se quella persona sia destinata a soffrire "
        "di problemi cardiaci (**MALATO**) oppure no (**SANO**). "
        "Restituisce però soltanto la risposta finale, senza dire nulla su "
        "come ci sia arrivato.\n\n"
        "Proprio per questo esistono delle **tecniche di spiegazione**: strumenti "
        "separati che prendono la decisione presa dal modello e provano a "
        "renderla comprensibile a una persona e a rendere chiaro il modo di ragionare del modello. Ne esistono di diversi tipi, e "
        "raccontano la stessa decisione in modi molto diversi tra loro.\n\n"
        "Vedrai prima alcune decisioni del modello da sole, poi accompagnate "
        "da tre tipi di spiegazione, uno alla volta. Ogni volta ti chiederemo "
        "di prevedere come il modello classificherebbe alcuni nuovi pazienti "
        "e, dove è presente una spiegazione, quanto ti sia sembrata utile.\n\n"
        "**Non serve alcuna competenza medica.** Non ti chiediamo di fare una "
        "diagnosi, ma di capire il ragionamento di un sistema di apprendimento automatico.\n\n"
        "La partecipazione è **anonima** e richiede circa **15-20 minuti**. "
        "Sei libero di interrompere in qualsiasi momento: in quel caso le tue "
        "risposte non verranno registrate.\n\n"
        "⚠️ **Completa il questionario in un'unica sessione.** Le risposte "
        "vengono salvate solo alla fine: se ricarichi o chiudi la pagina prima "
        "di arrivare in fondo, dovrai ricominciare da capo."
    )
    ok = st.checkbox("Ho letto le informazioni e acconsento a partecipare.")
    if st.button("Inizia", disabled=not ok):
        st.session_state.blocks = [BASELINE] + claim_balanced_order()
        st.session_state.step = "demographics"
        st.rerun()


def page_demographics():
    st.title("Qualche informazione su di te")
    st.caption("Nessun dato identificativo viene raccolto.")
    age = st.selectbox("Fascia d'età", ["18-24", "25-34", "35-49", "50+"])
    bg = st.radio(
        "Qual è il tuo background di studio/lavoro?",
        BACKGROUNDS,
    )
    fam = st.select_slider(
        "Quanta familiarità hai con i modelli di machine learning?",
        options=FAMILIARITA, value=FAMILIARITA[0],
    )
    med = st.radio(
        "Hai una formazione in ambito medico o sanitario?",
        ["No", "Sì"], horizontal=True,
    )
    if st.button("Prosegui"):
        st.session_state.group = classifica_gruppo(bg, fam)
        log("demographics", "-", "age", age)
        log("demographics", "-", "background", bg)
        log("demographics", "-", "familiarity", fam)
        log("demographics", "-", "medical_background", med)
        # nessuna scrittura qui: i dati partono tutti insieme alla fine
        st.session_state.step = "instructions"
        st.rerun()


def page_instructions():
    st.title("Come funziona")
    st.write(
        f"Lo studio è diviso in **{N_BLOCKS} parti**.\n\n"
        "**Nella prima parte** vedrai solo i dati clinici dei pazienti e la "
        "decisione del modello, senza alcuna spiegazione.\n\n"
        "**Nei tre blocchi successivi** vedrai tre diversi tipi di spiegazione, "
        "uno alla volta.\n\n"
        "In ogni blocco:\n\n"
        "1. osservi **2 pazienti di esempio** con la decisione del modello e una spiegazione di quella classificazione del paziente come malato o come sano;\n"
        "2. ti mostriamo **5 nuovi pazienti senza la classificazione del modello** e "
        "provi tu a prevederla, esprimendo il livello di sicurezza nella risposta;\n"
        "3. alla fine di ogni blocco ti verrà chiesto di valutare quanto la spiegazione ti è "
        "sembrata utile.\n\n"
        "Non esistono risposte 'giuste' nella parte di valutazione: ci interessa "
        "la tua impressione sincera sulla tipologia di spiegazione che ti è stata mostrata.\n\n"
        "---\n\n"
        "**Cosa significano le due risposte del modello**\n\n"
        "- **MALATO**: secondo il modello quella persona è destinata a "
        "soffrire di problemi cardiaci.\n"
        "- **SANO**: secondo il modello quella persona non ne soffrirà.\n\n"
        "Attenzione: si tratta della previsione del modello, non di una "
        "diagnosi medica. A te chiediamo di indovinare che cosa risponderebbe "
        "il modello, non che cosa sia giusto dal punto di vista clinico."
    )
    if st.button("Ho capito, inizia"):
        st.session_state.step = "block"
        _inizia_blocco()
        st.rerun()


def page_block():
    method = st.session_state.blocks[st.session_state.block_idx]
    pos = st.session_state.block_idx + 1
    is_baseline = (method == BASELINE)
    st.progress(pos / float(N_BLOCKS), text=f"Parte {pos} di {N_BLOCKS}")

    # ------------------------------------------------ FASE 1: esposizione
    if st.session_state.block_phase == "exposure":
        if is_baseline:
            st.title(f"Parte {pos}/{N_BLOCKS} - Solo i dati")
            st.info("Qui vedi soltanto i dati clinici del paziente e la decisione "
                    "del modello, senza alcuna spiegazione. Osserva con attenzione: "
                    "tra poco dovrai prevedere le decisioni del modello su nuovi "
                    "casi, avendo a disposizione solo i dati.\n\n"
                    "Con le freccette in alto a sinistra puoi aprire la finestra " \
                    "del glossario dei termini medici. Consultalo ogni volta che ne hai bisogno: "
                    "il questionario non mira a valutare competenze mediche.")
        else:
            st.title(f"Parte {pos}/{N_BLOCKS} - Tipo di spiegazione {pos - 1}/3")
            st.info("Osserva con attenzione: tra poco dovrai prevedere le decisioni "
                    "del modello su nuovi casi, con questo tipo di spiegazione.")
            intro = INTRO_METODO.get(method)
            if intro:
                st.subheader(intro["titolo"])
                st.markdown(intro["testo"])
            if method == "SHAP" and SHOW_SHAP_GLOBAL:
                st.subheader("Importanza generale delle variabili")
                plot_shap_global()

        if st.session_state.revisits > 0:
            st.success("Sei tornato alle spiegazioni. Le risposte che avevi già "
                       "dato sono state conservate.")

        esempi = EXAMPLES[method]
        for i, ex in enumerate(esempi, start=1):
            st.subheader(f"Esempio {i} di {len(esempi)}")
            render_example(method, ex)
            if i < len(esempi):
                st.divider()

        avanti = ("Torna alle domande" if st.session_state.revisits > 0
                  else "Ho capito, passo alle domande")
        if st.button(avanti):
            _accumula_tempo("exposure")
            st.session_state.block_phase = "questions"
            st.rerun()
        return

    # --------------------- FASE 2: simulatability (+ ESS se non baseline)
    st.title(f"Parte {pos}/{N_BLOCKS} - Le tue previsioni")
    st.write("Per ogni paziente, prevedi la decisione del modello. "
             "**La decisione del modello non è mostrata.**")
    st.caption("Ricorda: **Malato** = secondo il modello soffrirà di problemi "
               "cardiaci · **Sano** = secondo il modello non ne soffrirà.")

    test_items = TESTS[method]
    show_attention = (pos == ATTENTION_CHECK_AT_BLOCK)

    with st.form(key=f"form_block_{pos}"):
        # Il pulsante "indietro" e' un form_submit_button, non un st.button:
        # solo cosi' Streamlit invia i valori dei widget del form e le risposte
        # gia' date vengono conservate. Sta in alto perche' serve proprio quando
        # si sta guardando il primo paziente, senza dover scorrere fino in fondo.
        etichetta_rivedi = ("◀ Rivedi i dati di esempio" if is_baseline
                            else "◀ Rivedi la spiegazione")
        indietro = st.form_submit_button(
            etichetta_rivedi,
            help="Torni alla pagina precedente di QUESTA parte. "
                 "Le risposte già date vengono conservate.")
        st.caption("")

        OPZ_PRED = ["Malato", "Sano"]
        OPZ_CONF = list(CONFIDENCE.keys())
        OPZ_SAT = [NON_RISPOSTO] + list(LIKERT.keys())

        sim_answers, conf_answers = {}, {}
        for n, t in enumerate(test_items, start=1):
            st.markdown(f"#### Paziente {n} di {len(test_items)}")
            features_table(t["values"], visible_features(t))
            k_sim = f"sim_{pos}_{t['id']}"
            sim_answers[t["id"]] = st.radio(
                "Secondo te il modello prevede:", OPZ_PRED,
                key=k_sim, horizontal=True,
                index=_radio_index(k_sim, OPZ_PRED),
            )
            k_conf = f"conf_{pos}_{t['id']}"
            conf_answers[t["id"]] = st.radio(
                "Quanto sei sicuro/a della tua risposta?",
                options=OPZ_CONF,
                format_func=lambda x: CONFIDENCE[x],
                key=k_conf, horizontal=True,
                index=_radio_index(k_conf, OPZ_CONF),
            )
            st.divider()

        att = None
        if show_attention:
            st.markdown("#### Domanda di controllo")
            att = st.radio("Per verificare l'attenzione, seleziona 'Sano'.",
                           OPZ_PRED, key=f"att_{pos}", horizontal=True,
                           index=_radio_index(f"att_{pos}", OPZ_PRED))
            st.divider()

        sat = {}
        if not is_baseline:
            st.markdown("### Quanto ti è sembrata utile questa spiegazione?")
            st.caption("Trascina il cursore per rispondere a ogni affermazione.")
            # in modalita' sviluppo con precompilazione parte da una risposta
            # valida, altrimenti dal segnaposto "non risposto"
            iniziale = OPZ_SAT[1] if autofill() else NON_RISPOSTO
            for i, item in enumerate(SATISFACTION_ITEMS):
                k_sat = f"sat_{pos}_{i}"
                sat[i] = st.select_slider(
                    item, options=OPZ_SAT,
                    format_func=lambda x: LIKERT_SLIDER[x],
                    key=k_sat,
                    value=_slider_value(k_sat, OPZ_SAT, iniziale))

        submitted = st.form_submit_button("Conferma e prosegui")

    # gemello ancorato allo schermo, cosi' resta a portata di mano anche in
    # fondo alla pagina
    _pulsante_rivedi_flottante(True, etichetta_rivedi)

    def _memorizza():
        """Copia le risposte correnti nel dizionario persistente."""
        mappa = {}
        for t in test_items:
            mappa[f"sim_{pos}_{t['id']}"] = sim_answers[t["id"]]
            mappa[f"conf_{pos}_{t['id']}"] = conf_answers[t["id"]]
        if show_attention:
            mappa[f"att_{pos}"] = att
        for i in sat:
            mappa[f"sat_{pos}_{i}"] = sat[i]
        _salva_risposte(mappa)

    # "indietro": nessuna validazione, ma le risposte vanno messe al sicuro
    # prima che Streamlit distrugga i widget del form.
    if indietro:
        _memorizza()
        _accumula_tempo("questions")
        st.session_state.revisits += 1
        st.session_state.block_phase = "exposure"
        st.rerun()

    if submitted:
        mancanti = []
        if any(v is None for v in sim_answers.values()):
            mancanti.append("le previsioni")
        if any(v is None for v in conf_answers.values()):
            mancanti.append("i livelli di sicurezza")
        if show_attention and att is None:
            mancanti.append("la domanda di controllo")
        if not is_baseline and any(v in (None, NON_RISPOSTO)
                                   for v in sat.values()):
            mancanti.append("la valutazione della spiegazione")
        _memorizza()
        if mancanti:
            st.warning("Prima di proseguire completa: " + ", ".join(mancanti) + ".")
            return

        _accumula_tempo("questions")
        for t in test_items:
            resp = sim_answers[t["id"]]
            log("simulatability", method, t["id"], resp,
                correct=(resp == t["truth"]),
                confidence=conf_answers[t["id"]])
        # tempi TOTALI del blocco, somma di tutte le visite alle due fasi
        log("timing", method, "exposure_total", None,
            rt=round(st.session_state.t_exposure, 1))
        log("timing", method, "questions_total", None,
            rt=round(st.session_state.t_questions, 1))
        # quante volte e' tornato a rivedere: misura di quanto la spiegazione
        # sia stata memorizzabile o quanto sia servito riconsultarla
        log("behavior", method, "revisits", st.session_state.revisits)
        if show_attention:
            log("attention", method, "attention_check", att,
                correct=(att == "Sano"))
        for i in sorted(sat):
            log("satisfaction", method, f"ESS_{i + 1}", sat[i])

        if st.session_state.block_idx < N_BLOCKS - 1:
            st.session_state.block_idx += 1
            _inizia_blocco()
            st.session_state.step = "block"
        else:
            log("timing", "-", "total", None,
                rt=round(time.time() - st.session_state.t_start, 1))
            salva_tutto()          # unica scrittura, a questionario completato
            st.session_state.step = "done"
        st.rerun()


def page_done():
    # se la scrittura non era riuscita (rete lenta, quota Google), si ritenta
    # qui: il flag 'saved' garantisce che un secondo tentativo riuscito non
    # produca righe doppie
    ok = salva_tutto() or st.session_state.get("saved") or dev_mode()

    st.title("Grazie per aver partecipato! 🫀")
    if ok:
        st.write("Le tue risposte sono state registrate in forma anonima.")
        st.balloons()
    else:
        st.error("Non è stato possibile salvare le risposte per un problema "
                 "tecnico. **Non chiudere questa pagina**: premi il pulsante "
                 "qui sotto per riprovare.")
        if st.button("Riprova a salvare"):
            st.rerun()
    if st.session_state.get("storage_error") and ok:
        st.caption("Il salvataggio è andato a buon fine dopo un primo tentativo "
                   "non riuscito.")
    # L'anteprima dei dati serve solo a chi conduce lo studio: mostrarla ai
    # partecipanti li espone alle risposte corrette e alla struttura interna.
    # Compare quindi soltanto in modalita' sviluppo.
    if dev_mode():
        with st.expander("Anteprima dati salvati (solo per il ricercatore)"):
            dove = {"sheets": "Google Sheets", "local": "file locale responses.csv"}
            st.caption("Destinazione: " +
                       dove.get(st.session_state.get("storage"), "nessuna scrittura")
                       + (f" — errore: {st.session_state.storage_error}"
                          if st.session_state.get("storage_error") else ""))
            # la colonna 'response' mescola testo e numeri: la converto in
            # stringa per evitare avvisi di serializzazione nell'anteprima
            st.dataframe(pd.DataFrame(st.session_state.rows).astype(str))


# =========================================================================
# GLOSSARIO DEI TERMINI CLINICI
# =========================================================================
# Disponibile nella barra laterale dal blocco baseline in poi (cioe' su tutte
# le pagine dove il partecipante vede dati clinici).
GLOSSARY = [
    ("Angina",
     "Dolore transitorio al torace o sensazione di pressione che si manifesta "
     "quando il muscolo cardiaco non riceve una sufficiente quantità di ossigeno."),
    ("Pressione a riposo (trestbps)",
     "Pressione arteriosa misurata a riposo, in mmHg."),
    ("Colesterolo (chol)",
     "Colesterolo sierico nel sangue, in mg/dl."),
    ("Glicemia a digiuno (fbs)",
     "Indica se la glicemia a digiuno supera 120 mg/dl (sì/no)."),
    ("Elettrocardiogramma a riposo (restecg)",
     "Esito dell'ECG effettuato a riposo (normale o con anomalie)."),
    ("Frequenza cardiaca massima (thalach)",
     "Frequenza cardiaca massima raggiunta durante lo sforzo, in battiti al minuto."),
    ("Angina da sforzo (exang)",
     "Presenza di dolore toracico indotto dall'attività fisica (sì/no)."),
    ("Depressione ST (oldpeak)",
     "Abbassamento in millimetri del tratto ST nell'ECG durante lo sforzo "
     "rispetto al riposo: un valore più alto indica maggiore sofferenza cardiaca.\n\n"
     "Il tratto ST è un segmento del tracciato grafico "
     "dell'elettrocardiogramma (ECG)."),
    ("Pendenza del tratto ST (slope)",
     "Andamento del tratto ST durante il picco dello sforzo "
     "(in salita, piatto o in discesa)."),
    ("Vasi colorati (ca)",
     "Numero di vasi sanguigni principali (da 0 a 3) resi visibili tramite "
     "fluoroscopia."),
    ("Test di stress al tallio (thal)",
     "Tipo di difetto ematico rilevato (normale, difetto fisso o "
     "difetto reversibile)."),
]


def glossary_sidebar():
    """Glossario nella barra laterale, dal blocco baseline in poi."""
    if st.session_state.step != "block":
        return
    with st.sidebar:
        st.markdown("### 📖 Glossario")
        st.caption("Puoi consultarlo in qualsiasi momento. "
                   "Clicca su un termine per leggerne la definizione.")
        for termine, definizione in GLOSSARY:
            with st.expander(termine):
                st.write(definizione)
        st.divider()


# =========================================================================
# PANNELLO DI SVILUPPO  (visibile solo con ?dev=<chiave> nell'URL)
# =========================================================================
def dev_sidebar():
    if not dev_mode():
        return
    with st.sidebar:
        st.warning("🛠️ MODALITÀ SVILUPPO")
        st.caption("Le risposte **non** vengono salvate su responses.csv e il "
                   "contatore del bilanciamento non viene toccato.")

        st.checkbox("Precompila tutte le risposte", key="dev_autofill",
                    help="Riempie previsioni, confidenza e ESS con la prima "
                         "opzione, così basta premere 'Conferma e prosegui'.")

        st.divider()
        st.markdown("**Salta a…**")

        nomi_blocchi = [BASELINE] + list(ORDERS[0])
        blocco = st.selectbox("Blocco", nomi_blocchi, key="dev_block")
        fase = st.radio("Fase", ["esposizione", "domande"],
                        key="dev_phase", horizontal=True)

        if st.button("Vai al blocco"):
            st.session_state.blocks = nomi_blocchi
            st.session_state.group = st.session_state.group or "dev"
            st.session_state.block_idx = nomi_blocchi.index(blocco)
            _inizia_blocco()
            st.session_state.block_phase = (
                "exposure" if fase == "esposizione" else "questions")
            st.session_state.step = "block"
            st.rerun()

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Pagina finale"):
                st.session_state.blocks = st.session_state.blocks or nomi_blocchi
                st.session_state.step = "done"
                st.rerun()
        with col_b:
            if st.button("Ricomincia"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

        st.divider()
        import sys as _sys
        st.caption(f"Python in uso: `{_sys.executable}`")
        doc, motivo = _apri_documento()
        if doc is not None:
            st.caption(f"✅ Google Sheets collegato — «{doc.title}»")
        else:
            st.caption(f"⚠️ Google Sheets non attivo → si userebbe il CSV locale"
                       f"\n\nMotivo: {motivo}")
        if st.session_state.get("storage_error"):
            st.caption(f"ultimo errore: {st.session_state.storage_error}")
        st.caption(
            f"pid `{st.session_state.get('pid', '-')}` · "
            f"gruppo `{st.session_state.get('group', '-')}` · "
            f"righe in memoria `{len(st.session_state.get('rows', []))}`"
        )


# =========================================================================
# ROUTER
# =========================================================================
PAGES = {
    "consent": page_consent,
    "demographics": page_demographics,
    "instructions": page_instructions,
    "block": page_block,
    "done": page_done,
}

_scroll_se_cambio_pagina()
# l'avviso di uscita resta attivo durante tutto il questionario e sparisce
# quando il partecipante ha finito
_avvisa_prima_di_uscire(st.session_state.step not in ("consent", "done"))
# il pulsante flottante esiste solo nella fase domande: qui lo si rimuove da
# tutte le altre schermate, poi page_block lo ricrea dove serve
if not (st.session_state.step == "block"
        and st.session_state.get("block_phase") == "questions"):
    _pulsante_rivedi_flottante(False, "")
glossary_sidebar()
dev_sidebar()
PAGES[st.session_state.step]()