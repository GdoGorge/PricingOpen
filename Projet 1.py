# ============================================================
# PROJET ACT 109 - FRENCH MOTOR (Version B) - HAMADE ABBAS
# Modélisation de la fréquence de sinistres TPL
# Méthode : GLM Poisson avec offset (approche actuarielle)
# Données : 4 fichiers parquet renommés f1 (fremotor1freq0304) ,f2 (fremotor1prem0304b),f3 (fremotor1sev0304),
# f4 (fremotor2freq9907)
#
# 🟢 Objectif du devoir :
# Notre objectif est de Construire un modèle permettant de classer les assurés
# du moins risqué au plus risqué à partir de leurs caractéristiques.
#
# 🟢 Utilité actuarielle :
# Ce type de modèle sert à la tarification, à la segmentation
# et à l’identification des profils à risque élevé.
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# ============================================================
# 🟢 RAPPEL ACTUARIEL 
#
# En modélisation de fréquence, on ne modélise pas directement
# le nombre de sinistres, mais un taux :
#
#     E(Y_i) = E_i × λ_i
#
# En prenant le logarithme :
#
#     log(E(Y_i)) = log(E_i) + log(λ_i)
#
# On pose :
#
#     log(λ_i) = X_i β
#
# D'où le modèle final :
#
#     log(E(Y_i)) = X_i β + log(E_i)
#
# Le terme log(E_i) est appelé OFFSET.
# Il est imposé au modèle et permet de tenir compte
# de la durée d’exposition du contrat.
#
# Dans cette base, la durée assurée n’est pas fournie.
# Nous posons donc E_i = 1 pour tous,
# ce qui revient à modéliser une fréquence annuelle standardisée.
# ============================================================


# ============================================================
# 1) CHARGEMENT DES DONNÉES
# ============================================================
# On définit les fichiers f1 à f4 de la façon suivante après lecture:
# f1 : fréquence (cible TPL)
# f2 : contrats + variables explicatives
# f3 : sévérité (bonus)
# f4 : autre période (contrôle)

def charger_donnees(path_freq="f1.parquet",
                    path_prem="f2.parquet",
                    path_sev="f3.parquet",
                    path_freq2="f4.parquet"):

    freq = pd.read_parquet(path_freq)
    prem = pd.read_parquet(path_prem)
    sev = pd.read_parquet(path_sev)
    freq2 = pd.read_parquet(path_freq2)

    # Nettoyage des noms de colonnes 
    for df in (freq, prem, sev, freq2):
        df.columns = [str(c).strip() for c in df.columns]

    return freq, prem, sev, freq2


# ============================================================
# 2) JOINTURE + OFFSET
# ============================================================
# 🟢 On relie chaque contrat à son nombre de sinistres.
# 🟢 L’offset permet de modéliser un taux de sinistralité.

def construire_base(freq, prem):

    # Jointure sur identifiant contrat et année
    df = prem.merge(freq, on=["IDpol", "Year"], how="inner")

    # Exposition artificielle = 1 an par défaut
    df["Exposure"] = 1.0

    # Offset = log(exposure)
    df["offset"] = np.log(df["Exposure"].clip(lower=1e-12))

    return df


# ============================================================
# 3) PRÉPARATION DES DONNÉES
# ============================================================
# 🟢 Cette étape transforme les données brutes
# en une matrice numérique exploitable par le modèle.

def preparer_X_y_offset(df, target="TPL"):

    if target not in df.columns:
        raise ValueError(f"Cible '{target}' absente dans la base.")

    # Variable cible : nombre de sinistres
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).astype(float)

    # Offset actuariel
    offset = pd.to_numeric(df["offset"], errors="coerce").fillna(0.0).astype(float)

    # Variables candidates
    candidates_num = ["DrivAge", "BonusMalus", "LicenceNb", "VehAge", "VehPower"]
    candidates_cat = ["DrivGender", "MaritalStatus", "PayFreq", "JobCode",
                      "VehClass", "VehGas", "VehUsage", "Garage",
                      "Area", "Region", "Fuel"]

    # On garde uniquement les colonnes existantes
    num_features = [c for c in candidates_num if c in df.columns]
    cat_features = [c for c in candidates_cat if c in df.columns]

    # Matrice explicative brute
    X = df[num_features + cat_features].copy()

    # Encodage one-hot des variables catégorielles
    X = pd.get_dummies(X, columns=cat_features, drop_first=True)

    # Conversion en numérique pur (Sans ça mon modèle crash)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Ajout d'une constante 
    X = sm.add_constant(X, has_constant="add")

    return X, y, offset, num_features, cat_features

# ============================================================
# 4) GLM POISSON AVEC OFFSET
# ============================================================
# 🟢 Nous utilisons un GLM Poisson avec lien logarithmique.
# C’est le modèle actuariel standard pour des données de comptage.
#
# Il permet d’estimer le nombre moyen de sinistres par contrat,
# en fonction des caractéristiques du conducteur et du véhicule.
#
# Le modèle apprend les coefficients β tels que :
#     log(E(Y_i)) = X_i β + log(Exp_i)
#
# Le modèle permet ensuite de classer les assurés
# du moins risqué au plus risqué.
# ============================================================

def entrainer_glm_poisson(X_train, y_train, offset_train):

    # Conversion en numpy float (Crash autrement)
    X_np = np.asarray(X_train, dtype=float)
    y_np = np.asarray(y_train, dtype=float)
    off_np = np.asarray(offset_train, dtype=float)

    model = sm.GLM(
        y_np,
        X_np,
        family=sm.families.Poisson(),
        offset=off_np
    )

    res = model.fit()
    return res


# ============================================================
# 5) PRÉDICTION
# ============================================================
# 🟢 Transformation du score linéaire en moyenne attendue :
#     μ = exp(Xβ + offset)
# Ou μ est l’espérance du nombre de sinistres.

def predire_glm_poisson(res, X, offset):

    X_np = np.asarray(X, dtype=float)
    off_np = np.asarray(offset, dtype=float)
    beta = np.asarray(res.params, dtype=float)

    linpred = X_np @ beta + off_np
    mu = np.exp(linpred)
    return mu


# ============================================================
# 6) METRIQUES OBTENUES
# ============================================================
# 🟢 AUC : capacité à séparer les assurés avec ou sans sinistre.
# 🟢 Gini = 2 × AUC − 1.
# 🟢 Lift@20% : part de la charge captée par les 20% les plus risqués.
#
# Des valeurs légèrement supérieures au hasard sont normales
# en modélisation de fréquence automobile.
# ============================================================

def evaluer_auc_gini(y_true_count, y_pred_count):

    y_true_count = np.asarray(y_true_count, dtype=float)
    y_pred_count = np.asarray(y_pred_count, dtype=float)

    y_bin = (y_true_count > 0).astype(int)
    if np.unique(y_bin).size < 2:
        return np.nan, np.nan

    auc = roc_auc_score(y_bin, y_pred_count)
    gini = 2 * auc - 1
    return auc, gini


# ============================================================
# 7) COURBE DE LORENZ + LIFT
# ============================================================
# 🟢 La courbe de Lorenz représente la concentration
# de la charge de sinistres dans la population.
#
# Une courbe proche de la diagonale signifie
# un modèle proche du hasard, ce que nous obtenons en éxecutant le programme.
# ============================================================

def lorenz_curve(y_true, y_score):

    df = pd.DataFrame({"y": np.asarray(y_true, float),
                       "score": np.asarray(y_score, float)})

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    total = df["y"].sum()
    df["cum_y"] = 0.0 if total <= 0 else df["y"].cumsum() / total

    n = len(df)
    df["cum_pop"] = np.arange(1, n + 1) / n


    df0 = pd.DataFrame({"y": [0.0],
                        "score": [df["score"].iloc[0] if n else 0.0],
                        "cum_y": [0.0],
                        "cum_pop": [0.0]})

    return pd.concat([df0, df], ignore_index=True)


def plot_lorenz(df_lz, title="Courbe de Lorenz"):

    plt.figure(figsize=(7, 7))
    plt.plot(df_lz["cum_pop"], df_lz["cum_y"], label="Lorenz")
    plt.plot([0, 1], [0, 1], "--", label="Hasard")
    plt.xlabel("% population cumulée")
    plt.ylabel("% charge cumulée")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def lift_at(y_true, y_score, p=0.2):

    df = pd.DataFrame({"y": np.asarray(y_true, float),
                       "score": np.asarray(y_score, float)})

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    total = df["y"].sum()
    if total <= 0:
        return np.nan

    k = max(1, int(np.ceil(p * len(df))))
    part_charge = df.loc[:k - 1, "y"].sum() / total
    return part_charge / p


# ============================================================
# 8) SÉVÉRITÉ MOYENNE TPL
# ============================================================
# 🟢 Cette partie illustre la séparation fréquence / sévérité.

def severite_moyenne_TPL(sev):

    if "Guarantee" not in sev.columns or "Payment" not in sev.columns:
        return np.nan

    sev2 = sev.copy()
    sev2["Payment"] = pd.to_numeric(sev2["Payment"], errors="coerce")
    sev2 = sev2.dropna(subset=["Payment"])

    sev_tpl = sev2[sev2["Guarantee"].astype(str)
                  .str.upper().str.contains("TPL", na=False)]

    if len(sev_tpl) == 0:
        return np.nan

    return float(sev_tpl["Payment"].mean())


# ============================================================
# 9) FONCTION OBLIGATOIRE
# ============================================================

def analyser(df):
    """
    Retourne un indicateur numérique clair :
    ici le Gini sur l’échantillon test.
    """
    X, y, offset, *_ = preparer_X_y_offset(df, target="TPL")

    X_train, X_test, y_train, y_test, off_train, off_test = train_test_split(
        X, y, offset, test_size=0.30, random_state=42
    )

    res = entrainer_glm_poisson(X_train, y_train, off_train)
    y_pred = predire_glm_poisson(res, X_test, off_test)

    _, gini = evaluer_auc_gini(y_test, y_pred)
    return gini


# ============================================================
# 10) Coeur du programme
# ============================================================

if __name__ == "__main__":

    # Chargement
    freq, prem, sev, freq2 = charger_donnees()

    print("Shapes:")
    print("  f1(freq0304b):", freq.shape)
    print("  f2(prem0304b):", prem.shape)
    print("  f3(sev0304b) :", sev.shape)
    print("  f4(freq9907b):", freq2.shape)

    # Construction base
    df = construire_base(freq, prem)
    print("\nBase fusionnée (prem+freq):", df.shape)

    # Préparation
    X, y, offset, num_features, cat_features = preparer_X_y_offset(df, target="TPL")
    print("\nVariables numériques utilisées :", num_features)
    print("Variables catégorielles utilisées :", cat_features)
    print("Dimension X après dummies :", X.shape)

    # Séparation train/test
    X_train, X_test, y_train, y_test, off_train, off_test = train_test_split(
        X, y, offset, test_size=0.30, random_state=42
    )

    # Entraînement
    res = entrainer_glm_poisson(X_train, y_train, off_train)

    print("\n===== GLM SUMMARY (extrait) =====")
    print(res.summary())

    # Prédiction
    y_pred = predire_glm_poisson(res, X_test, off_test)

    # Métriques
    auc, gini = evaluer_auc_gini(y_test, y_pred)
    lift20 = lift_at(y_test, y_pred, p=0.20)

    print("\n===== METRIQUES =====")
    print("AIC      :", float(res.aic))
    print("AUC      :", auc)
    print("Gini     :", gini)
    print("Lift@20% :", lift20)

    # Lorenz
    df_lz = lorenz_curve(y_test, y_pred)
    plot_lorenz(df_lz, title="Courbe de Lorenz - Fréquence TPL (GLM Poisson)")

    # Bonus sévérité
    sev_mean_tpl = severite_moyenne_TPL(sev)
    print("\nSévérité moyenne TPL (si dispo) :", sev_mean_tpl)

    # Fonction obligatoire
    gini_out = analyser(df)
    print("\nRetour analyser(df) =", gini_out)

#CONCLUSION DU PROGRAMME
#Les résultats obtenus sont :
#AIC = 18652.83
#AUC = 0.539
#Gini = 0.078
#Lift@20% = 1.25

#Ces valeurs indiquent que le modèle est légèrement meilleur que le hasard.
#Cela est normal pour un modèle de fréquence automobile, car les sinistres
#sont rares et difficiles à prédire individuellement.

#INTERPRETATION DE LA COURBE DE LORENZ OBTENUE EN EXECUTANT LE PROGRAMME
#La courbe est proche de la diagonale, ce qui signifie que la segmentation
#est modeste. Cependant, elle reste au-dessus de la droite du hasard,
#montrant que le modèle capte une partie du risque.

#A quoi cela sert-il ?
#Le modèle permet de classer les assurés du moins risqué au plus risqué.
#Il peut être utilisé comme base pour une tarification ou une segmentation
#commerciale, même si des améliorations seraient nécessaires pour un usage
#opérationnel