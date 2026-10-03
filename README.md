# 🎯 Lead Scoring — Prédire la conversion des prospects avec la régression logistique

Projet de **classification binaire** : prédire si un prospect (*lead*) va devenir client.

> 📚 **Basé sur le cours [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) (DataTalks.Club) — module Classification.**
> J'ai appliqué la méthode du cours sur un autre jeu de données que celui des cours : le *Lead Scoring* de X Education.

---

##  Problème métier

- Une entreprise de formation reçoit beaucoup de prospects.
- Appeler tout le monde coûte du temps.
- Le modèle donne une **probabilité de conversion** pour chaque prospect.
- Les commerciaux appellent d'abord ceux qui ont la plus forte probabilité.

---

##  Dataset

- **Fichier :** `Leads.csv` (Lead Scoring, X Education, disponible sur Kaggle)
- **Taille :** 9 240 prospects, 37 colonnes
- **Cible :** `converted` (1 = devenu client, 0 = non)
- **Taux de conversion :** environ **38 %**

### Types de variables

| Type | Exemples |
|---|---|
| Numériques | `totalvisits`, `total_time_spent_on_website`, `page_views_per_visit`, `asymmetrique_activity_score`, `asymmetrique_profile_score` |
| Catégorielles | `lead_origin`, `lead_source`, `last_activity`, `specialization`, `tags`, `lead_quality`, `city`... (29 colonnes) |
| Identifiants (retirés) | `prospect_id`, `lead_number` |

---

##  Étapes du notebook

### 1. Nettoyage
- Noms de colonnes en minuscules, espaces remplacés par `_`.
- Valeurs texte en minuscules, espaces remplacés par `_`.

### 2. Valeurs manquantes
- Variables catégorielles → remplies avec `'missing'`.
- Variables numériques → remplies avec `0`.
- Garder `'missing'` comme catégorie permet au modèle d'apprendre si l'absence d'info est significative.

### 3. Cadre de validation
- **Train / validation / test = 60 % / 20 % / 20 %**
- `random_state=1` pour des résultats reproductibles.
- La cible `converted` est retirée des features (pas de fuite).

### 4. Importance des variables
- **Taux de conversion par groupe** : différence et *risk ratio*.
- **Information mutuelle** (variables catégorielles).
- **Corrélation** (variables numériques).

### 5. One-Hot Encoding
- `DictVectorizer` : `fit` sur le train, `transform` sur validation et test.

### 6. Modèle
- `LogisticRegression(max_iter=1000)`
- Seuil de décision : **0,5** sur `predict_proba`.

### 7. Entraînement final
- Ré-entraînement sur **train + validation**.
- Évaluation sur le **test**.
- Prédiction pour un client individuel.

---

##  Deux décisions importantes

###  1. Suppression de `lead_number`

**Pourquoi la supprimer ?**
- C'est un **identifiant** : chaque ligne a une valeur unique (7 392 valeurs pour 7 392 lignes).
- Il ne décrit pas le prospect, il l'identifie seulement.
- Sa corrélation avec `converted` est quasi nulle (≈ 0,02).

**Pourquoi ça fait du mal au modèle ?**
- Ses valeurs sont énormes (~600 000), alors que les colonnes One-Hot valent 0 ou 1.
- Analogie : un haut-parleur placé à côté de 100 personnes qui chuchotent.
- Le solveur a du mal à ajuster les poids → mauvaise convergence.

**Règle retenue :** retirer toute colonne dont `nunique() == len(df)` **et** qui représente une identité (pas une mesure).

###  2. Augmentation de `max_iter`

- La régression logistique cherche ses poids **étape par étape**.
- `max_iter` = nombre maximum d'étapes (**100 par défaut**).
- Analogie : descendre une montagne dans le brouillard avec un nombre de pas limité.
- Si on s'arrête trop tôt, les poids sont arbitraires → résultats instables.
- Avec `max_iter=1000`, le modèle a le temps de finir.

---

##  Expérience : quel effet de chaque décision ?

Même split (`random_state=1`), seuls `lead_number` et `max_iter` changent :

| `lead_number` | `max_iter` | Validation | Test |
|---|---|---|---|
| gardé | 100 | 83,5 % | 76,9 % |
| gardé | 1000 | 89,4 % | 90,8 % |
| supprimé | 100 | 89,2 % | 90,0 % |
| **supprimé** | **1000** | **≈ 93,9 %** | **≈ 94,5 %** |

### Ce qu'on en retire
- Chaque décision aide **séparément** : les deux ensemble donnent le meilleur résultat.
- Avec la configuration initiale, l'écart validation/test était de **~6,6 points** (signe d'un modèle instable).
- Après les corrections, l'écart passe à **0,6 point**.

---


### Autres limites
- Un `ConvergenceWarning` reste affiché : les colonnes numériques ne sont pas mises à l'échelle.


---

##  Notions apprises

- Cadre de validation (train / val / test) et fuite de données.
- Taux, différence et *risk ratio* par groupe.
- Information mutuelle et corrélation.
- One-Hot Encoding avec `DictVectorizer`.
- Régression logistique : prédiction dure vs douce.
- Rôle de `max_iter` et de la convergence.
- Repérer et supprimer un identifiant.

---

##  Structure du dépôt

```
.
├── data_prep.ipynb      # Notebook principal
├── Leads.csv            # Dataset
├── requirements.txt     # Dépendances
└── README.md
```

---



### `requirements.txt`

```
pandas
numpy
scikit-learn
jupyter
```



---


---

##  Remerciements

- [DataTalks.Club](https://datatalks.club/) pour le cours **ML Zoomcamp**.
- X Education pour le jeu de données (via Kaggle).

##  Licence

Projet pédagogique. Le dataset appartient à ses auteurs d'origine.
