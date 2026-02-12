# 🦸 Projet FCUNIST – Classification d’Images avec ResNet50

Projet de Deep Learning pour la classification multi-classes d’images à l’aide de PyTorch et du Transfer Learning.

---

## 📌 Description du projet

Ce projet implémente une pipeline complète de classification d’images sur le dataset **FCUNIST** (15 classes de personnages fictifs).

Objectifs :

- Entraîner un modèle CNN pour la classification multi-classes
- Utiliser le Transfer Learning avec ResNet50 pré-entraîné
- Effectuer une validation robuste
- Générer des prédictions sur un jeu de test
- Soumettre les résultats sur un leaderboard

---

## 🧠 Architecture du modèle

Le modèle utilisé est **ResNet50 pré-entraîné sur ImageNet**.

### 🔹 Principe : Transfer Learning

Au lieu d’entraîner un réseau depuis zéro :

- On conserve les couches convolutionnelles pré-entraînées
- On remplace la dernière couche fully connected
- On adapte la sortie au nombre de classes du dataset

Couche finale :
```python
model.fc = nn.Linear(2048, 15)
```

Fonction de perte :
```python
nn.CrossEntropyLoss()
```

Optimiseur :
```python
optim.Adam(model.parameters(), lr=3e-5, weight_decay=3e-4)
```

Scheduler :
```python
optim.lr_scheduler.ReduceLROnPlateau(...)
```

---
---

## 📂 Structure du Dataset

```
FCUNIST/
├── train/
│   ├── batman/
│   ├── blackpanther/
│   ├── blackwidow/
│   └── ...
└── test/
    ├── 0001.png
    ├── 0002.png
    └── ...
```
- Les images d’entraînement sont organisées par classe.
- Les images de test sont stockées dans un dossier unique (structure flat).

---
## ⚙️ Prétraitement

Images redimensionnées en **224×224** (format attendu par ResNet).

- Augmentations sur le train (flip, rotation, jitter)
- Normalisation avec les statistiques ImageNet
- Pas d’augmentation sur validation / test

---

## 🔁 Entraînement

- Split 65% Train / 35% Validation
- Sauvegarde automatique du meilleur modèle (`best_model.pth`)
- Génération des courbes Loss / Accuracy

---
## 🏆 Soumission Leaderboard

Les prédictions du test sont générées :

- Format CSV (Top-1)
- Format JSON (scores complets par classe)

Soumission via :

```python
lb.submit_test(predictions_dict)
```

---

## 📊 Résultats

- Accuracy validation ≈ 99%
- Convergence stable
- Bonne généralisation
