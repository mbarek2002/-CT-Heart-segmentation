# Heart Image Preprocessing

Script Python de preparation d'images medicales et de masques pour un projet de deep learning.

## Installation

Creer et activer l'environnement virtuel :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installer les dependances :

```powershell
pip install -r requirements.txt
```

## Donnees

Le script attend le dataset extrait dans :

```text
C:\Users\ihebm\Downloads\data\train
```

La structure attendue est :

```text
data/
└── train/
    └── <patient>/
        ├── image/
        │   └── *.png
        └── mask/
            └── *.png
```

Si le dataset est encore archive :

```powershell
Expand-Archive -LiteralPath 'C:\Users\ihebm\Downloads\data.zip' -DestinationPath 'C:\Users\ihebm\Downloads' -Force
```

## Execution

Depuis le dossier du projet :

```powershell
python data.py
```

Le script :

- separe les donnees en entrainement et validation ;
- applique des augmentations horizontales, verticales et par rotation aux donnees d'entrainement ;
- redimensionne les images et les masques en `512x512` ;
- genere les resultats dans `data/train/` et `data/valid/`.

## Fichiers principaux

- `data.py` : chargement, separation et augmentation des donnees ;
- `requirements.txt` : dependances Python ;
- `.gitignore` : fichiers exclus du suivi Git.
