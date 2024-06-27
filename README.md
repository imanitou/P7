# Import Jupyter Notebook 

# 1. Configurer Git (Git Bash)
git config --global user.name imanitou

git config --global user.email shannaoffi@hotmail.com

# 3. Initialiser un dépôt Git
cd C:/Users/guill/Imane/P7

git init

# 4. Ajouter le notebook au dépôt
git add C:/Users/guill/Imane/P7/P7_20juin2024.ipynb

# 5. Commiter les changements
git commit -m "Ajout de mon Jupyter Notebook"

# 6. Créer un dépôt distant sur GitHub
# (À faire sur GitHub.com, puis copier l'URL du dépôt)

# 7. Lier le dépôt local au dépôt distant
git remote add origin https://github.com/imanitou/P7.git


# Pour changer de branche

git branch -a

git checkout main

# Si la branche n'existe pas localement

git fetch origin

git checkout -b main origin/main
# 8. Pousser les changements vers GitHub
git push -u origin main
