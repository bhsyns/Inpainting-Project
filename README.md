# Inpainting-Project

## Fait par : Mohamed Ali SRIR et Youness Bahous dans le cadre du projet IMA201
<p align = "center" >
<img src="./output/fiche.gif">
</p>
Ce projet est une implémentation et une examination de la méthode proposé par l’article [1], l’ensemble
de codes Python de ce projet sont disponilbles sur ce dépot Git.

Il s’agit d’un algorithme qui sert à remplir des régions inconnues d’une image en se basant sur son
contenu en choisissant un ordre de remplissage et des patches permettant de bien propager les structures
de l’image et de prioriser les pixels pour lesquels on a plus d’information sure.

C’est une méthode particulièrement efficace pour la suppression d’objets en les remplaçant par des
fonds convainquant,et qui sert aussi à la complétion de textures.

On a implémenté une interface graphique pour tester l'algorithme.

[1] A. Criminisi, P. Perez, and K. Toyama. Object removal by exemplar-based inpainting. In 2003 IEEE
Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings., vol-
ume 2, pages II–II, 2003.
