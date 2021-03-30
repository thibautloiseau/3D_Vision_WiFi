import process_csi as process
import matplotlib.pyplot as plt
import numpy as np

class reflecteur:
    """Pour les fichiers d'acquisition avec le réflecteur"""
    def __init__(self, path1, path2):
        """On initialise les deux fichiers CSI dont on va se servir pour localiser le réflecteur par 'triangularisation'"""
        self.path1 = path1
        self.path2 = path2

        # On récupère les CSI des deux fichiers
        self.CSI1 = process.CSI(self.path1)
        self.CSI2 = process.CSI(self.path2)

        # On initialise les distances entre les 2 routeurs, donnée que l'on doit avoir pour récupérer la position du réflecteur
        self.R1 = (0, 0)
        self.R2 = (300, 0)
        self.reflecteur = (155, 75)

    def localisation_ref(self):
        """Localisation du réflecteur à partir des deux fichiers"""
        pass

    def plot(self):
        plt.figure()
        plt.scatter(self.R1[0], self.R1[1], c='b')
        plt.annotate("R25", self.R1)
        plt.scatter(self.R2[0], self.R2[1], c='b')
        plt.annotate("R27", self.R2)
        plt.scatter(self.reflecteur[0], self.reflecteur[1], c='b')
        plt.annotate("réflecteur", self.reflecteur)
        plt.show()



