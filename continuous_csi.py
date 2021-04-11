import numpy as np
import matplotlib.pyplot as plt

class CSI:

    def __init__(self, path):
        self.path = path
        self.params = {
            'Nrx': 3,
            'Ntx': 1,
            'Nsubcarriers': 114,
        }

    ####################################################################################################################

    def get_raw_data(self):
        """Récupérer les CSI brutes"""
        res = np.load(self.path)

        return res

    def get_raw_amp(self):
        """Récupérer les amplitudes brutes"""
        res = self.get_raw_data()

        return np.abs(res)

    def get_raw_phase(self):
        """Récupérer les phases brutes"""
        res = self.get_raw_data()

        return np.unwrap(np.angle(res))

    ####################################################################################################################
    # Amplitude processing
    def process_amp(self):
        """Traiter l'amplitude des CSI"""
        res = self.get_raw_amp()

        amp_means = np.zeros(shape=(self.params['Nrx'], self.params['Ntx'], self.params['Nsubcarriers']))
        amp_std = np.zeros(shape=(self.params['Nrx'], self.params['Ntx'], self.params['Nsubcarriers']))
        amp_filters = np.zeros(shape=(res.shape[0], self.params['Nrx'], self.params['Ntx'], self.params['Nsubcarriers']))

        for i in range(res.shape[1]):
            for j in range(res.shape[2]):
                amp_means[i, j] = [np.mean(res[:, i, j, k]) for k in range(res.shape[3])]
                amp_std[i, j] = [np.std(res[:, i, j, k]) for k in range(res.shape[3])]

                # On garde les valeurs brutes pour les premières valeurs et les dernières qui ne sont pas moyennées avec les deux voisines
                amp_filters[0, i, j, :] = res[0, i, j, :]
                amp_filters[-1, i, j, :] = res[-1, i, j, :]

                # On traite toutes les autres valeurs en moyennant sur les deux voisines
                for k in range(1, res.shape[0] - 1):
                    amp_filter = 1 / 3 * (res[k - 1, i, j, :] + res[k, i, j, :] + res[k + 1, i, j, :])
                    if np.cov(amp_filter, amp_means[i, j])[0, 1] > 0:
                        amp_filters[k, i, j, :] = amp_filter

        return amp_filters

    ####################################################################################################################
    # Phase processing
    def process_phase(self):
        """Traiter la phase des CSI"""
        res = self.get_raw_phase()

        corrected_phases = np.zeros(shape=res.shape)

        for k in range(res.shape[0]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    slope = (res[k, i, j, -1] - res[k, i, j, 0]) / (res.shape[3] - 1)
                    intercept = np.mean(res[k, i, j, :])
                    corrected_phases[k, i, j] = res[k, i, j] - slope * np.array([i for i in range(res.shape[3])]) - intercept

        return np.unwrap(corrected_phases)

    ####################################################################################################################
    # Algorithmes MUSIC pour trouver les directions d'arrivée

    def noise_subspaces(self):
        """Récupérer le sous-espace bruit pour chaque paquet"""
        amps = self.process_amp()
        phases = self.process_phase()

        proc_csi = amps * np.exp(1j * phases)
        noise_subspaces = np.zeros(shape=(proc_csi.shape[0], self.params['Nrx'], 2), dtype=complex)

        # Pour chaque paquet, on récupère le sous-espace bruit
        for i in range(proc_csi.shape[0]):
            paquet = proc_csi[i]
            paquet = np.reshape(paquet, (paquet.shape[0] * paquet.shape[1], paquet.shape[2]))

            Rx = paquet @ np.conj(paquet.T)

            eigvals, eigvecs = np.linalg.eig(Rx)

            # On ordonne les val p et les vec p par ordre décroissant pour récupérer le sous espace bruit constitué des deux vec p issues des deux val p les plus petites
            idx = eigvals.argsort()[::-1]
            eigvecs = eigvecs[:, idx]

            noise_subspace = eigvecs[:, -2:]
            noise_subspaces[i] = noise_subspace

        return noise_subspaces

    def a(self, theta, d_antenne):
        freq = 5805e6
        c = 2.9972e8
        return (np.array([np.power(np.exp(-1j * 2 * np.pi * freq * d_antenne * np.sin(theta) / c), k) for k in
                          range(self.params["Nrx"])]))

    def pseudo_spectrum(self, d_antenne):
        """Récupérer les pseudo-spectre issu de l'algorithme MUSIC pour chaque paquet"""
        noise_subspaces = self.noise_subspaces()
        thetas = np.linspace(-np.pi, np.pi, 360)
        pseudo_spectrums = np.zeros(shape=(noise_subspaces.shape[0], thetas.shape[0]))
        maximums = np.zeros(shape=(noise_subspaces.shape[0]))

        for i in range(noise_subspaces.shape[0]):
            print(i)
            pseudo_spectrum = np.zeros(shape=thetas.shape)

            for j in range(thetas.shape[0]):
                pseudo_spectrum[j] = np.reciprocal(np.abs(
                    np.conj(self.a(thetas[j], d_antenne)).T @ noise_subspaces[i] @ np.conj(
                        noise_subspaces[i]).T @ self.a(thetas[j], d_antenne)))

            pseudo_spectrums[i] = pseudo_spectrum

            maximums[i] = 180 / np.pi * thetas[np.argmax(pseudo_spectrum)]

        return (maximums)

    def plot_pseudo_spectrum(self):
        """Tracé du pseudo-spectre de l'algorithme MUSIC"""
        pseudo_spectrum = self.pseudo_spectrum()

        plt.figure()
        plt.title(self.path + "\nMaximum : " + str(pseudo_spectrum[2]) + "$\pm$" + str(pseudo_spectrum[3]))
        plt.plot(180 / np.pi * pseudo_spectrum[0], pseudo_spectrum[1], '+')
        plt.show()

        return 0

    ####################################################################################################################
    # Méthode MMP

    def Ce_matrix(self, paquet):
        """Création de la matrice Ce à partir d'un seul paquet"""
        res = self.get_raw_data()[paquet]

        m = self.params["Nsubcarriers"] // 2 # Nombre de lignes des sous-matrices
        r = self.params["Nsubcarriers"] // 2 + 1 # Nombre de colonnes des sous-matrices

        # On récupère les CSI de chaque antenne
        c1 = np.reshape(res[0].T, (res[0].T.shape[0]*res[0].T.shape[1]))
        c2 = np.reshape(res[1].T, (res[1].T.shape[0]*res[1].T.shape[1]))
        c3 = np.reshape(res[2].T, (res[2].T.shape[0]*res[2].T.shape[1]))

        # On initialise les matrices composantes de Ce
        C1 = np.zeros(shape=(m, r), dtype=complex)
        C2 = np.zeros(shape=(m, r), dtype=complex)
        C3 = np.zeros(shape=(m, r), dtype=complex)

        # On remplit les matrices que l'on vient d'initialiser
        for i in range(m):
            C1[i] = c1[i: i+r]
            C2[i] = c2[i: i+r]
            C3[i] = c3[i: i+r]

        Ce = np.concatenate((np.concatenate((C1, C2), axis=1), np.concatenate((C2, C3), axis=1)))

        return Ce

    def DoA_MMP(self, paquet, d_antenne):
        """Méthode MMP pour retrouver les directions d'arrivée"""
        Ce = self.Ce_matrix(paquet)
        m = self.params["Nsubcarriers"]//2
        long_onde = 2.99792e8/5805e6 # c/f

        U, _, _ = np.linalg.svd(Ce) # Les valeurs et vecteurs singuliers sont classés par ordre décroissant

        # On décompose U en deux matrices, une Us qui comprend les deux vecteurs correspondant aux deux valeurs singulières les plus grandes, l'autre Uv qui comprend le reste
        # On ne considère que Us pour les calculs
        Us = U[:, :2]

        # On redécompose Us en Us1 et Us2 avec rang(Us1) = 2, le nombre de DoA que l'on veut distinguer
        Us1 = Us[m:, :]
        Us2 = Us[:m, :]

        # On écrit notre matrice qui correspondra au sous-espace signal et on calcule les valeurs propres
        M = np.linalg.pinv(Us1) @ Us2
        eigvals, _ = np.linalg.eig(M)

        thetas = 180/np.pi * np.unwrap((np.arcsin(-long_onde*np.angle(eigvals) / (2*np.pi*d_antenne))))

        return thetas