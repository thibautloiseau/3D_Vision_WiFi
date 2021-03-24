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
    def process_phase(self, rx, tx, subcarrier):
        """Traiter la phase des CSI"""
        res = self.get_raw_phase()

        corrected_phases = np.zeros(shape=res.shape)

        for k in range(res.shape[0]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    slope = (res[k, i, j, -1] - res[k, i, j, 0]) / (res.shape[3] - 1)
                    intercept = np.mean(res[k, i, j, :])
                    corrected_phases[k, i, j] = res[k, i, j] - slope * np.array([i for i in range(res.shape[3])]) - intercept

        corrected_phases = corrected_phases - corrected_phases[0, rx, tx, subcarrier]

        return np.unwrap(corrected_phases)

    ####################################################################################################################
    # Algorithmes MUSIC pour trouver les directions d'arrivée

    def noise_subspaces(self, rx, tx, subcarrier):
        """Récupérer le sous-espace bruit pour chaque paquet"""
        amps = self.process_amp()
        phases = self.process_phase(rx, tx, subcarrier)

        proc_csi = amps * np.exp(1j * phases)
        noise_subspaces = np.zeros(shape=(proc_csi.shape[0], self.params['Nrx'], 2), dtype=complex)

        # Pour chaque paquet, on récupère le sous-espace bruit
        for i in range(proc_csi.shape[0]):
            paquet = proc_csi[i]
            paquet = np.reshape(paquet, (paquet.shape[0] * paquet.shape[1], paquet.shape[2]))

            Rx = np.dot(paquet, np.conj(paquet.T))

            eigvals, eigvecs = np.linalg.eig(Rx)

            # On ordonne les val p et les vec p par ordre décroissant pour récupérer le sous espace bruit constitué des deux vec p issues des deux val p les plus petites
            idx = eigvals.argsort()[::-1]
            eigvecs = eigvecs[:, idx]

            noise_subspace = eigvecs[:, -2:]
            noise_subspaces[i] = noise_subspace

        return noise_subspaces

    def DoA(self, rx, tx, subcarrier):
        """Récupérer les pseudo-spectre issu de l'algorithme MUSIC pour chaque paquet"""
        noise_subspaces = self.noise_subspaces(rx, tx, subcarrier)

        omegas = np.linspace(-np.pi, np.pi, 360)
        e_omegas = np.array([np.exp(1j * i * omegas) for i in range(noise_subspaces.shape[1])]).T

        pseudo_spectrums = np.zeros(shape=(noise_subspaces.shape[0], e_omegas.shape[0]), dtype=complex)
        maximums = np.zeros(shape=(noise_subspaces.shape[0]))

        for i in range(noise_subspaces.shape[0]):
            inv_spectrum = np.array([np.dot(
                np.dot(np.dot(np.conj(e_omegas[j]), noise_subspaces[i]), np.conj(noise_subspaces[i]).T), e_omegas[j])
                                     for j in range(e_omegas.shape[0])])

            pseudo_spectrums[i] = np.reciprocal(np.abs(inv_spectrum))
            maximums[i] = 180 / np.pi * omegas[np.argmax(pseudo_spectrums[i])]

        return (maximums)

    def plot_DoA(self, rx, tx, subcarrier):
        res = np.unwrap(self.DoA(rx, tx, subcarrier))

        plt.figure()
        plt.title(self.path)
        plt.xlabel("Paquets (Vision temporelle)")
        plt.ylabel("DoA (°)")
        plt.plot(res)
        plt.show()

        return 0