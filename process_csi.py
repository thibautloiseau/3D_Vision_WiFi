import numpy as np
import matplotlib.pyplot as plt

class CSI:

    def __init__(self, path):
        self.path = path
        self.ENDIANESS = 'big'
        self.params = {
            'Nrx': 3,
            'Ntx': 3,
            'Nsubcarriers': 114,
        }

    ####################################################################################################################
    def signbit_convert(self, data, maxbit):
        if data & (1 << (maxbit - 1)):
            data -= (1 << maxbit)

        return data

    def ath_processCsiBuffer_online(self, data, params, res):
        blocklen = int.from_bytes(data[:2], byteorder=self.ENDIANESS)
        nsubc = int(data[18])
        nr = int(data[19])
        nc = int(data[20])
        if nr != params['Nrx'] or nc != params['Ntx'] or nsubc != params['Nsubcarriers']:
            return False
        if nr * nc * nsubc * 20 / 8 > blocklen - 27:
            return False

        self.ath_decodeCSIMatrix(nr, nc, nsubc, res, data)

        return True

    def ath_decodeCSIMatrix(self, nr, nc, nsubc, matrix, data):
        bitmask = (1 << 10) - 1
        idx = 27
        current_data = 0
        bits_left = 0  # process 16 bits at a time

        for k in range(nsubc):
            for nc_idx in range(nc):
                for nr_idx in range(nr):

                    if bits_left < 10:
                        h_data = data[idx]
                        idx += 1
                        h_data += (data[idx] << 8)
                        idx += 1
                        current_data += h_data << bits_left
                        bits_left += 16

                    # img
                    img = float(self.signbit_convert(current_data & bitmask, 10))
                    bits_left -= 10
                    current_data = current_data >> 10

                    if bits_left < 10:
                        h_data = data[idx]
                        idx += 1
                        h_data += (data[idx] << 8)
                        idx += 1
                        current_data += h_data << bits_left
                        bits_left += 16

                    # real
                    real = float(self.signbit_convert(current_data & bitmask, 10))
                    bits_left -= 10
                    current_data = current_data >> 10

                    matrix[nr_idx, nc_idx, k] = real + 1j * img

        if nsubc == 114:  # for 40mhz need to apply 90deg rotation
            matrix[:, :, 57:] = matrix[:, :, 57:] * np.exp(-1j * np.pi / 2)

    def ath_parseFile(self, fpath, params, filepercent=100, limit=100000):
        csilist = np.empty((limit, params['Nrx'], params['Ntx'], params['Nsubcarriers']), dtype=complex)
        count = 0
        with open(fpath, 'rb') as fp:
            data = fp.read()
            l = len(data)
            p = 0
            s = (l / 100) * (100 - filepercent)
            while p < l - s and count < limit:
                bl = int.from_bytes(data[p:p + 2], byteorder=self.ENDIANESS)
                self.ath_processCsiBuffer_online(data[p:], params, csilist[count])
                p += bl + 2
                count += 1

        return csilist[:count]

    ####################################################################################################################

    def get_raw_data(self):
        """Récupérer les CSI brutes"""
        res = self.ath_parseFile(self.path, self.params, filepercent=100)

        return res

    def get_raw_amp(self):
        """Récupérer les amplitudes brutes"""
        res = self.get_raw_data()

        return np.abs(res)

    def get_raw_phase(self):
        """Récupérer les phases brutes"""
        res = self.get_raw_data()

        return np.unwrap(np.angle(res))

    def plot_raw_amp(self):
        """Tracer les amplitudes brutes des CSI"""
        res = self.get_raw_amp()

        res = np.reshape(res, (res.shape[0], res.shape[1] * res.shape[2] * res.shape[3]))

        plt.figure()
        plt.plot(res.T)
        plt.title("Raw Amplitude")
        plt.xlabel('Subcarrier')
        plt.ylabel('Amplitude')
        plt.show()

        return 0

    def plot_raw_phase(self):
        """Tracer les phases brutes des CSI"""
        res = self.get_raw_phase()

        res = np.reshape(res, (res.shape[0], res.shape[1] * res.shape[2] * res.shape[3]))

        plt.figure()
        plt.plot(res.T)
        plt.title("Raw Phase")
        plt.xlabel('Subcarrier')
        plt.ylabel('Phase')
        plt.show()

        return 0

    def raw_phase_evolution(self, subcarrier):
        """Permet de voir l'évolution de la phase pour une seule antenne"""
        res = self.get_raw_phase()

        # On trace l'évolution de la phase d'une sous-porteuse pour le premier canal (<=> évolution temporelle)
        phase = np.unwrap(res[:, 0, 0, subcarrier])

        plt.figure()
        plt.title("Évolution de la phase pour la sous-porteuse " + str(subcarrier) + " du premier canal \n" + self.path)
        plt.plot(phase)
        plt.xlabel("Paquets")
        plt.ylabel("Phase d'une sous-porteuse du premier canal")
        plt.show()

        return 0

    def raw_channel_to_channel_phase_evolution(self, subcarrier):
        """Trace l'évolution de la différence de phase pour une sous-porteuse entre 2 canaux"""
        res = self.get_raw_phase()

        diff_phase = np.unwrap(res[:, 0, 0, subcarrier] - res[:, 1, 0, subcarrier])

        plt.figure()
        plt.title("Évolution de la différence de phase entre les deux premiers channels pour la sous-porteuse " + str(subcarrier) + "\n"
                  + self.path + "\n" + str(np.mean(diff_phase)) + "$\pm$" + str(3*np.std(diff_phase)))
        plt.plot(diff_phase)
        plt.xlabel("Paquets")
        plt.ylabel("Différence de phase")
        plt.show()

        return 0

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

    def plot_processed_amp(self):
        """Tracer l'amplitude corrigée"""
        res = self.process_amp()

        res = np.reshape(res, (res.shape[0], res.shape[1] * res.shape[2] * res.shape[3]))

        plt.figure()
        plt.plot(res.T)
        plt.title("Processed amplitude")
        plt.xlabel('Subcarrier')
        plt.ylabel('Amplitude')
        plt.show()

        return 0

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

    def plot_processed_phase(self, rx, tx, subcarrier):
        """Tracer la phase corrigée"""
        res = self.process_phase(rx, tx, subcarrier)

        res = np.reshape(res, (res.shape[0], res.shape[1] * res.shape[2] * res.shape[3]))

        plt.figure()
        plt.plot(res.T)
        plt.title("Processed phase")
        plt.xlabel('Subcarrier')
        plt.ylabel('Phase')
        plt.show()

        return 0

    def plot_processed_phase_evolution(self, rx, tx, subcarrier):
        """Tracer l'évolution temporelle de la phase corrigée"""
        res = self.process_phase(rx, tx, subcarrier)

        corrected_phase = res[:, rx, tx, subcarrier]

        plt.figure()
        plt.title("Évolution de la phase corrigée \n" + self.path)
        plt.plot(corrected_phase)
        plt.xlabel("Paquet")
        plt.ylabel("Évolution de la phase corrigée")
        plt.show()

        return 0

    ####################################################################################################################
    # Autres techniques de filtrage

    def mean_processed_phase(self, i, j):
        """Filtrage par moyennage"""
        res = self.process_phase()[:, i, j, :]
        mean_phase = np.array([np.mean(res[:, k]) for k in range(res.shape[1])])

        return (mean_phase)

    def plot_mean_phase(self, i, j):
        """Tracé de la phase traitée par la moyenne"""
        res = self.mean_processed_phase(i, j)

        plt.figure()
        plt.plot(res)
        plt.title('Mean phase')
        plt.xlabel('Subcarrier')
        plt.ylabel('Phase')
        plt.show()

        return 0

    def svd_processed_phase(self, i, j):
        """Filtrage par décomposition en valeurs singulières"""
        res = self.process_phase()[:, i, j, :].T

        U, _, _ = np.linalg.svd(res)

        return (U[:, 0])

    def plot_svd_phase(self, i, j):
        """Tracé de la phase traitée par SVD"""
        res = self.svd_processed_phase(i, j)

        plt.figure()
        plt.plot(res)
        plt.title('SVD Phase')
        plt.xlabel('Subcarrier')
        plt.ylabel('Phase')
        plt.show()

        return 0

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

    def pseudo_spectrum(self, rx, tx, subcarrier):
        """Récupérer les pseudo-spectre issu de l'algorithme MUSIC pour chaque paquet"""
        noise_subspaces = self.noise_subspaces(rx, tx, subcarrier)

        omegas = np.linspace(-np.pi, np.pi, 360)
        e_omegas = np.array([np.exp(1j * i * omegas) for i in range(noise_subspaces.shape[1])]).T

        pseudo_spectrums = np.zeros(shape=(noise_subspaces.shape[0], e_omegas.shape[0]), dtype=complex)
        maximums = np.zeros(shape=(noise_subspaces.shape[0]))

        for i in range(noise_subspaces.shape[0]):
            inv_spectrum = np.array([np.dot(np.dot(np.dot(np.conj(e_omegas[j]), noise_subspaces[i]), np.conj(noise_subspaces[i]).T), e_omegas[j]) for j in range(e_omegas.shape[0])])

            pseudo_spectrums[i] = np.reciprocal(np.abs(inv_spectrum))
            maximums[i] = 180 / np.pi * omegas[np.argmax(pseudo_spectrums[i])]

        pseudo_spectrum = np.abs(np.mean(pseudo_spectrums, axis=0))

        maximum = np.mean(maximums)
        std_error_maximum = 3*np.std(maximums) # Assumption of a Gaussian distribution

        return(omegas, pseudo_spectrum, maximum, std_error_maximum)

    def plot_pseudo_spectrum(self, rx, tx, subcarrier):
        """Tracé du pseudo-spectre de l'algorithme MUSIC"""
        pseudo_spectrum = self.pseudo_spectrum(rx, tx, subcarrier)

        plt.figure()
        plt.title(self.path + "\nMaximum : " + str(pseudo_spectrum[2]) + "$\pm$" + str(pseudo_spectrum[3]))
        plt.plot(180 / np.pi * pseudo_spectrum[0], pseudo_spectrum[1], '+')
        plt.show()
 
        return 0

    ####################################################################################################################
    def raw_calculus(self):
        """Calcul naïf des DoA"""
        res = self.process_phase(0, 0, 0)

        bande = 40e6
        freq = np.array([5805e6-bande/2 + bande/self.params["Nsubcarriers"]*i for i in range(self.params["Nsubcarriers"])])
        long_onde = 3e8/freq

        phase_diff = res[:, 0, 0, :] - res[:, 1, 0, :]

        # for k in range(phase_diff.shape[0]):
        thetas = 180/np.pi*np.arcsin(np.unwrap(long_onde*phase_diff/0.027))
        mean_theta = np.mean(thetas)
        std_err = np.std(thetas)

        return(mean_theta, std_err)

    ####################################################################################################################
    # Création de fichiers plus courts pour les acquisitions continues
    def shorten_continuous_file(self, dest):
        """Raccourcir les fichiers d'acquisitions qui font 30000 paquets de base, on moyenne sur 10 paquets"""
        res = self.get_raw_data()
        shorten_res = np.zeros(shape=(int(res.shape[0]/10), res.shape[1], res.shape[2], res.shape[3]), dtype=complex)

        for i in range(res.shape[0]):
            if i%10 == 0:
                shorten_res[int(i/10)] = np.mean(res[i: i+10], axis=0)

        np.save(dest, shorten_res)

        return shorten_res

    ####################################################################################################################
    # MMP technique

    def Ce_matrix(self, paquet):
        """Création de la matrice Ce à partir d'un seul paquet"""
        res = self.get_raw_data()[paquet]

        m = self.params["Nsubcarriers"] // 2  # Nombre de lignes des sous-matrices
        r = self.params["Nsubcarriers"] // 2 + 1  # Nombre de colonnes des sous-matrices

        #On crée Ntx différentes matrices Ce, une pour chaque antenne à l'émission
        Ce = np.zeros(shape=(self.params["Ntx"], 2*m, 2*r), dtype=complex)

        for tx in range(res.shape[1]):
            temp_res = res[:, tx, :]

            c1 = temp_res[0]
            c2 = temp_res[1]
            c3 = temp_res[2]

            C1 = np.zeros(shape=(m, r), dtype=complex)
            C2 = np.zeros(shape=(m, r), dtype=complex)
            C3 = np.zeros(shape=(m, r), dtype=complex)

            for i in range(m):
                C1[i] = c1[i: i + r]
                C2[i] = c2[i: i + r]
                C3[i] = c3[i: i + r]

            temp_Ce = np.concatenate((np.concatenate((C1, C2), axis=1), np.concatenate((C2, C3), axis=1)))
            Ce[tx] = temp_Ce

        return Ce

    def DoA_MMP(self, paquet, d_antenne):
        """Méthode MMP pour retrouver les directions d'arrivée"""
        Ce = self.Ce_matrix(paquet)
        m = self.params["Nsubcarriers"]//2
        long_onde = 2.99792e8/5805e6 # c/f

        thetas = np.zeros(shape=(Ce.shape[0], 2))

        for tx in range(Ce.shape[0]):
            U, _, _ = np.linalg.svd(Ce[tx]) # Les valeurs et vecteurs singuliers sont classés par ordre décroissant

            # On décompose U en deux matrices, une Us qui comprend les deux vecteurs correspondant aux deux valeurs singulières les plus grandes, l'autre Uv qui comprend le reste
            # On ne considère que Us pour les calculs
            Us = U[:, :2]

            # On redécompose Us en Us1 et Us2 avec rang(Us1) = 2, le nombre de DoA que l'on veut distinguer
            Us1 = Us[m:, :]
            Us2 = Us[:m, :]

            # On écrit notre matrice qui correspondra au sous-espace signal et on calcule les valeurs propres
            M = np.linalg.pinv(Us1) @ Us2
            eigvals, _ = np.linalg.eig(M)

            temp_thetas = 180/np.pi * np.unwrap((np.arcsin(-long_onde*np.angle(eigvals) / (2*np.pi*d_antenne))))
            thetas[tx] = temp_thetas

        return thetas

    def aggregation_DoA_MMP(self, d_antenne):
        """Aggregation de paquets pour un meilleur SNR et une mesure plus sûre"""
        res = self.get_raw_data()
        m = self.params["Nsubcarriers"]*3 // 2
        long_onde = 2.99792e8 / 5805e6  # c/f

        thetas = np.zeros(shape=(self.params["Ntx"], 2))

        for tx in range(self.params["Ntx"]):
            aggregate_csi = np.zeros(shape=(self.params["Nsubcarriers"] * 3, res.shape[0]), dtype=complex)

            for i in range(res.shape[0]):
                aggregate_csi[:, i] = np.reshape(res[i, :, tx, :], (res[i].shape[2]*3))

            U, _, _ = np.linalg.svd(aggregate_csi)

            Us = U[:, :2]

            Us1 = Us[m:, :]
            Us2 = Us[:m, :]

            # On écrit notre matrice qui correspondra au sous-espace signal et on calcule les valeurs propres
            M = np.linalg.pinv(Us1) @ Us2
            eigvals, _ = np.linalg.eig(M)

            thetas[tx] = 180 / np.pi * np.unwrap((np.arcsin(-long_onde * np.angle(eigvals) / (2 * np.pi * d_antenne))))

        return thetas

    # Tentative de calcul des ToF avec MMP
    def permutation_ToF(self):
        m = self.params["Nsubcarriers"]//2
        P = np.zeros(shape=(2 * m, 2 * m))

        for i in range(P.shape[0]):
            if i % 2 == 0:
                P[i, int(i/2) + m] = 1
            else:
                P[i, int((i + 1) / 2)] = 1

        return(P)

    def ToF_MMP(self, paquet):
        """Méthode de calcul du ToF pour un paquet précis"""
        Ce = self.Ce_matrix(paquet)
        delta = 40e6 / self.params["Nsubcarriers"] #Écart de fréquence entre 2 sous-porteuses

        U, _, _ = np.linalg.svd(Ce)
        Us = U[:, :2]

        P = self.permutation_ToF()

        Us_hat = P @ Us
        Us_hat = Us_hat[2: -2]

        Us_hat_1 = Us_hat[:Us_hat.shape[0]//2]
        Us_hat_2 = Us_hat[Us_hat.shape[0]//2:]

        M = np.linalg.pinv(Us_hat_1) @ Us_hat_2
        eigvals, _ = np.linalg.eig(M)

        tofs = np.angle(eigvals) / (-2*np.pi*delta)

        return tofs