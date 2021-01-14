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

    ####################################################################################################################
    # Amplitude processing
    def process_amp(self):
        """Traiter l'amplitude des CSI"""
        res = self.get_raw_amp()

        amp_means = np.zeros(shape=(3, 3, 114))
        amp_std = np.zeros(shape=(3, 3, 114))
        amp_filters = np.zeros(shape=(res.shape[0], 3, 3, 114))

        for i in range(res.shape[1]):
            for j in range(res.shape[2]):
                amp_means[i, j] = [np.mean(res[:, i, j, k]) for k in range(res.shape[3])]
                amp_std[i, j] = [np.std(res[:, i, j, k]) for k in range(res.shape[3])]

                for k in range(1, res.shape[0] - 1):
                    amp_filter = 1/3 * (res[k-1, i, j, :] + res[k, i, j, :] + res[k+1, i, j, :])
                    if np.cov(amp_filter, amp_means[i, j])[0, 1] > 0:
                        amp_filters[k, i, j, :] = amp_filter

        return amp_filters[1: -1]


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

        return corrected_phases


    def plot_processed_phase(self):
        """Tracer la phase corrigée"""
        res = self.process_phase()

        res = np.reshape(res, (res.shape[0], res.shape[1] * res.shape[2] * res.shape[3]))

        plt.figure()
        plt.plot(res.T)
        plt.title("Processed phase")
        plt.xlabel('Subcarrier')
        plt.ylabel('Phase')
        plt.show()

        return 0

    ####################################################################################################################
    # Autres techniques de filtrage

    def mean_processed_phase(self, i, j):
        """Filtrage par moyennage"""
        res = self.process_phase()[:, i, j, :]
        mean_phase = np.array([np.mean(res[:, k]) for k in range(res.shape[1])])

        return(mean_phase)


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

        return(U[:, 0])


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

    def noise_subspace(self):
        """Récupérer le sous-espace bruit"""
        amps = self.process_amp()
        phases = self.process_phase()

        mean_amps = amps.mean(axis=0)
        mean_phases = phases.mean(axis=0)

        mean_csi = mean_amps * np.exp(1j * mean_phases)
        mean_csi = np.reshape(mean_csi, (mean_csi.shape[0] * mean_csi.shape[1], mean_csi.shape[2]))

        Rx = np.dot(mean_csi, np.conj(mean_csi.T))

        eigvals, eigvecs = np.linalg.eig(Rx)

        # On ordonne les valeurs propres et les vecteurs propres par ordre décroissant
        idx = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:, idx]

        noise_subspace = eigvecs[:, 3:]

        return noise_subspace

    def pseudo_spectrum(self):
        """Récupérer le pseudo-spectre issu de l'algorithme MUSIC"""
        noise_subspace = self.noise_subspace()

        omegas = np.linspace(-np.pi, np.pi, 360)
        e_omegas = np.array([np.exp(1j * i * omegas) for i in range(9)]).T

        inv_spectrum = np.array([np.dot(np.dot(np.dot(np.conj(e_omegas[i]), noise_subspace), np.conj(noise_subspace).T), e_omegas[i]) for i in range(e_omegas.shape[0])])

        return (omegas, np.abs(1 / inv_spectrum))

    def plot_pseudo_spectrum(self):
        """Tracé du pseudo-spectre de l'algorithme MUSIC"""
        pseudo_spectrum = self.pseudo_spectrum()

        plt.figure()
        plt.title(self.path)
        plt.plot(180/np.pi*pseudo_spectrum[0], pseudo_spectrum[1])
        plt.show()

        return 0