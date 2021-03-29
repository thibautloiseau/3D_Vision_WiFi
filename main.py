import os
import process_csi as process
import continuous_csi as continuous
import json
import matplotlib.pyplot as plt
import numpy as np

def main():


    ####################################################################################################################
    # Tentative ToF avec MMP
    # CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/-10")
    # data = CSI.get_raw_data()
    #
    # tofs = np.zeros(shape=(data.shape[0], 2))
    #
    # for paquet in range(data.shape[0]):
    #     tofs[paquet] = CSI.ToF_MMP(paquet)
    #
    # print(tofs)


    ####################################################################################################################
    # # Technique MMP pour les acquisitions continues
    # CSI = continuous.CSI("acquisitions/03-03-2021_grosse_chambre_Philibert/shorten_continuous.npy")
    # data = CSI.get_raw_data()
    #
    # DoAs = np.zeros(shape=(data.shape[0], 2))
    #
    # for paquet in range(data.shape[0]):
    #     print(paquet/data.shape[0]*100)
    #     DoAs[paquet] = np.array([paquet, CSI.DoA_MMP(paquet, 7.8e-2)[0]])
    #
    # plt.figure()
    # plt.title("acquisitions/03-03-2021_grosse_chambre_Philibert/shorten_continuous.npy")
    # plt.xlabel("Paquet (Vision temporelle)")
    # plt.ylabel("DoA (°)")
    # plt.plot(DoAs[:, 0], DoAs[:, 1], '+')
    # plt.show()

    ####################################################################################################################
    # # Technique MMP pour les acquisitions discrètes
    # no_docs = len([doc for doc in os.listdir("acquisitions/03-03-2021_grosse_chambre_Thibaut") if "continuous" not in doc])
    # data = np.zeros(shape=(no_docs, 2))
    #
    # for idx, doc in enumerate(os.listdir("acquisitions/03-03-2021_grosse_chambre_Thibaut")):
    #     if "continuous" not in doc:
    #         print(doc)
    #         CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/" + doc)
    #         data[idx] = np.array([int(doc), CSI.aggregation_MMP(2.7e-2)[0]])
    #
    # x = np.array([-90, 90])
    #
    # plt.figure()
    # plt.title("acquisitions/03-03-2021_grosse_chambre_Thibaut")
    # plt.xlabel("Expected DoA (°)")
    # plt.ylabel("Calculated DoA (°)")
    # plt.plot(x, x, 'r')
    # plt.plot(data[:, 0], data[:, 1], '+')
    # plt.show()

    ####################################################################################################################
    # Traitement des fichiers d'acquisition continues
    # CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Philibert/continuous")
    # CSI.shorten_continuous_file("acquisitions/03-03-2021_grosse_chambre_Philibert/shorten_continuous")

    # CSI = continuous.CSI("acquisitions/03-03-2021_grosse_chambre_Philibert/shorten_continuous.npy")
    # CSI.plot_DoA(0, 0, 0)

    ####################################################################################################################
    # Stats avec le calcul naïf
    # thetas = {}
    # for doc in os.listdir("acquisitions/03-03-2021_grosse_chambre_Thibaut"):
    #     if doc != "continuous":
    #         thetas[doc] = {}
    #         print(doc)
    #         CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/" + doc)
    #         info = CSI.raw_calculus()
    #         thetas[doc]["mean_theta"] = info[0]
    #         thetas[doc]["std_err"] = info[1]
    #
    # print(thetas)

    # with open("thetas.json", "w") as file:
    #      json.dump(thetas, file, indent=2)

    ####################################################################################################################
    # Stats sur les DoA

    # stats = {}
    # subcarriers = [i for i in range(114) if i%10 == 0]
    #
    # for doc in os.listdir("Philibert"):
    #     if doc != "continuous":
    #         stats[str(doc)] = {}
    #
    #         print("Philibert/" + doc)
    #         CSI = process.CSI("Philibert/" + doc)
    #
    #         for rx in range(CSI.params["Nrx"]):
    #             print("Rx: " + str(rx))
    #             stats[str(doc)][str(rx)] = {}
    #
    #             for tx in range(CSI.params["Ntx"]):
    #                 print("\tTx: " + str(tx))
    #                 stats[str(doc)][str(rx)][str(tx)] = {}
    #
    #                 for subcarrier in subcarriers:
    #                     print("\t\tSubcarrier: " + str(subcarrier))
    #                     CSI.plot_pseudo_spectrum(rx, tx, subcarrier)
    #
    #                     info = CSI.pseudo_spectrum(rx, tx, subcarrier)
    #                     stats[str(doc)][str(rx)][str(tx)][str(subcarrier)] = {"max": info[-2], "std_err": info[-1]}
    #
    #     with open("stats.json", "w") as file:
    #         json.dump(stats, file, indent=2)

    ####################################################################################################################
    # Tracés des mesures

    # measures = []
    # expMeasures = []
    #
    # for doc in os.listdir("acquisitions/1tx_3rx"):
    #     print(doc)
    #     expMeasures.append(float(doc))
    #     CSI = process.CSI("acquisitions/1tx_3rx/" + doc)
    #     measures.append(CSI.pseudo_spectrum()[-1])
    #
    # plt.figure()
    # plt.title("Measures for calibration")
    # plt.xlim(-90, 90)
    # plt.ylim(-90, 90)
    # plt.xlabel("Effective measures with MUSIC algorithm (in degrees)")
    # plt.ylabel("Expected measures (in degrees)")
    # plt.plot(measures, expMeasures, '+')
    # plt.show()
    #
    # for doc in os.listdir("Philibert/"):
    #     print(doc)
    #     expMeasures.append(float(doc))
    #     CSI = process.CSI("Philibert/" + doc)
    #     measures.append(CSI.pseudo_spectrum()[-1])
    #
    # plt.figure()
    # plt.title("Measures for calibration")
    # plt.xlim(-90, 90)
    # plt.ylim(0, 90)
    # plt.xlabel("Effective measures with MUSIC algorithm (in degrees)")
    # plt.ylabel("Expected measures (in degrees)")
    # plt.plot(measures, expMeasures, '+')
    # plt.show()


if __name__ == "__main__":
    main()
