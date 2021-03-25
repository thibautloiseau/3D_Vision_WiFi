import os
import process_csi as process
import continuous_csi as continuous
import json
import matplotlib.pyplot as plt

def main():
    # MMP technique
    CSI = process.CSI("setup/03-03-2021_grosse_chambre_Thibaut/-80")

    for paquet in range(1000):
        print(CSI.MMP(paquet))

    ####################################################################################################################
    # Traitement des fichiers d'acquisition continues
    # CSI = process.CSI("setup/03-03-2021_grosse_chambre_Philibert/continuous")
    # CSI.shorten_continuous_file("setup/03-03-2021_grosse_chambre_Philibert/shorten_continuous")

    # CSI = continuous.CSI("setup/03-03-2021_grosse_chambre_Philibert/shorten_continuous.npy")
    # CSI.plot_DoA(0, 0, 0)

    ####################################################################################################################
    # Stats avec le calcul naïf
    # thetas = {}
    # for doc in os.listdir("setup/03-03-2021_grosse_chambre_Thibaut"):
    #     if doc != "continuous":
    #         thetas[doc] = {}
    #         print(doc)
    #         CSI = process.CSI("setup/03-03-2021_grosse_chambre_Thibaut/" + doc)
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
    # for doc in os.listdir("setup/1tx_3rx"):
    #     print(doc)
    #     expMeasures.append(float(doc))
    #     CSI = process.CSI("setup/1tx_3rx/" + doc)
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
