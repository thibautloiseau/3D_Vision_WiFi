import os
import process_csi as process
import continuous_csi as continuous
import json
import matplotlib.pyplot as plt
import numpy as np
import reflecteur

def main():
    # Traitement stats MMP réflecteur + sans LOS

    # Paquet par paquet
    # with open('mmp_ref_los.json', 'r') as f:
    #     data = json.load(f)
    #
    # dico_mmp_ref_los = {}
    #
    # for key in data.keys():
    #     if data[key]['Theta'] not in dico_mmp_ref_los.keys():
    #         dico_mmp_ref_los[data[key]['Theta']] = {}
    #
    #     dico_mmp_ref_los[data[key]['Theta']][key[-1]] = {}
    #
    #     dico_mmp_ref_los[data[key]['Theta']][key[-1]]['mean_1'] = np.mean(data[key]['1'])
    #     dico_mmp_ref_los[data[key]['Theta']][key[-1]]['std_1'] = np.std(data[key]['1'])
    #
    #     dico_mmp_ref_los[data[key]['Theta']][key[-1]]['mean_2'] = np.mean(data[key]['2'])
    #     dico_mmp_ref_los[data[key]['Theta']][key[-1]]['std_2'] = np.std(data[key]['2'])
    #
    # with open('stats_mmp_ref_los.json', 'w+') as f:
    #     json.dump(dico_mmp_ref_los, f, indent=2)

    # Agregation
    # dico_mmp_ref_no_los_agreg = {}
    #
    # for csi in os.listdir('acquisitions/01-04-2021_petite_chambre'):
    #     if 'no' in csi:
    #         CSI = process.CSI('acquisitions/01-04-2021_petite_chambre/' + csi)
    #         dico_mmp_ref_no_los_agreg[csi] = CSI.aggregation_DoA_MMP(2.7e-2)[0, 0]
    #
    # with open('mmp_ref_no_los_agr.json', 'w+') as f:
    #     json.dump(dico_mmp_ref_no_los_agreg, f, indent=2)
    ####################################################################################################################
    # # Mesures avec réflecteur et avec LoS MMP
    #
    # dico_mmp_ref = {}
    #
    # # Paquet par paquet
    # for csi in os.listdir("acquisitions/01-04-2021_petite_chambre"):
    #     if 'no' not in csi and 'H' in csi:
    #         dico_mmp_ref[csi] = {}
    #
    #         print(csi)
    #
    #         L1 = int(csi.split('L1_')[1].split('_L')[0])
    #         L = int(csi.split('L_')[1].split('_')[0])
    #         H = int(csi.split('H_')[1].split('_L1')[0])
    #
    #         dico_mmp_ref[csi]['Theta'] = 180/np.pi * np.arctan(H/(L-L1))
    #
    #         CSI = process.CSI("acquisitions/01-04-2021_petite_chambre/" + csi)
    #         no_paquets = CSI.get_raw_data().shape[0]
    #
    #         thetas_1 = np.zeros(shape=no_paquets)
    #         thetas_2 = np.zeros(shape=no_paquets)
    #
    #         for paquet in range(no_paquets):
    #             print('\t' + str(paquet))
    #             thetas = CSI.DoA_MMP(paquet, 2.7e-2)[0]
    #             thetas_1[paquet] = thetas[0]
    #             thetas_2[paquet] = thetas[1]
    #
    #         dico_mmp_ref[csi]['1'] = thetas_1.tolist()
    #         dico_mmp_ref[csi]['2'] = thetas_2.tolist()
    #
    #         with open('mmp_ref_los.json', 'w+') as f:
    #             json.dump(dico_mmp_ref, f, indent=2)

    ####################################################################################################################
    # Traitement stats MMP réflecteur + sans LOS

    # Paquet par paquet
    # with open('mmp_ref_no_los.json', 'r') as f:
    #     data = json.load(f)
    #
    # dico_mmp_ref_no_los = {}
    #
    # for key in data.keys():
    #     if data[key]['Theta'] not in dico_mmp_ref_no_los.keys():
    #         dico_mmp_ref_no_los[data[key]['Theta']] = {}
    #
    #     dico_mmp_ref_no_los[data[key]['Theta']][key[-1]] = {}
    #
    #     dico_mmp_ref_no_los[data[key]['Theta']][key[-1]]['mean'] = np.mean(data[key]['1'])
    #     dico_mmp_ref_no_los[data[key]['Theta']][key[-1]]['std'] = np.std(data[key]['1'])
    #
    # with open('stats_mmp_ref_no_los.json', 'w+') as f:
    #     json.dump(dico_mmp_ref_no_los, f, indent=2)

    # Agregation
    # dico_mmp_ref_no_los_agreg = {}
    #
    # for csi in os.listdir('acquisitions/01-04-2021_petite_chambre'):
    #     if 'no' in csi:
    #         CSI = process.CSI('acquisitions/01-04-2021_petite_chambre/' + csi)
    #         dico_mmp_ref_no_los_agreg[csi] = CSI.aggregation_DoA_MMP(2.7e-2)[0, 0]
    #
    # with open('mmp_ref_no_los_agr.json', 'w+') as f:
    #     json.dump(dico_mmp_ref_no_los_agreg, f, indent=2)

    ####################################################################################################################
    # Mesures avec réflecteur et sans LoS MMP

    # dico_mmp_ref = {}
    #
    # # Paquet par paquet
    # for csi in os.listdir("acquisitions/01-04-2021_petite_chambre"):
    #     if 'no' in csi:
    #         dico_mmp_ref[csi] = {}
    #
    #         print(csi)
    #
    #         L1 = int(csi.split('L1_')[1].split('_L')[0])
    #         L = int(csi.split('L_')[1].split('_')[0])
    #         H = int(csi.split('H_')[1].split('_L1')[0])
    #
    #         dico_mmp_ref[csi]['Theta'] = 180/np.pi * np.arctan(H/(L-L1))
    #
    #         CSI = process.CSI("acquisitions/01-04-2021_petite_chambre/" + csi)
    #         no_paquets = CSI.get_raw_data().shape[0]
    #
    #         thetas_1 = np.zeros(shape=no_paquets)
    #         thetas_2 = np.zeros(shape=no_paquets)
    #
    #         for paquet in range(no_paquets):
    #             print('\t' + str(paquet))
    #             thetas = CSI.DoA_MMP(paquet, 2.7e-2)[0]
    #             thetas_1[paquet] = thetas[0]
    #             thetas_2[paquet] = thetas[1]
    #
    #         dico_mmp_ref[csi]['1'] = thetas_1.tolist()
    #         dico_mmp_ref[csi]['2'] = thetas_2.tolist()
    #
    #         with open('mmp_ref_no_los.json', 'w+') as f:
    #             json.dump(dico_mmp_ref, f, indent=2)

    ####################################################################################################################
    # # Mesures continues MUSIC
    # CSI = continuous.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/shorten_continuous.npy")
    # thetas = CSI.pseudo_spectrum(2.7e-2)
    #
    # plt.plot([i for i in range(len(thetas))], thetas, '+')
    # plt.xlabel("Paquet (Vision temporelle)")
    # plt.ylabel("DoA estimée par MUSIC (°)")
    # plt.show()

    ####################################################################################################################
    # Mesures continues MMP
    # CSI = continuous.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/shorten_continuous.npy")
    #
    # no_acq = CSI.get_raw_data().shape[0]
    # DoAs = np.zeros(shape=(no_acq, 2))
    # print(DoAs.shape)
    #
    # for paquet in range(no_acq):
    #     print(paquet)
    #     DoAs[paquet] = [paquet, CSI.DoA_MMP(paquet, 2.7e-2)[0]]
    #
    # print(np.max(DoAs[:, 1]))
    #
    # plt.plot(DoAs[:, 0], DoAs[:, 1], '+')
    # plt.xlabel("Paquet (Vision temporelle)")
    # plt.ylabel("DoA estimée par MMP (°)")
    # plt.show()

    ####################################################################################################################
    # Traitement stats discrètes MMP

    # with open('mmp_discrete.json', 'r') as f:
    #     data = json.load(f)
    #
    # stats_mmp = np.zeros(shape=(len(data.keys()), 3))
    #
    # for idx, true_angle in enumerate(data.keys()):
    #     stats_mmp[idx] = [int(true_angle), np.mean(data[true_angle]['1']), np.std(data[true_angle]['1'])]
    #
    # plt.errorbar(stats_mmp[:, 0], stats_mmp[:, 1], yerr=stats_mmp[:, 2], ecolor='r', fmt='bo', label='MMP')
    # plt.plot(stats_mmp[:, 0], stats_mmp[:, 0], 'g', label='Valeurs attendues')
    #
    # plt.legend(loc='upper right')
    # plt.xlabel('AoA attendue (°)')
    # plt.ylabel('AoA estimée avec MMP (°)')
    #
    # print(stats_mmp)
    #
    # plt.show()

    # Agregation
    # no_files = len([csi for csi in os.listdir("acquisitions/03-03-2021_grosse_chambre_Thibaut") if 'continuous' not in csi])
    # stats_mmp_agr = np.zeros(shape=(no_files, 2))
    #
    # for idx, csi in enumerate(os.listdir("acquisitions/03-03-2021_grosse_chambre_Thibaut")):
    #     if 'continuous' not in csi:
    #         print(csi)
    #         CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/" + csi)
    #         stats_mmp_agr[idx] = [int(csi), CSI.aggregation_DoA_MMP(2.7e-2)[0, 0]]
    #
    # plt.plot(stats_mmp_agr[:, 0], stats_mmp_agr[:, 1], 'b+', label='MMP')
    # plt.plot([-80, 80], [-80, 80], 'r-', label='Valeurs attendues')
    # plt.legend(loc='upper right')
    # plt.xlabel('AoA attendue (°)')
    # plt.ylabel('AoA estimée avec MMP (°)')
    # plt.show()

    ####################################################################################################################
    # Mesures MMP, acquisitions discrètes
    dico_mmp = {}

    for idx, doc in enumerate(os.listdir("acquisitions/03-03-2021_grosse_chambre_Thibaut")):
        if "continuous" not in doc:
            print(doc)
            CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/" + doc)
            no_acq = CSI.get_raw_data().shape[0]

            dico_mmp[doc] = {}

            thetas_1 = np.zeros(shape=no_acq)
            thetas_2 = np.zeros(shape=no_acq)

            for paquet in range(no_acq):
                print('\t' + str(paquet))
                thetas = CSI.DoA_MMP(paquet, 2.7e-2)[0]
                thetas_1[paquet] = thetas[0]
                thetas_2[paquet] = thetas[1]

            dico_mmp[doc]['1'] = thetas_1.tolist()
            dico_mmp[doc]['2'] = thetas_2.tolist()

            with open('mmp_discrete_sanitized.json', 'w+') as fp:
                json.dump(dico_mmp, fp, indent=2)

    ####################################################################################################################
    # Traitement stats discrètes MUSIC

    # with open('music_discrete.json', 'r') as f:
    #     data = json.load(f)
    #
    # stats_music = np.zeros(shape=(len(data.keys()), 3))
    #
    # for idx, true_angle in enumerate(data.keys()):
    #     stats_music[idx] = [int(true_angle), np.mean(data[true_angle]), np.std(data[true_angle])]
    #
    # plt.errorbar(stats_music[:, 0 ], stats_music[:, 1], yerr=stats_music[:, 2], ecolor='r', fmt='bo', label='MUSIC')
    # plt.plot([-80, 80], [-80, 80], 'g', label='Valeurs attendues')
    #
    # plt.legend()
    # plt.xlabel('AoA attendue (°)')
    # plt.ylabel('AoA estimée avec MUSIC (°)')
    #
    # plt.show()

    ####################################################################################################################
    # Mesures MUSIC, acquisitions discrètes
    # dico_music = {}
    #
    # for idx, doc in enumerate(os.listdir("acquisitions/03-03-2021_grosse_chambre_Thibaut")):
    #     if "continuous" not in doc:
    #         print(doc)
    #         CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/" + doc)
    #         dico_music[doc] = CSI.pseudo_spectrum(2.7e-2).tolist()
    #
    #         with open('music_discrete.json', 'w') as fp:
    #             json.dump(dico_music, fp, indent=2)

    ####################################################################################################################

    # Traitement des amplitudes
    # CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/0")
    # CSI.params["Ntx"] = 1
    #
    # CSI.plot_raw_amp()
    # CSI.plot_processed_amp()

    # Traitement des phases
    # CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/0")
    # CSI.params["Ntx"] = 1
    #
    # CSI.plot_raw_phase()
    # CSI.plot_processed_phase(0, 0, 0)


    ####################################################################################################################
    # Acquisitions en chambre avec réflecteur 1TX / 3RX
    # CSI = process.CSI("acquisitions/01-04-2021_petite_chambre/LOS_H_190_L1_60_L_320_1")
    # print(CSI.path)
    # CSI.params["Ntx"] = 1
    # data = CSI.get_raw_data()
    # print(data.shape)
    #
    # doa = np.zeros(shape=(data.shape[0], 2))
    #
    # for paquet in range(data.shape[0]):
    #     print(paquet)
    #     # print(CSI.DoA_MMP(paquet, 2.7e-2))
    #     doa[paquet] = CSI.DoA_MMP(paquet, 2.7e-2)[0]
    #
    # plt.figure(1)
    # plt.plot(doa[:, 0], '+')
    # plt.show()
    #
    # plt.figure(2)
    # plt.plot(doa[:, 1], '+')
    # plt.show()
    #
    # print(CSI.aggregation_DoA_MMP(2.7e-2))

    ####################################################################################################################
    # MMP avec réflecteur
    # CSI = reflecteur.reflecteur("acquisitions/petite_chambre/L1_155_H_75_25send", "acquisitions/petite_chambre/L1_155_H_75_27send")
    # data = CSI.CSI1.get_raw_data()
    #
    # for paquet in range(data.shape[0]):
    #     print(CSI.CSI1.DoA_MMP(paquet, 2.7e2))
    #
    # CSI.plot()

    ####################################################################################################################
    # Tentative ToF avec MMP
    # CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/-10")
    # CSI.params["Ntx"] = 1
    # data = CSI.get_raw_data()
    #
    # tofs = np.zeros(shape=(data.shape[0], 2))
    # doa = np.zeros(shape=(data.shape[0], 2))
    #
    # for paquet in range(data.shape[0]):
    #     print(paquet)
    #     tofs[paquet] = CSI.ToF_MMP(paquet)
    #     doa[paquet] = CSI.DoA_MMP(paquet, 2.7e-2)
        # print(tofs[paquet]*2.997e8)

    ####################################################################################################################
    # Raccourcicement des fichiers d'acquisitions continues
    # CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Philibert/continuous")
    # CSI.shorten_continuous_file("acquisitions/03-03-2021_grosse_chambre_Philibert/shorten_continuous")
    #
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
