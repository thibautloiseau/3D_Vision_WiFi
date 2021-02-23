import os
import process_csi as process
import matplotlib.pyplot as plt

def main():
    for doc in os.listdir("setup/1tx_3rx"):
        print("setup/1tx_3rx/" + doc)
        CSI = process.CSI("setup/1tx_3rx/" + doc)
        CSI.plot_processed_phase_evolution()

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
