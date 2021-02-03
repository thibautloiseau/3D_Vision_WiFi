import os
import process_csi as process
import numpy as np
import matplotlib.pyplot as plt

def main():
    measures = []
    expMeasures = []

    for doc in os.listdir("setup/1tx_3rx"):
        print(doc)
        expMeasures.append(float(doc))
        CSI = process.CSI("setup/1tx_3rx/" + doc)
        measures.append(CSI.pseudo_spectrum()[-1])

    plt.figure()
    plt.title("Measures for calibration")
    plt.xlim(-90, 90)
    plt.ylim(-90, 90)
    plt.xlabel("Expected measures (in degrees)")
    plt.ylabel("Effective measures with MUSIC algorithm (in degrees")
    plt.plot(expMeasures, measures, '+')
    plt.show()


if __name__ == "__main__":
    main()
