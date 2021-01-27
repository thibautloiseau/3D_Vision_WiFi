import os
import process_csi as process


def main():
    for doc in os.listdir("setup/1tx_3rx"):
        CSI = process.CSI("setup/1tx_3rx/" + doc)
        CSI.plot_pseudo_spectrum()

if __name__ == "__main__":
    main()
