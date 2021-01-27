import os
import process_csi as process


def main():
    for doc in os.listdir("experiences/"):
        if 'L1' in doc:
            CSI = process.CSI("experiences/" + doc)
            print(CSI.path)
            CSI.plot_pseudo_spectrum()


if __name__ == "__main__":
    main()
