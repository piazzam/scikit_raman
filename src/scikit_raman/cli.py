import argparse
import scikit_raman.Classes.Experiment.DLExperiment as DLE
import scikit_raman.Classes.Experiment.MLExperiment as MLE

def main():
    parser = argparse.ArgumentParser(
        description='The Command Line Interface (CLI) to the scikit_raman library, used to analyze Raman spectral data.',
        epilog='Sample usage: scikit-raman-cli -p configuration.yaml -t dl'
    )
    parser.add_argument('--configuration-file', '-p', type=str, required=True,
                        help='The configuration file of the experiment to execute')
    parser.add_argument('--experiment-type', '-t', type=str, required=True, choices=['dl', 'ml'],
                        help='The type of the experiment (must be "dl" or "ml")')

    args = parser.parse_args()

    # Define the correct Experiment Object
    if args.experiment_type == 'dl':
        experiment = DLE.DLExperiment(args.configuration_file)
    elif args.experiment_type == 'ml':
        experiment = MLE.MLExperiment(args.configuration_file)

    # Start Experiment
    experiment.experiment()

if __name__ == "__main__":
    main()
