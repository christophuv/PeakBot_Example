# PeakBot example

This script shows how to use the PeakBot framework

## train.py
This script shows how to train a new PeakBot model. It generates a training dataset (T) and 4 validation datasets (V, iT, iV, eV) from different LC-HRMS chromatograms and different reference feature and background lists automatically. Then using the computer's GPU the new PeakBot model is trained and evaluated. 

The main functions to generate training instances from a LC-HRMS chromatogram and a reference peak and background list are the functions `peakbot.train.cuda.generateTestInstances` for generating a large set of training instances and `peakbot.trainPeakBotModel` for training a new PeakBot model with the previously generated training instances.

More information about how to specify the files and the LC-HRMS properties is directly documented in the script. 

## detect.py 
This script shows how to detect chromatographic peaks in a new chromatogram with a PeakBot model. 

The main functions to detect chromatographic peaks in a LC-HRMS chromatogram with a PeakBot model are `peakbot.cuda.preProcessChromatogram` for extracting the standardized areas from the chromatogram and `peakbot.runPeakBot` for testing the standardized area for chromatographic peaks or backgrounds. 

More information about how to specify the files and the LC-HRMS properties is directly documented in the script. 
