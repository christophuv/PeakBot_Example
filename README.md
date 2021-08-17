# PeakBot example

This script shows how to use the PeakBot framework. In general, first a PeakBot CNN model is trained from a series of LC-HRMS chromatograms and a list of user-provided reference features that resemble the expected chromatographic peaks. For this, the different chromatographic peak shapes (fronting, tailing, etc.) should be considered so that the new PeakBot model can recognize them correctly. Once the training has finished and satisfying results with respect to the different loss and metric values have been reached, the model can be used to detect all similar chromatographic peaks in new LC-HRMS chromatograms in an untargeted manner. 

For training, different reference features must be specified. A comparison has shown that at least 100 different reference features should be provided. However, this can also be different isotopologs of the same feature. Nevertheless care should be taken to select and specify the many different chromatographic peak forms present in the data. 

The format to specify the list of reference features is a simple tab-separated values (tsv) file with the following columns:
RT | MZ | LeftRTBorder | RightRTBorder | MZDeviation
--- | --- | --- | --- | ---
1842 | 555.283089 | 1835 | 1847 | 24.5

The column RT specifies the approximate retention time (peak apex) of the reference feature in seconds. 
The column MZ specifies the approximate mass-to-charge (m/z) ratio of the reference feature. 
The column LeftRTBorder specifies the approximate retention time of the start of the reference feature in seconds. 
The column RightRTBorder specifies the approximate retention time of the end of the reference feature in seconds. 
The column MZDeviation specifies the approximate m/z deviation of the reference feature (approximately the 95% confidence interval). 

The format to specify the list of wall backgrounds is also a simple tsv file with the following columns:
MZ | Rtstart | Rtend | MZdeviation
--- | --- | --- | ---
119.98628 | 900 | 2200 | 20

The column MZ specifies the approximate m/z of the wall background. 
The column Rtstart specifies the start of the wall in seconds. 
The column Rtend specifies the end of the wall in seconds. 
The column MZdevation specifies the approximate m/z deviation of the wall background (approximately the 95% confidence interval). 

A random retention time between Rtstart and Rtend will be used. To specifically train PeakBot for certain wall backgrounds narrow retention time windows can be used. 

The format to specify the list of other backgrounds is also a simple tsv file with the following columns:
RTStart | RTEnd | MZLow | MZHigh
--- |--- |--- |---
200 | 600 | 800 | 1000

The column RTStart specifies the start of the background area in seconds. 
The column RTEnd specifies the end of the background area in seconds. 
The column MZLow specifies the lowest m/z of the background area. 
The column MZHigh specifies the largest m/z of the background area. 

The actual values of the reference features and backgrounds in the different reference chromatograms are automatically calculated using a gradient-descend appraoch. Thus, these determined reference feature values can vary from chromatogram to chromatogram. However, care must be taken that no co-eluting compounds (isomers) are present. 

PeakBot in general has no difficult data processing parameters. However, the user has to specify the following minimal settings for their chromatograms in order to instruct PeakBot how to train a new model: 
* nosieLevel: signal intensity below which a signal is considered to represent noise.
* minIntensity: minimum intensity of signals to be used as a local maximum seed.
* minRT: all scans earlier than this retention time will not be used by PeakBot (e.g. waste).
* maxRT: all scans later than this retention time will not be used by PeakBot (e.g. column reconditioning).
* RTpeakWidth: minimum and maximum peak with of a chromatographic peak (list of minimum and maximum value) used by the gradient-descend algorithm.
* intraScanMaxAdjacentSignalDifferencePPM: the ppm difference between profile mode signals belonging to the same feautre. Here the highest m/z differences should be used.
* interScanMaxSimilarSignalDifferencePPM: the ppm difference between signals of different scans representing the same profile mode signal.
The two parameters intraScanMaxAdjacentSignalDifferencePPM and interScanMaxSimilarSignalDifferencePPM are a bit difficult to evaluate, but can in general be used with the same values. It can be done with any software that visualized LC-HRMS chromatograms. In this respect TOPPView from the OpenMS toolbox (https://pubmed.ncbi.nlm.nih.gov/19425593/) can be used as it allows the user to easily calculate the m/z difference in ppm when selecting two signals and pressing the shift key. Alternatively, PeakBot offers a function to estimate these values automatically (see script estimateParameters.py). 

The following figure illustrates the different reference feature values and LC-HRMS settings. 
![illustration of PeakBot settings](https://github.com/christophuv/PeakBot_Example/Parameters.png)



## train.py
This script shows how to train a new PeakBot model. It generates a training dataset (T) and 4 validation datasets (V, iT, iV, eV) from different LC-HRMS chromatograms and different reference feature and background lists automatically. Then using the computer's GPU the new PeakBot model is trained and evaluated. 

The main functions to generate training instances from a LC-HRMS chromatogram and a reference peak and background list are the functions `peakbot.train.cuda.generateTestInstances` for generating a large set of training instances and `peakbot.trainPeakBotModel` for training a new PeakBot model with the previously generated training instances.

More information about how to specify the files and the LC-HRMS properties is directly documented in the script. 

## detect.py 
This script shows how to detect chromatographic peaks in a new chromatogram with a PeakBot model. 

The main functions to detect chromatographic peaks in a LC-HRMS chromatogram with a PeakBot model are `peakbot.cuda.preProcessChromatogram` for extracting the standardized areas from the chromatogram and `peakbot.runPeakBot` for testing the standardized area for chromatographic peaks or backgrounds. 

More information about how to specify the files and the LC-HRMS properties is directly documented in the script. 
