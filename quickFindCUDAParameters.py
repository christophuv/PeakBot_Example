#!/usr/bin/env python
# coding: utf-8

import sys
import os

def getUserInput(message, options={"yes": ["yes", "y"], "no": ["no", "n"]}, default = None):
    
    ok = False
    printErr = False
    foundOpt = None
    while not ok:
        if printErr:
            print("Error, please only provide one of the listed options")
        print(message)
        print("Options (select an answer and provide one of the strings that qualify for this answer): ")
        for opt in options.keys():
            print("  %s%s: %s"%(opt, "(default if you just press Enter)" if default == opt else "", ", ".join(s for s in options[opt])))
        value = input("Input: ")
        if value == "":
            value = default

        for opt in options.keys():
            if value in options[opt]:
                foundOpt = opt
                ok = True
        printErr = True

    return foundOpt


print('''

    This is a quick example for PeakBot
    -----------------------------------

    It will download the samples of the demonstration dataset MTBLS1358 (https://www.ebi.ac.uk/metabolights/MTBLS1358)
    and place them into the folder MTBLS1358 in the current directory. 

    Then it will generate the training and validation instances for a new PeakBot CNN model and 
    finally train a new model for detecting the chromatographic peaks. 

    Finally it will predict the chromatographic peaks in the samples. 


    To run it, please make sure that PeakBot has been installed. For instructions please refer to 
    https://github.com/christophuv/PeakBot
    

''')


print("Download dataset")
if getUserInput("Do you want to continue?") == "yes":
    
    ##################################################
    ### Doanload data
    ## This step can be omitted for other datasets
    ##

    import urllib.request
    import shutil

    files = {
        "HT_SOL1_LYS_010_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/kKgd5gBlIlsG9Bj/download",
        "HT_SOL1_SUP_025_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/42fEfzSUn4OLFDl/download",
        "HT_SOL2_LYS_014_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/84gnlkVxYdWnXC8/download",
        "HT_SOL2_SUP_029_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/fidRZt5EtJ1y57J/download",
        "HT_SOL3_LYS_018_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/O5QoSHBwl97QuV7/download",
        "HT_SOL3_SUP_033_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/RPyxKpP0RqEhHpm/download",
    }
    refs = {
        "Backgrounds.tsv": "https://ucloud.univie.ac.at/index.php/s/4IYdl3sdV92l8l3/download",
        "Peaks.tsv": "https://ucloud.univie.ac.at/index.php/s/UirxEIecOW55zkv/download",
        "Walls.tsv": "https://ucloud.univie.ac.at/index.php/s/XMMVhdHpsOdVVT9/download",
    }
    models = {
        "PBmodel_MTBLS1358.model.h5": "https://ucloud.univie.ac.at/index.php/s/8J0B6X9HxWx9qQ6/download"
    }

    try:
        shutil.rmtree(os.path.join(".", "MTBLS1358"))
    except:
        pass

    for a in [os.path.join(".","MTBLS1358"), 
              os.path.join(".","MTBLS1358", "Data"), 
              os.path.join(".","MTBLS1358", "Reference"),
              os.path.join(".","MTBLS1358", "Model"), 
              os.path.join(".","MTBLS1358", "Temp")]:
        try:
            os.mkdir(a)
        except:
            pass

    print("Downloading data files")
    for fileName, url in files.items():
        print("  ", os.path.join(".","MTBLS1358", "Data", fileName))
        urllib.request.urlretrieve(url, os.path.join(".","MTBLS1358", "Data", fileName))
    print("Downloading reference files")
    for fileName, url in refs.items():
        print("  ", os.path.join(".","MTBLS1358", "Reference", fileName))
        urllib.request.urlretrieve(url, os.path.join(".","MTBLS1358", "Reference", fileName))
    print("Downloading pre-trained model")
    for fileName, url in models.items():
        print("  ", os.path.join(".","MTBLS1358", "Model", fileName))
        urllib.request.urlretrieve(url, os.path.join(".","MTBLS1358", "Model", fileName))
print("\n\n\n\n\n")













##################################################
### Parameter definition
## 

## Specific tensorflow configuration. Can re omitted
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.get_logger().setLevel('WARNING')


###############################################
### Name of the experiment
##
## expNAme: specifies the name of the experiment and the folder in which the data will be processed
expName = "MTBLS1358"


###############################################
### data parameters
##
## Different LC-HRMS settings can be used per chromatogram. To be able to reuse them, they are summarized
##    in dictionaries. The keys are then used as the setting values
## 
## PeakBotModel: file to save the new PeakBot CNN model to
## polarities: specifies which filter lines are to be used for detecting the chromatographic peaks
## noiseLevel: Everything below this threshold is considered noise and removed directly after the import
## minRT / maxRT: Area of the chromatogram in which chromatographic peaks are expected
## RTpeakWidth: array of [minimum, maximum] peak-width in scans
## intraScanMaxAdjacentSignalDifferencePPM: Maximum difference of signals belonging to the same profile mode peak
## interScanMaxSimilarSignalDifferencePPM: Maximum difference of signals representing the same profile mode signal
## minIntensity: All signals below this threshold are not considered for the local maximum detection
expParams = {"PeakBotModel": os.path.join(".", expName, "Model", "PBmodel_MTBLS1358.model.h5"),
             "polarities": {"positive": "Q Exactive HF (MS lvl: 1, pol: +)"},
             "minRT": 30, "maxRT": 680, "RTpeakWidth":[2,30], "SavitzkyGolayWindowPlusMinus": 3,
             "intraScanMaxAdjacentSignalDifferencePPM": 15, "interScanMaxSimilarSignalDifferencePPM": 3,
             "noiseLevel": 1E3, "minIntensity": 1E5}

###############################################
### chromatograms to process for training/validation
inFiles = {        
    "HT_SOL1_LYS_010_pos": os.path.join(".", expName, "Data", "HT_SOL1_LYS_010_pos.mzXML"),
    "HT_SOL2_LYS_014_pos": os.path.join(".", expName, "Data", "HT_SOL2_LYS_014_pos.mzXML"),
}
exFiles = {
    "HT_SOL3_LYS_018_pos": os.path.join(".", expName, "Data", "HT_SOL3_LYS_018_pos.mzXML"),
    #"HT_SOL3_SUP_033_pos": os.path.join(".", expName, "Data", "HT_SOL3_SUP_033_pos.mzXML"), ## Reference peaks are not taken from SUP samples, thus they should not be used here
}


###############################################
### chromatograms to process during detection
detFiles = {
    "HT_SOL1_LYS_010_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL1_LYS_010_pos.mzXML"),
    #"HT_SOL1_SUP_025_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL1_SUP_025_pos.mzXML"),
    #"HT_SOL2_LYS_014_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL2_LYS_014_pos.mzXML"),
    #"HT_SOL2_SUP_029_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL2_SUP_029_pos.mzXML"),
    #"HT_SOL3_LYS_018_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL3_LYS_018_pos.mzXML"),
    #"HT_SOL3_LYS_033_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL3_SUP_033_pos.mzXML"),
}

###############################################
## GPU information
##
## These values specify how the GPU is used for generating the training examples
## Please consult the documentation of your GPU.
## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
## The strategy specifies on which device tensorflow shall be executed.
blockdim = 256
griddim  = 256
exportBatchSize = 2048
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

###############################################
## Values for generating the training instances
##
## maxPopulation: This specifies how many different references (features, backgrounds) shall be combined in a
##   single training instance. The values specifies the maximum number and values between 1 and maxPopulation
##   will be picked randomly
## intensityScales: Specifies the factor with which the references will be scales (1/intensityScales to intensityScales)
## randomnessFactor: Specifies the factor which which the individual signals of the raw data are multiplied
maxPopulation = 4
intensityScales = 10
randomnessFactor = 0.1




# Imports
import os
import pickle
import shutil
import tempfile
import tqdm

## Load the PeakBot package
import sys
sys.path.append(os.path.join("..", "peakbot", "src"))
import peakbot
import peakbot.train.cuda
import peakbot.Chromatogram
from peakbot.core import tic, toc, tocP, TabLog


## Function for loading mzXML files (and saving them as pickle file for fast access)
def loadFile(path, verbose = False):
    tic()
    mzxml = None
    if os.path.exists(path+".pickle"):
        with open(path+".pickle", "rb") as inF:
            mzxml = pickle.load(inF)
        if verbose: print("Imported chromatogram.pickle for '%s'"%(path))
    else:
        mzxml = peakbot.Chromatogram.Chromatogram()
        mzxml.parse_file(path)
        with open(path+".pickle", "wb") as outF:
            pickle.dump(mzxml, outF)
        if verbose: print("Imported chromatogram for '%s'"%(path))
    return mzxml











##################################################
### Detect chromatographic peaks
##
print("The trained model (loading from '%s') will be used to detect chromatographic peaks in LC-HRMS chromatograms"%(expParams["PeakBotModel"]))
TabLog().reset()

###############################################
## Finisehd with specifying LC-HRMS chromatogram files and LC-HRMS settings
## Nothing to change from here on
for exportBatchSize in reversed([512,1024,2048,4096,8192,8192+4096]):
    for blockdim in reversed([1,2,4,8,16,32,64,128,256,512,1024,2048]):
        for griddim in reversed([1,2,4,8,16,32,64,128,256,512,1024,2048]):
            if blockdim * griddim >= 128:
                tic(label="overall")
                for trep in  range(10):
                    successfullAll = True
                    try:                        
                        for inFile, fileLoc in detFiles.items():
                            
                            for polarity, filterLine in expParams["polarities"].items():
                                with tempfile.TemporaryDirectory() as tmpdirname:
                                    
                                    ###############################################
                                    ### Preprocess chromatogram
                                    mzxml = loadFile(fileLoc)
                                    mzxml.keepOnlyFilterLine(filterLine)
                                    mzxml.removeBounds(minRT = expParams["minRT"], maxRT = expParams["maxRT"])
                                    mzxml.removeNoise(expParams["noiseLevel"])
                                    
                                    ###############################################
                                    ### Detect local maxima with peak-like shapes## CUDA-GPU
                                    tic(label="preProcessing")
                                    peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
                                            mzxml, "'%s':'%s'"%(inFile, filterLine), 
                                            intraScanMaxAdjacentSignalDifferencePPM = expParams["intraScanMaxAdjacentSignalDifferencePPM"],
                                            interScanMaxSimilarSignalDifferencePPM = expParams["interScanMaxSimilarSignalDifferencePPM"],
                                            RTpeakWidth = expParams["RTpeakWidth"],
                                            SavitzkyGolayWindowPlusMinus = expParams["SavitzkyGolayWindowPlusMinus"], 
                                            minIntensity = expParams["minIntensity"],
                                            exportPath = tmpdirname, 
                                            exportLocalMaxima = "peak-like-shape", # "all", "localMaxima-with-mzProfile", "peak-like-shape"
                                            exportBatchSize = exportBatchSize, 
                                            blockdim = blockdim,
                                            griddim  = griddim, 
                                            verbose = False)
                                    
                                    ###############################################
                                    ### Detect peaks with PeakBot
                                    peaks = []
                                    with strategy.scope():
                                        peaks, walls, backgrounds, errors = peakbot.runPeakBot(tmpdirname, expParams["PeakBotModel"], verbose = False)
                                    
                                    ###############################################
                                    ### Postprocessing
                                    peaks = peakbot.cuda.postProcess(mzxml, "'%s':'%s'"%(inFile, filterLine), peaks, 
                                                                        blockdim = blockdim,
                                                                        griddim  = griddim, 
                                                                        verbose = False)
                                    
                                    ## Log features
                                    peakbot.exportPeakBotResultsFeatureML(peaks, "%s_%sPeakBot.featureML"%(fileLoc.replace(".mzXML", ""), polarity))
                                    peakbot.exportPeakBotResultsTSV(peaks, "%s_%sPeakBot.tsv"%(fileLoc.replace(".mzXML", ""), polarity))
                            
                    except Exception:
                        print("(blockdim %s, griddim %s, exportBatchSize %s) failed"%(blockdim, griddim, exportBatchSize))
                        successfullAll = False
                        break
                if successfullAll:
                    TabLog().addData("Total time all files (blockdim %s, griddim %s, exportBatchSize %s)"%(blockdim, griddim, exportBatchSize), "time (sec)", "%.1f"%toc("overall"))
                    TabLog().print(sortby = "time (sec)", reverse = True)
                    print("\n\n\n\n\n")
