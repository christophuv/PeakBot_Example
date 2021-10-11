#!/usr/bin/env python
# coding: utf-8

## run in python >= 3.8
## activate conda environment on jucuda
## conda activate python3.8

# Imports
import os
import pickle
import tempfile
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf## set memory limit on the GPU to 2GB to not run into problems with the pre- and post-processing steps
tf.config.experimental.set_virtual_device_configuration(
    tf.config.experimental.list_physical_devices('GPU')[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2048)]
)

import sys
sys.path.append(os.path.join("..", "peakbot", "src"))
import peakbot
import peakbot.Chromatogram
import peakbot.cuda            
from peakbot.core import tic, toc, tocP, TabLog


def loadFile(path):
    tic()
    mzxml = None
    if os.path.exists(path+".pickle"):
        with open(path+".pickle", "rb") as inF:
            mzxml = pickle.load(inF)
        print("Imported chromatogram.pickle for '%s'"%(path))
    else:
        mzxml = peakbot.Chromatogram.Chromatogram()
        mzxml.parse_file(path)
        with open(path+".pickle", "wb") as outF:
            pickle.dump(mzxml, outF)
        print("Imported chromatogram for '%s'"%(path))
    return mzxml





###############################################
### Process files 
##
if __name__ == "__main__":
    tic(label="overall")

    ###############################################
    ### data parameters
    ##
    ## Different LC-HRMS settings can be used per chromatogram. To be able to reuse them, they are summarized
    ##    in dictionaries. The keys are then used as the setting values
    ##
    ## polarities: specifies which filter lines are to be used for detecting the chromatographic peaks
    ## noiseLevel: Everything below this threshold is considered noise and removed directly after the import
    ## minRT / maxRT: Area of the chromatogram in which chromatographic peaks are expected
    ## RTpeakWidth: array of [minimum, maximum] peak-width in scans
    ## SavitzkyGolayWindowPlusMinus: specifies the degree of smoothing. A value of x results in a smoothing window of 2*x + 1
    ## intraScanMaxAdjacentSignalDifferencePPM: Maximum difference of signals belonging to the same profile mode peak
    ## interScanMaxSimilarSignalDifferencePPM: Maximum difference of signals representing the same profile mode signal
    ## minIntensity: All signals below this threshold are not considered for the local maximum detection
    ##
    ###############################################
    ### chromatograms to process
    ##
    ## Different LC-HRMS chromatograms can be used for generating a training or validation dataset
    ##
    ## file: Path of the mzXML file
    ## params: parameter collection for the particular sample (see variable expParams)
    expParams = {}
    inFiles = OrderedDict()
    if False:
        # Unreleased data
        # Wheat ear (similar to untreated samples of: Stable Isotope–Assisted Plant Metabolomics: Combination of Global and Tracer-Based Labeling for Enhanced Untargeted Profiling and Compound Annotation)
        # https://doi.org/10.3389/fpls.2019.01366
        expParams["WheatEar"] = {"PeakBotModel": "./temp/PBmodel_WheatEar.model.h5",
                                "polarities": {"positive": "Q Exactive (MS lvl: 1, pol: +)", 
                                                "negative": "Q Exactive (MS lvl: 1, pol: -)"},
                                "minRT":150, "maxRT":2250, "RTpeakWidth":[8,120], "SavitzkyGolayWindowPlusMinus": 3,
                                "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                                "noiseLevel":1E3, "minIntensity":1E5}
        inFiles["670_Sequence3_LVL1_1"    ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1_1.mzXML"  , "params": "WheatEar"}
        inFiles["670_Sequence3_LVL1_2"    ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1_2.mzXML"  , "params": "WheatEar"}
        inFiles["670_Sequence3_LVL1_3"    ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1_3.mzXML"  , "params": "WheatEar"}
        inFiles["670_Sequence3_LVL1x2_1"  ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_1.mzXML", "params": "WheatEar"}
        inFiles["670_Sequence3_LVL1x2_2"  ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_2.mzXML", "params": "WheatEar"}
        inFiles["670_Sequence3_LVL1x2_3"  ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_3.mzXML", "params": "WheatEar"}
        inFiles["670_Sequence3_LVL1x4_1"  ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_1.mzXML", "params": "WheatEar"}
        inFiles["670_Sequence3_LVL1x4_2"  ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_2.mzXML", "params": "WheatEar"}
        inFiles["670_Sequence3_LVL1x4_3"  ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_3.mzXML", "params": "WheatEar"}
        
        # Unreleased data
        # AOH and AME metabolism in PHM cell lines
        expParams["PHM"] = {"PeakBotModel": "./temp/PBmodel_WheatEar.model.h5",
                            "polarities": {"positive": "LTQ Orbitrap Velos (MS lvl: 1, pol: +)"},
                            "minRT":100, "maxRT":750, "RTpeakWidth":[4,120], "SavitzkyGolayWindowPlusMinus":3,
                            "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                            "noiseLevel":1E3, "minIntensity":1E5}
        inFiles["05_EB3388_AOH_p_0" ] = {"file": "./Data/PHM/05_EB3388_AOH_p_0.mzXML" , "params": "PHM"}
        inFiles["06_EB3389_AOH_p_10"] = {"file": "./Data/PHM/06_EB3389_AOH_p_10.mzXML", "params": "PHM"}
        inFiles["07_EB3390_AOH_p_20"] = {"file": "./Data/PHM/07_EB3390_AOH_p_20.mzXML", "params": "PHM"}
        inFiles["08_EB3391_AOH_p_60"] = {"file": "./Data/PHM/08_EB3391_AOH_p_60.mzXML", "params": "PHM"}
        inFiles["16_EB3392_AME_p_0" ] = {"file": "./Data/PHM/16_EB3392_AME_p_0.mzXML" , "params": "PHM"}
        inFiles["17_EB3393_AME_p_10"] = {"file": "./Data/PHM/17_EB3393_AME_p_10.mzXML", "params": "PHM"}
        inFiles["18_EB3394_AME_p_20"] = {"file": "./Data/PHM/18_EB3394_AME_p_20.mzXML", "params": "PHM"}
        inFiles["19_EB3395_AME_p_60"] = {"file": "./Data/PHM/19_EB3395_AME_p_60.mzXML", "params": "PHM"}
        
    # MTBLS1358: Stable Isotope-Assisted Metabolomics for Deciphering Xenobiotic Metabolism in Mammalian Cell Culture
    # https://www.ebi.ac.uk/metabolights/MTBLS1358/descriptors
    expParams["MTBLS1358"] = {"PeakBotModel": "./temp/PBmodel_MTBLS1358.model.h5",
                            "polarities": {"positive": "Q Exactive HF (MS lvl: 1, pol: +)"},
                            "minRT":30, "maxRT":680, "RTpeakWidth":[3,30], "SavitzkyGolayWindowPlusMinus": 1,
                            "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                            "noiseLevel":1E4, "minIntensity":1E6}
    inFiles["HT_SOL1_LYS_010_pos_cm"] = {"file": "./Data/MTBLS1358/HT_SOL1_LYS_010_pos.mzXML", "params": "MTBLS1358"}
    inFiles["HT_SOL1_SUP_025_pos_cm"] = {"file": "./Data/MTBLS1358/HT_SOL1_SUP_025_pos.mzXML", "params": "MTBLS1358"}
    inFiles["HT_SOL2_LYS_014_pos_cm"] = {"file": "./Data/MTBLS1358/HT_SOL2_LYS_014_pos.mzXML", "params": "MTBLS1358"}
    inFiles["HT_SOL2_SUP_029_pos_cm"] = {"file": "./Data/MTBLS1358/HT_SOL2_SUP_029_pos.mzXML", "params": "MTBLS1358"}
    inFiles["HT_SOL3_LYS_018_pos_cm"] = {"file": "./Data/MTBLS1358/HT_SOL3_LYS_018_pos.mzXML", "params": "MTBLS1358"}
    inFiles["HT_SOL3_LYS_033_pos_cm"] = {"file": "./Data/MTBLS1358/HT_SOL3_SUP_033_pos.mzXML", "params": "MTBLS1358"}
    if False:
        # MTBLS797: Bluebell saponins: Application of metabolomics and molecular networking
        # https://www.ebi.ac.uk/metabolights/MTBLS797/descriptors
        expParams["MTBLS797"] = {"PeakBotModel": "./temp/PBmodel_MTBLS797.model.h5",
                                "polarities": {"positive": "Exactive (MS lvl: 1, pol: +)", 
                                                "negative": "Exactive (MS lvl: 1, pol: -)"},
                                "minRT":100, "maxRT":2200, "RTpeakWidth":[2,30], "SavitzkyGolayWindowPlusMinus": 2,
                                "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                                "noiseLevel":1E3, "minIntensity":1E4}
        inFiles["Dotsha01"] = {"file": "./Data/MTBLS797/Dotsha01.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha02"] = {"file": "./Data/MTBLS797/Dotsha02.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha03"] = {"file": "./Data/MTBLS797/Dotsha03.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha04"] = {"file": "./Data/MTBLS797/Dotsha04.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha05"] = {"file": "./Data/MTBLS797/Dotsha05.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha06"] = {"file": "./Data/MTBLS797/Dotsha06.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha07"] = {"file": "./Data/MTBLS797/Dotsha07.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha08"] = {"file": "./Data/MTBLS797/Dotsha08.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha09"] = {"file": "./Data/MTBLS797/Dotsha09.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha10"] = {"file": "./Data/MTBLS797/Dotsha10.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha11"] = {"file": "./Data/MTBLS797/Dotsha11.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha12"] = {"file": "./Data/MTBLS797/Dotsha12.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha13"] = {"file": "./Data/MTBLS797/Dotsha13.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha14"] = {"file": "./Data/MTBLS797/Dotsha14.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha15"] = {"file": "./Data/MTBLS797/Dotsha15.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha16"] = {"file": "./Data/MTBLS797/Dotsha16.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha17"] = {"file": "./Data/MTBLS797/Dotsha17.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha18"] = {"file": "./Data/MTBLS797/Dotsha18.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha19"] = {"file": "./Data/MTBLS797/Dotsha19.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha20"] = {"file": "./Data/MTBLS797/Dotsha20.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha21"] = {"file": "./Data/MTBLS797/Dotsha21.mzXML", "params": "MTBLS797"}
        inFiles["Dotsha22"] = {"file": "./Data/MTBLS797/Dotsha22.mzXML", "params": "MTBLS797"}

        # MTBLS868: Mining for natural product antileishmanials in a fungal extract library
        # https://www.ebi.ac.uk/metabolights/MTBLS868/descriptors
        expParams["MTBLS868"] = {"PeakBotModel": "./temp/PBmodel_MTBLS868.model.h5",
                                "polarities": {"positive": "Q Exactive (MS lvl: 1, pol: +)", 
                                                "negative": "Q Exactive (MS lvl: 1, pol: -)"},
                                "minRT":100, "maxRT":1400, "RTpeakWidth":[5,120], "SavitzkyGolayWindowPlusMinus": 1,
                                "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                                "noiseLevel":1E3, "minIntensity":1E5}
        inFiles["HD871_1_1"     ] = {"file": "./Data/MTBLS868/HD871_1_1.mzXML"     , "params": "MTBLS868"}
        inFiles["HD871_1_2"     ] = {"file": "./Data/MTBLS868/HD871_1_2.mzXML"     , "params": "MTBLS868"}
        inFiles["HD871_1_3"     ] = {"file": "./Data/MTBLS868/HD871_1_3.mzXML"     , "params": "MTBLS868"}
        inFiles["HD871_1_4"     ] = {"file": "./Data/MTBLS868/HD871_1_4.mzXML"     , "params": "MTBLS868"}
        inFiles["HD871_2_1"     ] = {"file": "./Data/MTBLS868/HD871_2_1.mzXML"     , "params": "MTBLS868"}
        inFiles["HD871_2_2"     ] = {"file": "./Data/MTBLS868/HD871_2_2.mzXML"     , "params": "MTBLS868"}
        inFiles["HD871_2_3"     ] = {"file": "./Data/MTBLS868/HD871_2_3.mzXML"     , "params": "MTBLS868"}
        inFiles["HD871_2_4"     ] = {"file": "./Data/MTBLS868/HD871_2_4.mzXML"     , "params": "MTBLS868"}
        inFiles["Untreated_1"   ] = {"file": "./Data/MTBLS868/Untreated_1.mzXML"   , "params": "MTBLS868"}
        inFiles["Untreated_2"   ] = {"file": "./Data/MTBLS868/Untreated_2.mzXML"   , "params": "MTBLS868"}
        inFiles["Untreated_3"   ] = {"file": "./Data/MTBLS868/Untreated_3.mzXML"   , "params": "MTBLS868"}
        inFiles["Untreated_4"   ] = {"file": "./Data/MTBLS868/Untreated_4.mzXML"   , "params": "MTBLS868"}
        inFiles["matrix_blank_1"] = {"file": "./Data/MTBLS868/matrix_blank_1.mzXML", "params": "MTBLS868"}
        inFiles["matrix_blank_2"] = {"file": "./Data/MTBLS868/matrix_blank_2.mzXML", "params": "MTBLS868"}
        inFiles["matrix_blank_3"] = {"file": "./Data/MTBLS868/matrix_blank_3.mzXML", "params": "MTBLS868"}
        inFiles["matrix_blank_4"] = {"file": "./Data/MTBLS868/matrix_blank_4.mzXML", "params": "MTBLS868"}

    ###############################################
    ## GPU information
    ##
    ## These values specify how the GPU is used for generating the training examples
    ## Please consult the documentation of your GPU.
    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
    ## The strategy specifies on which device tensorflow shall be executed.
    ## exportBatchSize: specifies how many putative areas shall be exported in one batch
    ## peakBotModelFile: specifies which model to load from the file system
    blockdim = 512
    griddim  = 256
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    exportBatchSize = 2048
    
    
    ###############################################
    ## Finisehd with specifying LC-HRMS chromatogram files and LC-HRMS settings
    ## Nothing to change from here on
    

    ###############################################
    ### Iterate files and polarities (if FPS is used)
    for inFile, fileProps in inFiles.items():
        tic(label="sample")
    
        ###############################################
        ### data parameters for chromatograms        
        params = expParams[fileProps["params"]]
        
        for polarity, filterLine in params["polarities"].items():
            print("Processing sample '%s', polarity '%s'"%(inFile, polarity))
            with tempfile.TemporaryDirectory() as tmpdirname:
                tic("instance")
                
                ###############################################
                ### Preprocess chromatogram
                tic("sample")
                mzxml = loadFile(fileProps["file"])
                mzxml.keepOnlyFilterLine(filterLine)
                print("  | .. filtered mzXML file for %s scan events only"%(polarity))
                mzxml.removeBounds(minRT = params["minRT"], maxRT = params["maxRT"])
                mzxml.removeNoise(params["noiseLevel"])
                print("  | .. removed noise and bounds")
                print("")
                
                ###############################################
                ### Detect local maxima with peak-like shapes## CUDA-GPU
                tic(label="preProcessing")
                peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
                        mzxml, "'%s':'%s'"%(inFile, filterLine), 
                        intraScanMaxAdjacentSignalDifferencePPM = params["intraScanMaxAdjacentSignalDifferencePPM"],
                        interScanMaxSimilarSignalDifferencePPM = params["interScanMaxSimilarSignalDifferencePPM"],
                        RTpeakWidth = params["RTpeakWidth"],
                        SavitzkyGolayWindowPlusMinus = params["SavitzkyGolayWindowPlusMinus"], 
                        minIntensity = params["minIntensity"],
                        exportPath = tmpdirname, 
                        exportLocalMaxima = "peak-like-shape", # "all", "localMaxima-with-mzProfile", "peak-like-shape"
                        exportBatchSize = exportBatchSize, 
                        blockdim = blockdim,
                        griddim  = griddim, 
                        verbose = True)
                print("")
                TabLog().addData("%s - %s"%(inFile, filterLine), "LM", len(peaks))
                peakbot.exportLocalMaximaAsFeatureML("%s_%sLM.featureML"%(fileProps["file"].replace(".mzXML", ""), polarity), peaks)
                
                ###############################################
                ### Detect peaks with PeakBot
                tic("PeakBotDetection")
                peaks = []
                with strategy.scope():
                    peaks, walls, backgrounds, errors = peakbot.runPeakBot(tmpdirname, params["PeakBotModel"])
                print("")
                
                ###############################################
                ### Postprocessing
                tic("postProcessing")
                peaks = peakbot.cuda.postProcess(mzxml, "'%s':'%s'"%(inFile, filterLine), peaks, 
                                                 blockdim = blockdim,
                                                 griddim  = griddim, 
                                                 verbose = True)
                print("")
                
                ## Log features
                TabLog().addData("%s - %s"%(inFile, filterLine), "Features", len(peaks))
                TabLog().addData("%s - %s"%(inFile, filterLine), "Walls", len(walls))
                TabLog().addData("%s - %s"%(inFile, filterLine), "Backgrounds", len(backgrounds))
                TabLog().addData("%s - %s"%(inFile, filterLine), "Errors", len(errors))
                peakbot.exportPeakBotResultsFeatureML(peaks, "%s_%sPeakBot.featureML"%(fileProps["file"].replace(".mzXML", ""), polarity))
                peakbot.exportPeakBotResultsTSV(peaks, "%s_%sPeakBot.tsv"%(fileProps["file"].replace(".mzXML", ""), polarity))
                peakbot.exportPeakBotWallsFeatureML(walls, "%s_%sPeakBotWalls.featureML"%(fileProps["file"].replace(".mzXML", ""), polarity))
                print("Exported PeakBot detected peaks..")
                print("")
                
                tocP("File '%s':'%s': Preprocessed, exported and predicted with PeakBot"%(inFile, filterLine), label="instance")
                TabLog().addData("%s - %s"%(inFile, filterLine), "time (sec)", "%.1f"%toc("instance"))
                print("\n\n\n\n\n")                
        
    TabLog().addData("Total time all files", "time (sec)", "%.1f"%toc("overall"))
    print("")
    TabLog().print()
