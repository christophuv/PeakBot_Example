clear



echo Cleaning old results
find ./Data -name "SummaryPlot*.png" -type f -delete
find ./Data -name "History*.pickle" -type f -delete
find ./Data -name "*.featureML" -type f -delete
find ./Data -name "*.tsv" -type f -delete
rm -rf ./temp/*
echo



echo Training PeakBot models
echo

# Wheat ear dataset
python trainPB_WheatEar.py
echo
echo

# Wheat PHM dataset
python trainPB_PHM.py
echo
echo

# HT29 (Mira Flasch) dataset
python trainPB_MTBLS1358.py
echo
echo

# MTBLS797 dataset
python trainPB_MTBLS797.py
echo
echo

# MTBLS868 dataset
python trainPB_MTBLS868.py
echo
echo

