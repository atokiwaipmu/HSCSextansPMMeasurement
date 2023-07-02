
# This is the main program to run the whole project.
conda activate base
cd /Users/akiratokiwa/Git/HSCSextansPMMeasurement
python ./data/dataprocess.py 
python ./data/galaxycorrection.py
python ./data/QSOpm.py
python ./structure/memberselection.py
python ./structure/structure.py --flag True
python ./structure/halflightrad.py
python ./scripts/errorestimation.py
python ./scripts/numberdensity.py
python ./scripts/pmmeasurement.py
python ./visualization/plots.py
python ./revision/reflexmotion.py