# US-CENSUS-ANALYSIS
US-CENSUS-ANALYSIS develops a classification model to predict if an individual is making more than $50,000 per year.
GitHub: https://github.com/olivier-lapabe/US-CENSUS-ANALYSIS

The full documentation can be accessed here : 

## Getting Started

### Dependencies

Refer to `requirements.txt` for a full list of dependencies.

### Installing

#### For Users:

* To install US-CENSUS-ANALYSIS:
```
git clone https://github.com/olivier-lapabe/US-CENSUS-ANALYSIS.git
cd US-CENSUS-ANALYSIS
pip install .
```

#### For Developers/Contributors:

If you're planning to contribute or test the latest changes, you should first set up a virtual environment and then install the package in "editable" mode. This allows any changes you make to the source files to immediately affect the installed package without requiring a reinstall.

* Clone the repository:
```
git clone https://github.com/olivier-lapabe/US-CENSUS-ANALYSIS.git
cd US-CENSUS-ANALYSIS
```

* Set up a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate # On Windows, use: .venv\Scripts\activate
```

* Install the required dependencies:
```
pip install -r requirements.txt
```

* Install US-CENSUS-ANALYSIS in editable mode:
```
pip install -e .
```

### Executing program

After adapting config.py, launch the program that trains and evaluate a classifiation model:  
```
python3 main.py
```
