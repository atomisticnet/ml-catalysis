<p align="center"><a href="http://ann.atomistic.net">
<img width="200" src="https://raw.githubusercontent.com/atomisticnet/ml-catalysis/master/aenet-logo-mlcat.png?token=ADZKL7KVB4YNWM6VGDX2YE27E5LMO" />
<a/></p>

# ml-catalysis – Machine Learning for Catalysis

This repository contains a collection of machine learning models for catalysis applications.

## Prediction of Ethanol Reforming Activity and Selectivity

This model is described in detail in: 

N. Artrith<super>*</super>, Z. Lin, and J. G Chen, <br/>
Predicting the Activity and Selectivity of Bimetallic Metal Catalysts for Ethanol Reforming using Machine Learning,<br/>
*ACS Catal.* 2020, **10**, 9438−9444, DOI: https://doi.org/10.1021/acscatal.0c02089

Please cite this reference if you make use of any parts of the source code or model or the DFT database.

<super>*</super>Contact: nartrith@atomistic.net

**Subdirectory: ethanol-reforming**

The scripts `01-activation-energy-model.py` and
`02-activity-and-selectivity-model.py` have to be run sequentially.  The
first script predicts transition-state energies based on DFT
thermochemical data.  The second script predicts reforming activities
and selectivities based on the transition-state energies from script 1.

### 01-activation-energy-model.py

    usage: 01-activation-energy-model.py [-h] [dft_data]

    Construct ML Model 1 for predicting transition-state energies from
    thermochemical DFT data and chemical information.

    The model uses a combination of Random Forest Regression and Gaussian
    Process Regression.

    2019-11-10 Nongnuch Artrith

    positional arguments:
      dft_data    CSV file with DFT data.

    optional arguments:
      -h, --help  show this help message and exit

### 02-activity-and-selectivity-model.py

    usage: 02-activity-and-selectivity-model.py [-h]
                                                [dft_data] [transition_state_data]
                                                [experimental_data]

    Construct ML Model 2 for predicting catalytic activities and
    selectivities.

    The models are based on linear regression.

    2019-11-10 Nongnuch Artrith

    positional arguments:
      dft_data              CSV file with DFT data.
      transition_state_data
                            CSV file with transition-state data from Model 1.
      experimental_data     CSV file with data from experimental characterization.

    optional arguments:
      -h, --help            show this help message and exit

### Example Output

    $ ./01-activation-energy-model.py
    CV RMSE (RFR+GPR) = 0.31367854134356526
    CV MAE  (RFR+GPR) = 0.19685553022494306
    $ ./02-activity-and-selectivity-model.py
    Reforming Activity Model:
      CV RMSE = 0.00360602875964415
      CV MAE  = 0.0033449441185262325

### Generated Output Files

**Output from script 1**

* validation-TS-model-RFR+GPR.png
* validation-TS-model-RFR+GPR.pdf
* predicted-TS-RF+GPR.csv

**Output from script 2**

* validation-reforming-activity-model.png
* validation-reforming-activity-model.pdf
* predicted-reforming-activity.csv
* validation-reforming-selectivity-from-total-activity.png
* validation-reforming-selectivity-from-total-activity.pdf
* predicted-reforming-selectivity-from-total-activity.csv
* validation-reforming-selectivity-logit.png
* validation-reforming-selectivity-logit.pdf
* predicted-reforming-selectivity-logit.csv

### Acknowledgments

DFT calculations and machine-learning model construction made use of the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1053575 (allocation no. DMR14005). Calculations were also performed on the computational resources of the Center for Functional Nano- materials, which is a U.S. DOE Office of Science Facility, at Brookhaven National Laboratory under Contract No. DE- SC0012704. We also acknowledge computing resources from Columbia University’s Shared Research Computing Facility project, which is supported by NIH Research Facility Improvement Grant 1G20RR030893-01, and associated funds from the New York State Empire State Development, Division of Science Technology and Innovation (NYSTAR) Contract C090171, both awarded April 15, 2010. This article was sponsored by the Catalysis Center for Energy Innovation (CCEI), an Energy Frontier Research Center (EFRC) funded by the U.S. Department of Energy, Office of Basic Energy Sciences under Award Number DE-SC0001004. N.A. thanks Dr. Jose Garrido Torres and Dr. Mark S Hybertsen for discussions.
