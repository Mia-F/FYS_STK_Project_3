# FYS-STK3155/4155 Project 3

- [Assignment](https://github.com/CompPhysics/MachineLearning/blob/master/doc/Projects/2023/Project3/pdf/Project3.pdf)

We have chosen to try and predict Bitcoin prices by using various machine learning methods, including an LSTM neural network, Decision Tree, and Random Forest, both with and without bagging. This repository contains all the necessary components to replicate the results presented in our report.

### Requirements
To run the python programs, the following python packages must be installed
- numpy
- pandas
- mathplotlib
- seaborn
- sklearn
- keras
- TA-Lib
- tensorflow
- joblib
- random

### Structure
- `Decision_tree.py`: This file contains all the necessary code to reproduce the results we obtained for the Decision Tree segment of the project. Due to some inherent randomness, the outcomes may not be identical to those reported, but they should be very similar.
- `Random_forest.py`:This file contains all the necessary code to reproduce the results we obtained for the Random forest segment of the project. Due to some inherent randomness, the outcomes may not be identical to those reported, but they should be very similar.
- `NN_BTC_PROJ.py`: This file contains all the necessary code to reproduce the results we obtained for the neural network segment of the project.

### Run code
To successfully execute the code, please note that you might need to modify the file path in the script to correctly access the data file located in the 'Data' folder. Ensure that all required packages are installed, and then enter the following command in the terminal to run the codes: 

```bash
python3 NN_BTC_PROJ.py
python3 Decision_tree.py
python3 Random_forest.py
```

### Authors
- Mia Synnøve Frivik
- Andrea Myrvang
- Max Jan Willem Schuringa
- Oskar Våle
