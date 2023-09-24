# Spy Inside：Formal Verification of Dependable Transformers and Recurrent Neural Networks for Data Analysis Systems

This repository contains the related material of our work: Spy Inside：Formal Verification of Dependable Transformers and Recurrent Neural Networks for Data Analysis Systems.

## Preparation

### System Requirement

* Running the experiments on GPU is highly recommended. 
* The verification approach relies on the [Gurobi optimiser](https://www.gurobi.com/) version 9.1.  Make sure to activate the Gurobi license for version 9.1 before running the experiments. Gurobi has provided the insturction of academic licence [here](https://www.gurobi.com/academia/academic-program-and-licenses/) to apply for academic ues.
* [Conda](https://docs.conda.io/en/latest/) environment is recommended to easily install the dependencies. 

### Virtual Environment Setup

`requirements.txt` is the frozen conda list of the environment. The necessary virtual environment can be setup by:

```
conda create --name TrustworthySystems --file requirements.txt --channel gurobi --channel pytorch
conda activate TrustworthySystems
```

Then the Gurobi licence needs to be activated.


## Main Codes

1. `Abstraction_*.py`: The core code of the formal verification approach, including the abstract classes of neural network layers in RNNs and Transformers.
2. `relaxation_*.py`: The code majorly for the linear programming based relaxation of the non-linear operations.
3. `utils.py`:  All methods needed throughout the approach.
4. `requirements.txt`: The dependencies in our virtual environment.


## Reproducing Results


### 1 Network Traffic Classifier

The folder *Classifier* contains the datasets and files to retrain a classifier model. Run *run_evaluation_scenarios.sh* to check the guidence.

The folder *Model* contains the trained model of the classifier in our experiments. We import the training weights and biases from the trained model to the abstract model for verification.

The file *Abstraction_Classifier.py* contains the encoded formal model of the classifier by connecting the layers implemented with the neural network classes in *Abstraction_Transformer.py*.

Run the file *Verification_Classifier.py* to verify whether the inaccuracy in input will disturb the correctness of the system classification. The experiments are conducted on 100 network traffic that are correctly classified by the original classifier in folder */Classifier/traffic_classifier/datasets*.

### 2 News Recommendation

The folder *data* contains the datasets we used to train LSTUR model. Run *data_preprocess.py* to preprocess data into appropriate format. The folder *LSTUR* contains the files to retrain a LSTUR model.

The folder *Model* contains the trained model of news recommender system LSTUR. We import the training weights and biases from the trained model to the abstract model for verification.

The file *Abstraction_LSTUR.py* contains the encoded formal model of the news recommder system by connecting the layers implemented with the neural network classes in *Abstraction_RNN.py*.

Run the file *Verification_LSTUR.py* to verify the impact of data deficiency in input on the system output. The experiments are conducted on the news impression logs of 100 users in *test.tsv*. 

You can also do your own single verification by directly  changing the `--test_userlogs` from the value in the script to verify on different datasets of historical user behavior.

```
python Verification_LSTUR.py --test_userlogs testfile.tsv
```

### 3 Video Quality Prediction

The folder *data* contains the example of the datasets we employed to train and test VQPN model. The original datasets are very large with *6.20 GB*. We will provide the link to download the datasets later.

The folder *Model* contains the trained model of the video quality prediction network. We import the training weights and biases from the trained model to the abstract model for verification.

The file *Abstraction_VQPN.py* contains the encoded formal model of the classifier by connecting the layers implemented with the neural network classes in *Abstraction_RNN.py*.

Run the file *Verification_VQPN.py* to verify whether the attack perturbation to the input data will affect the the systems prediction. We can also verify how VQPN respond when missing features happen to the input follow the guidence in our work.

## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
- Network Trasffic Classifier model, see https://github.com/RadionBik/ML-based-network-traffic-classifier
- VQPN implemented with Tensorflow, see https://github.com/thu-media/QARC

