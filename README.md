# Quantitative Structure-Activity Relationship of Fish Toxicity

This project aims to analyze the quantitative structure-activity relationship to predict acute aquatic toxicity towards the fish Pimephales promelas (fathead minnow) on a set of 908 chemicals using K-Nearest Neighbor (kNN).

## Dataset

The dataset consists of six molecular descriptor as features and their corresponding LC50 values, which are:

| Feature   | Description |
| ----------| ------------------------------------------------------------ |
| CIC0      | Set of indices of neighbourhood symmetry                     |
| SM1_Dz(Z) | Set of descriptors calculated from 2D matrices derived from the molecular graph (2D matrix-based descriptors) |
| GATS1i    | 2D Geary autocorrelation descriptor                              |
| NdsCH     | count the number of unsaturated sp2 carbon atoms of the type =CH-|
| NdssC     | Count the number of unsaturated sp2 carbon atoms of the type =C  |
| MLOGP     | The octanol-water partitioning coefficient (log P) calculated by means of the Moriguchi model |

Dataset source : [here](https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity).

## Results

During the training process, we used 10-fold cross validation to evaluate the model's performance for a range of values for the number of nearest neighbors (k) from 2 to 9. After training the model with different values of k, we found that the model performed best when k was set to 8.

The results are shown below:

| Metric | Training Score | Testing Score |
|--------|----------------|---------------|
| R-Squared | 0.683 | 0.600 |
| RMSE      | 0.804 | 0.984 |

## Requirements

In order to run the python script and notebook, you will need to have the following packages installed:

* matplotlib
* numpy
* pandas
* scikit-learn

## Related Paper:

M. Cassotti, D. Ballabio, R. Todeschini, V. Consonni. A similarity-based QSAR model for predicting acute toxicity towards the fathead minnow (Pimephales promelas), SAR and QSAR in Environmental Research (2015), 26, 217-243; doi: 10.1080/1062936X.2015.1018938
