# EnsembleRobustSVM
Code related to the manuscript "Ensemble Methods for Robust Support Vector Machines using Integer Programming" by J. Kurtz (http://www.optimization-online.org/DB_HTML/2022/02/8797.html)

In the following you find a description of the related Python-files:

MainSVM.py:
Runs computations where the following parameters can be adjusted:
- epsDefend: defense-level r_d
- normDefend: norm for defense during training; 1: l1-norm, 2: l2-norm, 3: max-norm
- normAttack: norm for attacking the test-set; 1: l1-norm, 2: l2-norm, 3: max-norm
- attackType: worstcase: adversarial calculates worstcase attack for test-data-points
- scaler: "Standardize": standardize normalization; "MinMax": min-max normalization
- numTrainTestSplits: number of average train-test-splits
- k_range: values for k which will be used
- model: "SVMEnsemble": SVM-Ens, "ROSVM": RO-SVM, "MROSVM": Ensemble method in Algorithm 1
- worstCaseType: "Ens-E", "Ens-R", "Ens-H" (see paper)
- dataset: see list in code below
- fixed_digit: digit which gets class label 1 in Digits dataset

All calculations are performed for attack-levels [0.0,0.1,0.2,0.3,0.4,0.5,0.75,1.0,1.25,1.5,1.75,2.0].
All results are written to the file in variable fileName.


Parameters.py:
Contains the following global parameters: 
- fracTest: fractional amount of data which is used for the test-set
- timeLimit: timelimit for each Gurobi optimize() call
- MIPGap: MIPGap for Gurobi optimizer
- eps = accuracy parameter for calculations

Functions.py:
- trainSVM(X,y): trains a classical SVM on dataset X with labels y
- trainSVMEnsemble(X,y,k): trains SVM ensemble (bagging) on dataset X with labels y and k different models
- trainRobustSVM(X,y,norm, r): trains RO-SVM on X with labels y, defense-norm norm and defense-radius r
- predictRobustSVM(w,b,X): predicts labels for test-set X with given SVM solution w,b
- iterativeHeuristic(X,y,norm, r,k, worstCaseType): runs Algorithm 1 (iterative robust ensemble method). worstCaseType defines the adversarial problem which is used
- getWorstCaseAttackNumHyperplanes(x,y,w,b,k,r,norm): runs the exact adversarial problem (used in Ens-E)
- getOptimalDeltaAverage(x,y,w,b,k,r,norm): runs the relaxed adversarial problem (used in Ens-R)
- getWorstCaseAttackHeuristic(x,y,w,b,k,r,norm): runs the heuristic adversarial problem (used in Ens-H)
- predictMultiRobustSVM(w,b,X): predicts label for multi-hyperplanes models

