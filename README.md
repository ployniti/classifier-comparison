# Classifier Comparison
This project sets out to explore correlations, if any, between different attributes of secondary school students and their performance in math and Portuguese language courses as represented by their final grades in the courses. 

As part of the project, I attempted to train and tune Multilayer Perceptron (MLP)/Neural Network (NN), using PyTorch, and Support Vector Machine (SVM), using Scikit Learn, on the [student dataset](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption/code?datasetId=251&sortBy=voteCount) in order to predict their final grades and conducted a comparison of the two models’ performance.

## Usage
Both SVM and MLP models can be run through the command line interface with different parameters.

### Flags
`--plot` plot accuracy and/or confusion matrix from the model.

`--onehotall` run the model using the one-hot preprocessing for all columns.

`--ts` configure the test size to partition the dataset for training and testing.

### Multilayer Perceptron/Neural Network
To run the final optimized model:
```
python3 final.py --onehotall nn
```

Additional options:

`-hidden` set the number of hidden nodes.

`-lr` set the learning rate of the model.

`-e` set the number of epochs to run the training.

`-bs` set the batch size for training.

### Support Vector Machine
To run the final optimized model:
```
python3 final.py --onehotall svm -k rbf
```

Additional options:

`-k` set the kernel functions.

`-d` set degree for `poly` kernel function.
