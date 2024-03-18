import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sb

### GOALS ###
# Train NN and SVM on student-mat and student-por data to predict student's 
# grade in math and portugese. Compare models' performance.
#############

# Import student-mat.csv and student-por.csv
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
student_mat = pd.read_csv("./studentAlcCon/student-mat.csv")
student_por = pd.read_csv("./studentAlcCon/student-por.csv")

# Add a subject column student_mat and student_por before combining
student_mat["subject"] = 1
student_por["subject"] = 0

# Combine student_mat and student_por
student_data = pd.concat([student_mat, student_por], ignore_index=True) # (1044, 34)

### DATA PREPROCESSING ###
def data_preprocess(one_hot_all, ts, student_data=student_data):
    if (one_hot_all):
        # One hot encode all data columns
        encoded_student_data = pd.get_dummies(student_data, 
                               columns=["age", "Medu", "Fedu", "Mjob", "Fjob", "reason", 
                               "guardian", "traveltime", "failures", "studytime",
                               "famrel", "freetime", "goout", "Dalc", "Walc", "health",
                               "absences", "G1", "G2"], dtype=int)
    else:
        # One hot encode categorical columns
        # https://www.kdnuggets.com/2023/07/pandas-onehot-encode-data.html
        # https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
        encoded_student_data = pd.get_dummies(student_data, 
                               columns=["Mjob", "Fjob", "reason", "guardian"], dtype=int)

        # Normalize numeric columns by dividing values in the column by the largest value of the column
        encoded_student_data["age"] = encoded_student_data["age"]/encoded_student_data["age"].max()
        encoded_student_data["Medu"] = encoded_student_data["Medu"]/encoded_student_data["Medu"].max()
        encoded_student_data["Fedu"] = encoded_student_data["Fedu"]/encoded_student_data["Fedu"].max()
        encoded_student_data["traveltime"] = encoded_student_data["traveltime"]/encoded_student_data["traveltime"].max()
        encoded_student_data["failures"] = encoded_student_data["failures"]/encoded_student_data["failures"].max()
        encoded_student_data["studytime"] = encoded_student_data["studytime"]/encoded_student_data["studytime"].max()
        encoded_student_data["famrel"] = encoded_student_data["famrel"]/encoded_student_data["famrel"].max()
        encoded_student_data["freetime"] = encoded_student_data["freetime"]/encoded_student_data["freetime"].max()
        encoded_student_data["goout"] = encoded_student_data["goout"]/encoded_student_data["goout"].max()
        encoded_student_data["Dalc"] = encoded_student_data["Dalc"]/encoded_student_data["Dalc"].max()
        encoded_student_data["Walc"] = encoded_student_data["Walc"]/encoded_student_data["Walc"].max()
        encoded_student_data["health"] = encoded_student_data["health"]/encoded_student_data["health"].max()
        encoded_student_data["absences"] = encoded_student_data["absences"]/encoded_student_data["absences"].max()
        encoded_student_data["G1"] = encoded_student_data["G1"]/encoded_student_data["G1"].max()
        encoded_student_data["G2"] = encoded_student_data["G2"]/encoded_student_data["G2"].max()

    # Convert binary columns to 0 and 1
    # https://stackoverflow.com/questions/40901770/is-there-a-simple-way-to-change-a-column-of-yes-no-to-1-0-in-a-pandas-dataframe
    encoded_student_data["school"] = encoded_student_data["school"].map(dict(GP=1, MS=0))
    encoded_student_data["sex"] = encoded_student_data["sex"].map(dict(F=1, M=0))
    encoded_student_data["address"] = encoded_student_data["address"].map(dict(U=1, R=0))
    encoded_student_data["famsize"] = encoded_student_data["famsize"].map(dict(GT3=1, LE3=0))
    encoded_student_data["Pstatus"] = encoded_student_data["Pstatus"].map(dict(T=1, A=0))
    encoded_student_data["schoolsup"] = encoded_student_data["schoolsup"].map(dict(yes=1, no=0))
    encoded_student_data["famsup"] = encoded_student_data["famsup"].map(dict(yes=1, no=0))
    encoded_student_data["paid"] = encoded_student_data["paid"].map(dict(yes=1, no=0))
    encoded_student_data["activities"] = encoded_student_data["activities"].map(dict(yes=1, no=0))
    encoded_student_data["nursery"] = encoded_student_data["nursery"].map(dict(yes=1, no=0))
    encoded_student_data["higher"] = encoded_student_data["higher"].map(dict(yes=1, no=0))
    encoded_student_data["internet"] = encoded_student_data["internet"].map(dict(yes=1, no=0))
    encoded_student_data["romantic"] = encoded_student_data["romantic"].map(dict(yes=1, no=0))

    # https://stackoverflow.com/questions/67692245/applying-a-function-to-all-but-one-column-in-pandas
    # data_cols = encoded_student_data.columns.difference(["G3"])

    # Try mean centering data
    # https://www.statology.org/center-data-in-python/
    # encoded_student_data[data_cols] = encoded_student_data[data_cols].apply(lambda x: x-x.mean())

    # print(encoded_student_data)
    # print(encoded_student_data.mean())

    # Try min max data
    # encoded_student_data[data_cols] = encoded_student_data[data_cols].apply(lambda x: (x-x.min())/(x.max()-x.min()))

    # Move G3, the final grade target from 0 to 20, to the end of the dataframe
    G3_col = encoded_student_data.pop("G3")
    encoded_student_data["G3"] = G3_col

    # Convert df to np array
    encoded_student_data_np = encoded_student_data.to_numpy()
    dcols = encoded_student_data_np.shape[1]

    # Delete rows with G3 == 1, 4, and 20 from student data 
    # because there is only one instance of each to do train_test_split
    encoded_student_data_np = np.delete(encoded_student_data_np, 
                            np.asarray(encoded_student_data_np[:,-1] == 1).nonzero(), axis=0)
    encoded_student_data_np = np.delete(encoded_student_data_np, 
                            np.asarray(encoded_student_data_np[:,-1] == 4).nonzero(), axis=0)
    encoded_student_data_np = np.delete(encoded_student_data_np, 
                            np.asarray(encoded_student_data_np[:,-1] == 20).nonzero(), axis=0)

    # Split training and testing data and target with even distribution
    train_input, test_input, train_grade, test_grade = train_test_split(encoded_student_data_np, 
                                                        encoded_student_data_np[:,-1],
                                                        test_size=ts, 
                                                        random_state=0,
                                                        shuffle=True,
                                                        stratify=encoded_student_data_np[:,-1])

    # Remove G3 grade col from training and testing data
    train_input = train_input[:,:(dcols-1)]
    test_input = test_input[:,:(dcols-1)]

    # Convert train_grade and test_grade elements to int
    train_grade = train_grade.astype(int)
    test_grade = test_grade.astype(int)

    return train_input, test_input, train_grade, test_grade, dcols

### NEURAL NETWORK ###
def neural_network(plot, one_hot_all, ts, h, lr, e, bs):
    # Preprocess data
    train_input, test_input, train_grade, test_grade, dcols = data_preprocess(one_hot_all, ts)

    # Set output and hidden layer nodes
    output_nodes = 21
    hidden_nodes = h

    # Create tensor out of train np array
    train_input_tensor = torch.tensor(train_input, dtype=torch.float32)

    # One hot encode the train target
    train_target_matrix = np.full((len(train_grade), output_nodes), 0.1)
    train_target_matrix[np.arange(len(train_grade)), train_grade] = 0.9
    train_target_matrix_tensor = torch.tensor(train_target_matrix, dtype=torch.float32)

    # Create tensor out of test np array
    test_input_tensor = torch.tensor(test_input, dtype=torch.float32)

    # One hot encode the test target
    test_target_matrix = np.full((len(test_grade), output_nodes), 0.1)
    test_target_matrix[np.arange(len(test_grade)), test_grade] = 0.9
    test_target_matrix_tensor = torch.tensor(test_target_matrix, dtype=torch.float32)

    # Define NN model
    nn_model = nn.Sequential(
        nn.Linear((dcols-1), hidden_nodes),
        nn.Sigmoid(),
        nn.Linear(hidden_nodes, hidden_nodes),
        nn.Sigmoid(),
        nn.Linear(hidden_nodes, output_nodes),
        nn.Sigmoid()
    )

    # Define loss function (mean square error)
    # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    loss_func = nn.MSELoss()

    # Define optimizer
    # https://pytorch.org/docs/stable/optim.html
    optimizer = optim.Adam(nn_model.parameters(), lr=lr)

    # Train model
    epochs = e
    batch_size = bs
    train_accuracy_list = []
    test_accuracy_list = []
    loss_list = []
    latest_test_pred_argmax = []
    prev_test_acc = 0.0

    for epoch in tqdm(range(epochs)):
        for i in range(0, len(train_input_tensor), batch_size):
            input_batch = train_input_tensor[i:i+batch_size]
            grade_pred = nn_model(input_batch)
            grade_batch = train_target_matrix_tensor[i:i+batch_size]
            loss = loss_func(grade_pred, grade_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss)
        print(f"\nFinished epoch {epoch}, latest loss: {loss}")
        
        # Train accuracy
        train_pred = nn_model(train_input_tensor)
        train_accuracy = (torch.eq(torch.argmax(train_pred, dim=1),
                        torch.argmax(train_target_matrix_tensor, dim=1)).sum())/len(train_pred)
        train_accuracy_list.append(train_accuracy*100)
        print(f"Training set accuracy: {train_accuracy*100}")

        # Test accuracy
        test_pred = nn_model(test_input_tensor)
        test_pred_argmax = torch.argmax(test_pred, dim=1)
        test_accuracy = (torch.eq(test_pred_argmax,
                        torch.argmax(test_target_matrix_tensor, dim=1)).sum())/len(test_pred)
        prev_test_acc = test_accuracy
        latest_test_pred_argmax = test_pred_argmax.tolist()
        test_accuracy_list.append(test_accuracy*100)
        print(f"Testing set accuracy: {test_accuracy*100}")

        if (epoch > 0):
            # Check for early stopping after first epoch
            if (abs(prev_test_acc - test_accuracy) < 1e-6):
                break

    if (plot):
        # Plot accuracy for test and training data
        fig, ax = plt.subplots()
        x = np.arange(0, len(test_accuracy_list), 1)
        ax.plot(x, train_accuracy_list)
        ax.plot(x, test_accuracy_list)
        ax.set(xlabel='Epoch', ylabel='Accuracy (%)', title='MLP Predicting Student Grade')
        ax.grid()
        ax.legend(['Accuracy on training data', 'Accuracy on test data'])
        fig.savefig("1nn.png")

        # Clear previous plot
        plt.clf()

        # Plot confusion matrix for test data after training
        cm = metrics.confusion_matrix(test_grade, latest_test_pred_argmax)

        # https://seaborn.pydata.org/generated/seaborn.heatmap.html
        sb.heatmap(cm, annot=True, fmt='g')

        plt.title('Confusion Matrix of MLP')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('1nn_cm.png')

### SVM ###
def support_vector_machine(plot, one_hot_all, ts, kernel, degree):
    # Preprocess data
    train_input, test_input, train_grade, test_grade, dcols = data_preprocess(one_hot_all, ts)

    # Define SVM classifier
    svm_clf = svm.SVC(C=2.0, kernel=kernel)

    if (kernel == "poly"):
        svm_clf = svm.SVC(C=2.0, kernel=kernel, degree=degree)

    # Train the model using train_input
    svm_clf.fit(train_input, train_grade)

    # Prediction on train_input and test_input
    train_pred_svm = svm_clf.predict(train_input)
    test_pred_svm = svm_clf.predict(test_input)
    
    if (plot):
        # Plot confusion matrix for test data
        cm = metrics.confusion_matrix(test_grade, test_pred_svm)

        sb.heatmap(cm, annot=True, fmt='g')

        plt.title('Confusion Matrix of SVM')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('1svm_cm.png')

    # Evaluate accuracy, precision, and recall on train and test sets
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    train_accuracy_svm = metrics.accuracy_score(train_grade, train_pred_svm)
    # train_precision_svm = metrics.precision_score(train_grade, train_pred_svm, average=None)
    # train_recall_svm = metrics.recall_score(train_grade, train_pred_svm, average=None)

    print(f"Training set accuracy (SVM): {train_accuracy_svm*100}")
    # print(f"Training set precision (SVM): {train_precision_svm*100}")
    # print(f"Training set recall (SVM): {train_recall_svm*100}")

    test_accuracy_svm = metrics.accuracy_score(test_grade, test_pred_svm)
    # test_precision_svm = metrics.precision_score(test_grade, test_pred_svm, average=None)
    # test_recall_svm = metrics.recall_score(test_grade, test_pred_svm, average=None)

    print(f"Testing set accuracy (SVM): {test_accuracy_svm*100}")
    # print(f"Testing set precision (SVM): {test_precision_svm*100}")
    # print(f"Testing set recall (SVM): {test_recall_svm*100}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command line options for running models')
    parser.add_argument('--onehotall', action='store_true', 
                        help='One hot all data during preprocessing')
    parser.add_argument('--plot', action='store_true', 
                        help='Plot the accuracy and/or confusion matrix of model')
    parser.add_argument('-ts', metavar='ts', 
                        type=float, default=0.2, help='Test split ratio')

    # https://gist.github.com/amarao/36327a6f77b86b90c2bca72ba03c9d3a
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_nn = subparsers.add_parser('nn', help='Run NN')
    parser_nn.add_argument('-hidden', metavar='hidden', 
                        type=int, default=25, help='Number of hidden nodes in NN')
    parser_nn.add_argument('-lr', metavar='lr',
                        type=float, default=0.01, help='Learning rate of NN')
    parser_nn.add_argument('-e', metavar='e', 
                        type=int, default=100, help='Number of epochs to run NN')
    parser_nn.add_argument('-bs', metavar='bs',
                        type=int, default=5, help='Batch size to run NN')

    parser_svm = subparsers.add_parser('svm', help='Run SVM') 
    parser_svm.add_argument('-k', metavar='k',
                        type=str, required=True, help='Kernel function to use in SVM')
    parser_svm.add_argument('-d', metavar='d',
                        type=int, default=None, help='Degree of poly kernel function')

    args = parser.parse_args()

    if (args.subcommand == "nn"):
        neural_network(args.plot, args.onehotall, args.ts, args.hidden, args.lr, args.e, args.bs)
    elif (args.subcommand == "svm"):
        support_vector_machine(args.plot, args.onehotall, args.ts, args.k, args.d)