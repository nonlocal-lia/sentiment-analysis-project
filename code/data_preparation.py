import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

def get_scores(model, X_test, y_test):

    """
    Gets accuracy, and weighted precision, recall and F1 scores of a Sklearn classification model using sklearn metrics

    Arg:
        model: a sklearn classification model
        X_test: a Pandas dataframe containing the data to evaluate the model on
        y_test: a numpy array containing the known target values for the test data

    Return:
        output: a dictionary with the model's scores on the 4 metrics
    """

    acc = accuracy_score(y_test, model.predict(X_test))
    prec = precision_score(y_test, model.predict(X_test), average='weighted')
    f1 = f1_score(y_test, model.predict(X_test), average='weighted')
    rec = recall_score(y_test, model.predict(X_test), average='weighted')
    output = {'Accuracy': acc, "Precision": prec, "Recall": rec, "F1 Score": f1}
    return output

def get_network_metrics(model, X_train, y_train, X_test, y_test, batch_size=32):

    """
    Gets accuracy, and weighted F1 scores of a Keras classification model for both training and test data
    Also determines the rarest category in the data and calculates the model's precision, recall, and F1
    on the rare category

    Arg:
        model: a sklearn classification model
        X_train: a Pandas dataframe containing the data used to train the model
        y_train: a numpy array containing the known target values the model was trained using
        X_test: a Pandas dataframe containing the data to evaluate the model on
        y_test: a numpy array containing the known target values for the test data

    Return:
        Prints scores
        output: a dictionary with the model's scores
    """
    
    # Transforming predictions into form for evaluation with Sklearn functions
    train_predictions = model.predict(X_train, batch_size=batch_size, verbose=0)
    rounded_train_predictions = np.argmax(train_predictions, axis=1)
    rounded_train_labels=np.argmax(y_train, axis=1)
    test_predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
    rounded_test_predictions = np.argmax(test_predictions, axis=1)
    rounded_test_labels=np.argmax(y_test, axis=1)

    # Evaluating Training
    results_train = model.evaluate(X_train, y_train)
    print('----------')
    print(f'Training Loss: {results_train[0]:.3} \nTraining Accuracy: {results_train[1]:.3}')
    train_f1 = f1_score(rounded_train_labels, rounded_train_predictions, average='weighted')
    print(f'Train Average Weighted F1 Score: {train_f1:.3}')
    train_cm = confusion_matrix(rounded_train_labels, rounded_train_predictions)
    length_train = len(rounded_train_labels)
    for i in range(train_cm.shape[0]):
        count = np.count_nonzero(rounded_train_labels == i)
        if count < length_train:
            length_train = count
            index_value = i
    recall_rare_train = train_cm[index_value, index_value]/np.sum(train_cm[0])
    print(f'Train Recall on Rarest Category: {recall_rare_train:.3}')
    precision_rare_train = train_cm[index_value, index_value]/np.sum(train_cm[:,0])
    print(f'Train Precision on Rarest Category: {precision_rare_train:.3}')
    f1_rare_train = 2*(recall_rare_train*precision_rare_train)/(recall_rare_train+precision_rare_train)
    print(f'Train F1 on Rarest Category: {f1_rare_train:.3}')

    # Evaluating Testing
    results_test = model.evaluate(X_test, y_test)
    print('----------')
    print(f'Test Loss: {results_test[0]:.3} \nTest Accuracy: {results_test[1]:.3}')
    test_f1 = f1_score(rounded_test_labels, rounded_test_predictions, average='weighted')
    print(f'Test Average Weighted F1 Score: {test_f1:.3}')
    test_cm = confusion_matrix(rounded_test_labels, rounded_test_predictions)
    length_test = len(rounded_test_labels)
    for i in range(test_cm.shape[0]):
        count = np.count_nonzero(rounded_test_labels == i)
        if count < length_test:
            length_test = count
            index_value = i
    recall_rare_test = test_cm[index_value, index_value]/np.sum(test_cm[0])
    print(f'Test Recall on Rarest Category: {recall_rare_test:.3}')
    precision_rare_test = test_cm[index_value, index_value]/np.sum(test_cm[:,0])
    print(f'Test Precision on Rarest Category: {precision_rare_test:.3}')
    f1_rare_test = 2*(recall_rare_test*precision_rare_test)/(recall_rare_test+precision_rare_test)
    print(f'Test F1 on Rarest Category: {f1_rare_test:.3}')

    output = {"Training Accuracy": results_train[1],
            "Test Accuracy": results_test[1],
            "Training F1": train_f1,
            "Test F1": test_f1,
            "Training Rare Recall": recall_rare_train,
            "Test Rare Recall": recall_rare_test,
            "Training Rare Precision": precision_rare_train,
            "Test Rare Precision": precision_rare_test,
            "Training Rare F1": f1_rare_train,
            "Test Rare F1": f1_rare_test
            }
    return output
