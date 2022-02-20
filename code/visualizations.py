import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import seaborn as sns
# large = 28
# med = 20
# small = 18
# params = {'axes.titlesize': large,
#          'legend.fontsize': med,
#          'figure.figsize': (12, 8),
#          'axes.labelsize': med,
#          'xtick.labelsize': med,
#          'ytick.labelsize': med,
#          'figure.titlesize': large}
# plt.rcParams.update(params)
plt.style.use('seaborn-darkgrid')


def plot_feature_importances(model, X_train, model_title = 'This', n_features = 10, sort_features = True, size=(15,8), save_name=None):
    """
    Plots feature importances of a sklearn model save the plot if save_name is input

    Arg:
        model: a sklearn model
        X_train: a pd DataFrame containing the training data
        model_title: a string which will be used in the graph title as the name of the model
        n_features: an int number of features to include in the plot
        sort_features: a boolean determining whether to sort the features prior to graphing
        size: a tuple determining the size of the plot
        save_name: a str containing the file name and path to save the plot

    Return:
        Displays feature importance graph
        Stores graph in image folder if save_name is input
    """
    importances = model.feature_importances_
    labels = []
    for col in list(X_train.columns):
        label = col.replace('_', ' ').title()
        labels.append(label)
    importances_df = pd.Series(importances, labels)
    if sort_features:
        importances_df = importances_df.sort_values(ascending=False)
    plt.figure(figsize=size)
    importances_df.head(n_features).sort_values().plot.barh()
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importances for {} Model'.format(model_title))
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()

def plot_permutation_importance(model, X_test, y_test, model_title = 'This', n_features = 10, sort_features = True, size=(15,8), save_name=None):
    """
    Plots permuation importances of a sklearn model save the plot if save_name is input

    Arg:
        model: a sklearn model
        X_train: a pd DataFrame containing the training data
        model_title: a string which will be used in the graph title as the name of the model
        n_features: an int number of features to include in the plot
        sort_features: a boolean determining whether to sort the features prior to graphing
        size: a tuple determining the size of the plot
        save_name: a str containing the file name and path to save the plot

    Return:
        Displays feature importance graph
        Stores graph in image folder if save_name is input
    """
    result = permutation_importance(model, X_test, y_test)
    labels = []
    for col in list(X_test.columns):
        label = col.replace('_', ' ').title()
        labels.append(label)
    importances_df = pd.Series(result.importances_mean, labels)
    if sort_features:
        importances_df = importances_df.sort_values(ascending=False)
    plt.figure(figsize=size)
    importances_df.head(n_features).sort_values().plot.barh()
    plt.xlabel('Feature importance')
    plt.ylabel('Features')
    plt.title('Feature Importances for {} Model'.format(model_title))
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()

def create_heatmap(df, size=(15,8), save_name=None):
    """
    Produces a correlation heat map from a panda dataframe and saves it if save_name input

    Arg:
        df(pdDataFrame): a dataframe to plot
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays heat map plot
        Stores graph in image folder if save_name is input
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(
        data=corr,
        mask=np.triu(np.ones_like(corr, dtype=bool)),
        ax=ax,
        annot=True,
        cbar_kws={"label": "Correlation",
                  "orientation": "horizontal", "pad": .2, "extend": "both"}
    )
    ax.set_title(
        "Heatmap of Correlation Between Attributes (Including Target)")
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()


def linearity_graph(model, X_test, y_test, size=(15,8), save_name=None):
    """
    Produces a linearity test scatter plot from a panda dataframe and saves it if save_name input

    Arg:
        model: a Scikit learn OLS model constructed for the variables input
        X_test(pdDataFrame): a dataframe wih the independent variables to test the model on
        y_test(pdDataFrame): a dataframe wih the dependent variable to test the model on
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the linearity plot
        Stores graph in image folder if save_name is input
    """

    preds = model.predict(X_test)
    fig, ax = plt.subplots(figsize=size)
    perfect_line = np.arange(y_test.min(), y_test.max())
    ax.plot(perfect_line, perfect_line, linestyle="--",
            color="orange", label="Perfect Fit")
    ax.scatter(y_test, preds, alpha=0.5)
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.legend()
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()


def homoscedasticity_graph(model, X_test, y_test, size=(15,8), save_name=None):
    """
    Produces a homoscedasticity test scatter plot from a panda dataframe and saves it if save_name input

    Arg:
        model: a Scikit learn OLS model constructed for the variables input
        X_test(pdDataFrame): a dataframe wih the independent variables to test the model on
        y_test(pdDataFrame): a dataframe wih the dependent variable to test the model on
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the homoscedasticity plot
        Stores graph in image folder if save_name is input
    """
    fig, ax = plt.subplots(figsize=size)
    preds = model.predict(X_test)
    residuals = (y_test - preds)
    ax.scatter(preds, residuals, alpha=0.5)
    ax.plot(preds, [0 for i in range(len(X_test))],
            linestyle="--", color='orange', label="Perfect Fit")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Actual - Predicted Value")
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()


def one_hot_coef_graph(coef_df, categories, dropped_var, target_name='Price', increase_type='Percent', size=(15,8), rotate=None, fix_names = True, save_name=None):
    """
    Produces a bar graph of the coefficients from a panda dataframe and saves it if save_name input

    Arg:
        coef_df(pdDataFrame): a dataframe containing the coefficient to graph
        categories(array): a list of the category names whose coefficients you wish to graph
        dropped_var(str): the name of the variable droped during one-hot encoding
        target_name(str): the name of the target variable
        increase_type(str): the type of increase represented by the coefficient, either 'Percent' or 'Absolute'
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the bar graph of the listed categories coefficients
        Stores graph in image folder if save_name is input
    """
    coefs = []
    for cat in categories:
        coefs.append(float(coef_df[cat]))
    fig, ax = plt.subplots(figsize=size)
    x_labels = categories.copy()
    if fix_names:
        for i, cat in enumerate(categories):
            x_labels[i] = cat.replace('_', ' ').title()
    ax = plt.bar(x_labels, coefs)
    plt.ylabel("{x} Increase in {y}".format(x=increase_type, y=target_name))
    plt.title("Predicted {x} Increase in {y} Relative to {z}".format(
        x=increase_type, y=target_name, z=dropped_var))
    if rotate:
        plt.xticks(rotation = rotate)
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()

def make_network_confusion_matrices(model, X_train, y_train, X_test, y_test, labels, title, batch_size=32):
    """
    Plots confusion matrixes for a Keras classification mode

    Arg:
        model: a Keras classification model
        X_train: a Pandas dataframe containing the data used to train the model
        y_train: a numpy array containing the known target values the model was trained using
        X_test: a Pandas dataframe containing the data to evaluate the model on
        y_test: a numpy array containing the known target values for the test data
        labels: an array containing the original category names of the target values
        title: a string naming the model to be used in the plot titles
        batch_size: the batch size used in training the model
        save_name: a str containing the file name and path to save the plot

    Return:
        Displays a pair of confusion matrices, one for the training and the other for the test data
    """

    # Converting predictions into form for confusion matrices
    train_predictions = model.predict(X_train, batch_size=batch_size, verbose=0)
    rounded_train_predictions = np.argmax(train_predictions, axis=1)
    rounded_train_labels=np.argmax(y_train, axis=1)
    test_predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
    rounded_test_predictions = np.argmax(test_predictions, axis=1)
    rounded_test_labels=np.argmax(y_test, axis=1)

    #plotting training matrix
    train_cm = confusion_matrix(rounded_train_labels, rounded_train_predictions)
    disp = ConfusionMatrixDisplay(train_cm, display_labels=labels)
    disp.plot(cmap="Greens")
    plt.grid(False)
    plt.title('Training Matrix: {}'.format(title))
    plt.show()

    #plotting test matrix
    test_cm = confusion_matrix(rounded_test_labels, rounded_test_predictions)
    disp = ConfusionMatrixDisplay(test_cm, display_labels=labels)
    disp.plot(cmap="Greens")
    plt.grid(False)
    plt.title('Test Matrix: {}'.format(title))
    plt.show()

def training_graph(val_dict, title):
    """
    Plots a graph of the training and validation losses during the training process or a Keras model.

    Arg:
        val_dict: the validation dictionary produced by the Keras model during training
        title: a string naming the model to be used in the plot titles

    Return:
        Displays a graph of the training and validation losses vs the epochs of training
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    loss_values = val_dict['loss']
    val_loss_values = val_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)
    ax.plot(epochs, loss_values, label='Training loss')
    ax.plot(epochs, val_loss_values, label='Validation loss')

    ax.set_title('Training & Validation Loss: {}'.format(title))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    return plt.show()

def visualize_top_10(freq_dist, title, ylabel="Count"):
    """
    Plots a bar graph of the top 10 values in frequency dictionary.

    Arg:
        freq_dict: a dictionary containing keys and values that corespond to frequencies of the keys
        title: a string to use in the plot title

    Return:
        A bar graph of the top 10 values in the frequency dictionary
    """

    # Extract data for plotting
    top_10 = list(zip(*freq_dist.most_common(10)))
    tokens = top_10[0]
    counts = top_10[1]

    # Set up plot and plot data
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(tokens, counts)

    # Customize plot appearance
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", rotation=90)
    plt.show()
