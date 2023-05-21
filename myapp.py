from sklearn.datasets import load_iris

iris = load_iris()
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay



st.write("""
Decision Tree Maker App
This app predicts makes a decision tree and predicts decision boundary!
""")

st.sidebar.header('User Input Parameters')
dataset= st.sidebar.selectbox('Choose the dataset', ['Iris', 'Breast Cancer'])
if dataset == 'Iris':
    criterion= st.sidebar.selectbox('Choose the criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = st.sidebar.number_input('Maximum Depth Of Tree', min_value=1, value=3)

    st.subheader('User Input parameters')


    iris=datasets.load_iris()
    X=iris.data
    Y=iris.target

    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)

    clf .fit(X,Y)


    fig = plt.figure(figsize=(10, 8))
    _ = tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)

    st.pyplot(fig)

    n_classes = 3
    plot_colors = "ryb"
    plot_step = 0.02

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

    # Train
        clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
        ax = axes[pairidx // 3, pairidx % 3]
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
            xlabel=iris.feature_names[pair[0]],
            ylabel=iris.feature_names[pair[1]],
             )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax.scatter(
        X[idx, 0],
        X[idx, 1],
        c=color,
        label=iris.target_names[i],
        cmap=plt.cm.RdYlBu,
        edgecolor="black",
        s=15,
            )

    dy = plt.suptitle("Decision surface of decision trees trained on pairs of features")

# Add the legend separately
    leg = plt.legend(loc="lower right", borderpad=0, handletextpad=0)

# Adjust the plot axes
    _ = plt.axis("tight")

# Display the plot
    st.pyplot(fig)

# Display the title and legend separately
    st.write(dy)

elif dataset=='Breast Cancer':
    criterion= st.sidebar.selectbox('Choose the criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = st.sidebar.number_input('Maximum Depth Of Tree', min_value=1, value=3)

    st.subheader('User Input parameters')


    from sklearn.datasets import load_breast_cancer

    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)

    clf .fit(X,y)

    fig = plt.figure(figsize=(10, 8))
    _ = tree.plot_tree(clf, feature_names=breast_cancer.feature_names, class_names=breast_cancer.target_names, filled=True)

    st.pyplot(fig)


else:
    st.write('No dataset selected')
