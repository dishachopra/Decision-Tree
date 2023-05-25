from sklearn.datasets import load_iris
import itertools
iris = load_iris()
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import plot_tree



st.write("""
Decision Tree Maker App
This app predicts makes a decision tree and predicts decision boundary!
""")

st.sidebar.header('User Input Parameters')
dataset= st.sidebar.selectbox('Choose the dataset', ['Iris', 'Breast Cancer', 'Make Moon', 'Make Circles', 'Make Blobs'])
if dataset == 'Iris':
    criterion= st.sidebar.selectbox('Choose the criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = st.sidebar.number_input('Maximum Depth Of Tree', min_value=1, value=3)




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
    

elif dataset=='Breast Cancer':
    criterion= st.sidebar.selectbox('Choose the criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = st.sidebar.number_input('Maximum Depth Of Tree', min_value=1, value=3)
    feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']

# Get the selected feature names from the multiselect widget
    selected_features = st.multiselect(
    'Select any four features for Decision Boundary',
    feature_names,
    ['mean radius', 'mean texture', 'mean perimeter', 'mean area'],
    key="features"
)

# Convert feature names to indices
    selected_indices = [feature_names.index(feature) for feature in selected_features]

# Generate pairs of selected indices
    pairs = list(itertools.combinations(selected_indices, 2))
    


    from sklearn.datasets import load_breast_cancer
    

    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)

    clf .fit(X,y)

    fig = plt.figure(figsize=(10, 8))
    _ = tree.plot_tree(clf, feature_names=breast_cancer.feature_names, class_names=breast_cancer.target_names, filled=True)

    st.pyplot(fig)

    n_classes = 2
    plot_colors = "ry"
    plot_step = 0.02
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for pairidx, pair in enumerate(pairs):
    # We only take the two corresponding features
        X = breast_cancer.data[:, pair]
        y = breast_cancer.target

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
            xlabel=breast_cancer.feature_names[pair[0]],
            ylabel=breast_cancer.feature_names[pair[1]],
             )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax.scatter(
        X[idx, 0],
        X[idx, 1],
        c=color,
        label=breast_cancer.target_names[i],
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

if dataset == 'Make Moon':
    criterion= st.sidebar.selectbox('Choose the criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = st.sidebar.number_input('Maximum Depth Of Tree', min_value=1, value=3)

# Generate the make_moons dataset
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Create a decision tree classifier
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)
    clf.fit(X, y)


    fig = plt.figure(figsize=(10, 8))
    _ = tree.plot_tree(clf, filled=True, feature_names=["x1", "x2"], class_names=["Class 0", "Class 1"])

# Show the plot
    st.pyplot(fig)

    n_classes = 2
    plot_colors = "ry"
    plot_step = 0.02
    
    fig, ax = plt.subplots(figsize=(12, 8))

    

    # Train
    clf = DecisionTreeClassifier().fit(X, y)
    DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
        
             )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax.scatter(
        X[idx, 0],
        X[idx, 1],
        c=color,
        cmap=plt.cm.RdYlBu,
        edgecolor="black",
        s=15,
            )

    dy = plt.suptitle("Decision surface of decision trees")

# Add the legend separately
    leg = plt.legend(loc="lower right", borderpad=0, handletextpad=0)

# Adjust the plot axes
    _ = plt.axis("tight")

# Display the plot
    st.pyplot(fig)


if dataset == 'Make Circles':
    criterion= st.sidebar.selectbox('Choose the criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = st.sidebar.number_input('Maximum Depth Of Tree', min_value=1, value=3)

# Generate the make_moons dataset
    X, y = make_circles(n_samples=100, noise=0.1, random_state=42)

# Create a decision tree classifier
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)
    clf.fit(X, y)


    fig = plt.figure(figsize=(10, 8))
    _ = tree.plot_tree(clf, filled=True, feature_names=["x1", "x2"], class_names=["Class 0", "Class 1"])

# Show the plot
    st.pyplot(fig)

    n_classes = 2
    plot_colors = "ry"
    plot_step = 0.02
    
    fig, ax = plt.subplots(figsize=(12, 8))

    

    # Train
    clf = DecisionTreeClassifier().fit(X, y)
    DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
        
             )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax.scatter(
        X[idx, 0],
        X[idx, 1],
        c=color,
        cmap=plt.cm.RdYlBu,
        edgecolor="black",
        s=15,
            )

    dy = plt.suptitle("Decision surface of decision trees")

# Add the legend separately
    leg = plt.legend(loc="lower right", borderpad=0, handletextpad=0)

# Adjust the plot axes
    _ = plt.axis("tight")

# Display the plot
    st.pyplot(fig)




if dataset == 'Make Blobs':
    criterion= st.sidebar.selectbox('Choose the criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = st.sidebar.number_input('Maximum Depth Of Tree', min_value=1, value=3)
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Train a decision tree classifier
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    clf.fit(X, y)

# Plot the decision tree
    
    fig = plt.figure(figsize=(10, 8))
    _ = plot_tree(clf, filled=True, feature_names=["x1", "x2"], class_names=["Class 0", "Class 1"])

# Show the plot
    st.pyplot(fig)

# Plot the decision boundary
    fig = plt.figure(figsize=(10, 8))
    plot_decision_regions(X, y, clf=clf, legend=2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

    st.pyplot(fig)


