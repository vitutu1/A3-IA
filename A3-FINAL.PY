import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    dataset = "wdbc.data"
    data = pd.read_csv(dataset, header=None)
    data.columns = ['ID', 'target'] + [f'feature_{i}' for i in range(1, 31)]
    return data

def split_data(data):
    X = data.iloc[:, 2:]
    y = data.iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train, **kwargs):
    model = DecisionTreeClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train, **kwargs):
    model = MLPClassifier(max_iter=1000, random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, confusion, report

def run_decision_tree_experiments():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    results = []
    
    tree_model = train_decision_tree(X_train, y_train)
    acc_tree, conf_matrix_tree, report_tree = evaluate_model(tree_model, X_test, y_test)
    results.append(("Árvore de Decisão Padrão", acc_tree, conf_matrix_tree, report_tree))
    
    tree_model = train_decision_tree(X_train, y_train, max_depth=3)
    acc_tree, conf_matrix_tree, report_tree = evaluate_model(tree_model, X_test, y_test)
    results.append(("Árvore de Decisão Profundidade Limitada", acc_tree, conf_matrix_tree, report_tree))
    
    tree_model = train_decision_tree(X_train, y_train, criterion='entropy')
    acc_tree, conf_matrix_tree, report_tree = evaluate_model(tree_model, X_test, y_test)
    results.append(("Árvore de Decisão Entropia", acc_tree, conf_matrix_tree, report_tree))
    
    tree_model = train_decision_tree(X_train, y_train, min_samples_leaf=5)
    acc_tree, conf_matrix_tree, report_tree = evaluate_model(tree_model, X_test, y_test)
    results.append(("Árvore de Decisão Mínimo Amostras por Folha", acc_tree, conf_matrix_tree, report_tree))
    
    tree_model = train_decision_tree(X_train, y_train, max_features=10)
    acc_tree, conf_matrix_tree, report_tree = evaluate_model(tree_model, X_test, y_test)
    results.append(("Árvore de Decisão Máximo de Features", acc_tree, conf_matrix_tree, report_tree))
    
    return results

def run_neural_network_experiments():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    results = []
    
    nn_model = train_neural_network(X_train, y_train)
    acc_nn, conf_matrix_nn, report_nn = evaluate_model(nn_model, X_test, y_test)
    results.append(("Rede Neural Simples", acc_nn, conf_matrix_nn, report_nn))
    
    nn_model = train_neural_network(X_train, y_train, hidden_layer_sizes=(100, 50))
    acc_nn, conf_matrix_nn, report_nn = evaluate_model(nn_model, X_test, y_test)
    results.append(("Rede Neural 2 Camadas Ocultas", acc_nn, conf_matrix_nn, report_nn))
    
    nn_model = train_neural_network(X_train, y_train, hidden_layer_sizes=(100, 50, 25), activation='tanh')
    acc_nn, conf_matrix_nn, report_nn = evaluate_model(nn_model, X_test, y_test)
    results.append(("Rede Neural Mais Camadas Tanh", acc_nn, conf_matrix_nn, report_nn))
    
    nn_model = train_neural_network(X_train, y_train, alpha=0.01)
    acc_nn, conf_matrix_nn, report_nn = evaluate_model(nn_model, X_test, y_test)
    results.append(("Rede Neural Alpha Regularização", acc_nn, conf_matrix_nn, report_nn))
    
    nn_model = train_neural_network(X_train, y_train, solver='lbfgs')
    acc_nn, conf_matrix_nn, report_nn = evaluate_model(nn_model, X_test, y_test)
    results.append(("Rede Neural Solver lbfgs", acc_nn, conf_matrix_nn, report_nn))
    
    return results

def print_results(results, model_type):
    st.subheader(f"Resultados dos experimentos com {model_type}")
    for i, (name, accuracy, conf_matrix, report) in enumerate(results):
        st.write(f"{model_type} Experimento {i+1}: {name}")
        st.write(f"Acurácia: {accuracy:.2f}")
        st.write("Matriz de Confusão:")
        st.write(conf_matrix)
        st.write("Relatório de Classificação:")
        st.write(report)
        st.write("---")

def plot_accuracy(results, model_type):
    accuracies = [result[1] for result in results]
    labels = [result[0] for result in results]
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, accuracies, color=colors)
    ax.set_xlabel('Acurácia')
    ax.set_ylabel('Experimentos')
    ax.set_title(f'Acurácia dos experimentos com {model_type}')
    ax.set_xlim(0.90, 1.0)
    st.pyplot(fig)

def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão para {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot()

if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Análise de Modelos de Machine Learning")
    
    st.sidebar.title("Configurações")
    st.sidebar.markdown("Selecione o modelo para visualizar os resultados:")
    model_type = st.sidebar.radio("Modelo", ("Árvore de Decisão", "Rede Neural"))
    
    if model_type == "Árvore de Decisão":
        decision_tree_results = run_decision_tree_experiments()
        st.subheader("Resultados com Árvore de Decisão")
        print_results(decision_tree_results, "Árvore de Decisão")
        plot_accuracy(decision_tree_results, "Árvore de Decisão")
        for result in decision_tree_results:
            plot_confusion_matrix(result[2], result[0])
    
    elif model_type == "Rede Neural":
        neural_network_results = run_neural_network_experiments()
        st.subheader("Resultados com Rede Neural")
        print_results(neural_network_results, "Rede Neural")
        plot_accuracy(neural_network_results, "Rede Neural")
        for result in neural_network_results:
            plot_confusion_matrix(result[2], result[0])
