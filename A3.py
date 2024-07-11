import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def carregar_dados():
    arquivo_dataset = "breast+cancer+wisconsin+diagnostic/wdbc.data"
    dados = pd.read_csv(arquivo_dataset, header=None)
    dados.columns = ['ID', 'diagnostico'] + [f'caracteristica_{i}' for i in range(1, 31)]
    dados['diagnostico'] = dados['diagnostico'].map({'M': 1, 'B': 0})
    return dados


def dividir_dados(dados):
    X = dados.iloc[:, 2:]
    y = dados.iloc[:, 1]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
    normalizador = StandardScaler()
    X_treino = normalizador.fit_transform(X_treino)
    X_teste = normalizador.transform(X_teste)
    return X_treino, X_teste, y_treino, y_teste


def treinar_arvore_decisao(X_treino, y_treino, **kwargs):
    modelo = DecisionTreeClassifier(random_state=42, **kwargs)
    modelo.fit(X_treino, y_treino)
    return modelo


def treinar_rede_neural(X_treino, y_treino, **kwargs):
    modelo = MLPClassifier(max_iter=1000, random_state=42, **kwargs)
    modelo.fit(X_treino, y_treino)
    return modelo


def avaliar_modelo(modelo, X_teste, y_teste):
    previsoes = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsoes)
    matriz_confusao = confusion_matrix(y_teste, previsoes)
    relatorio_classificacao = classification_report(y_teste, previsoes)
    return acuracia, matriz_confusao, relatorio_classificacao


def executar_experimentos_arvore_decisao():
    dados = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = dividir_dados(dados)
    resultados = []
    
    modelo_arvore = treinar_arvore_decisao(X_treino, y_treino)
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_arvore, X_teste, y_teste)
    resultados.append(("Árvore de Decisão Padrão", acuracia, matriz_confusao, relatorio))
    
    modelo_arvore = treinar_arvore_decisao(X_treino, y_treino, max_depth=3)
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_arvore, X_teste, y_teste)
    resultados.append(("Árvore de Decisão Profundidade Limitada", acuracia, matriz_confusao, relatorio))
    
    modelo_arvore = treinar_arvore_decisao(X_treino, y_treino, criterion='entropy')
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_arvore, X_teste, y_teste)
    resultados.append(("Árvore de Decisão Entropia", acuracia, matriz_confusao, relatorio))
    
    modelo_arvore = treinar_arvore_decisao(X_treino, y_treino, min_samples_leaf=10)
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_arvore, X_teste, y_teste)
    resultados.append(("Árvore de Decisão Mínimo Amostras por Folha", acuracia, matriz_confusao, relatorio))
    
    modelo_arvore = treinar_arvore_decisao(X_treino, y_treino, max_features=10)
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_arvore, X_teste, y_teste)
    resultados.append(("Árvore de Decisão Máximo de Features", acuracia, matriz_confusao, relatorio))
    
    return resultados


def executar_experimentos_rede_neural():
    dados = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = dividir_dados(dados)
    resultados = []
    
    modelo_rede_neural = treinar_rede_neural(X_treino, y_treino)
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_rede_neural, X_teste, y_teste)
    resultados.append(("Rede Neural Simples", acuracia, matriz_confusao, relatorio))
    
    modelo_rede_neural = treinar_rede_neural(X_treino, y_treino, hidden_layer_sizes=(100, 50))
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_rede_neural, X_teste, y_teste)
    resultados.append(("Rede Neural 2 Camadas Ocultas", acuracia, matriz_confusao, relatorio))
    
    modelo_rede_neural = treinar_rede_neural(X_treino, y_treino, hidden_layer_sizes=(100, 50, 25), activation='tanh')
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_rede_neural, X_teste, y_teste)
    resultados.append(("Rede Neural Mais Camadas Tanh", acuracia, matriz_confusao, relatorio))
    
    modelo_rede_neural = treinar_rede_neural(X_treino, y_treino, alpha=0.01)
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_rede_neural, X_teste, y_teste)
    resultados.append(("Rede Neural Alpha Regularização", acuracia, matriz_confusao, relatorio))
    
    modelo_rede_neural = treinar_rede_neural(X_treino, y_treino, solver='lbfgs')
    acuracia, matriz_confusao, relatorio = avaliar_modelo(modelo_rede_neural, X_teste, y_teste)
    resultados.append(("Rede Neural Solver lbfgs", acuracia, matriz_confusao, relatorio))
    
    return resultados


def imprimir_resultados(resultados, tipo_modelo):
    st.subheader(f"Resultados dos experimentos com {tipo_modelo}")
    for i, (nome, acuracia, matriz_confusao, relatorio) in enumerate(resultados):
        st.write(f"{tipo_modelo} Experimento {i+1}: {nome}")
        st.write(f"Precisão: {acuracia:.2f}")
        st.write("Matriz de Confusão:")
        st.write(matriz_confusao)
        st.write("Relatório de Classificação:")
        st.write(relatorio)
        st.write("---")


def plotar_acuracia(resultados, tipo_modelo):
    acuracias = [resultado[1] for resultado in resultados]
    labels = [resultado[0] for resultado in resultados]
    cores = ['blue', 'green', 'red', 'purple', 'orange']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, acuracias, color=cores)
    ax.set_xlabel('Precisão')
    ax.set_ylabel('Experimentos')
    ax.set_title(f'Precisão dos experimentos com {tipo_modelo}')
    ax.set_xlim(0.90, 1.0)
    st.pyplot(fig)


def plotar_matriz_confusao(matriz_confusao, nome_modelo):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão para {nome_modelo}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot()


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Análise de Modelos de Machine Learning")
    
    st.sidebar.title("Configurações")
    st.sidebar.markdown("Selecione o modelo para visualizar os resultados:")
    tipo_modelo = st.sidebar.radio("Modelo", ("Árvore de Decisão", "Rede Neural"))
    
    dados = carregar_dados()
    
    dados.hist(figsize=(20, 20))
    st.pyplot()

    
    sns.pairplot(dados.iloc[:, 1:5])
    st.pyplot()
    
    if tipo_modelo == "Árvore de Decisão":
        resultados_arvore_decisao = executar_experimentos_arvore_decisao()
        st.subheader("Resultados com Árvore de Decisão")
        imprimir_resultados(resultados_arvore_decisao, "Árvore de Decisão")
        plotar_acuracia(resultados_arvore_decisao, "Árvore de Decisão")
        for resultado in resultados_arvore_decisao:
            plotar_matriz_confusao(resultado[2], resultado[0])
    
    elif tipo_modelo == "Rede Neural":
        resultados_rede_neural = executar_experimentos_rede_neural()
        st.subheader("Resultados com Rede Neural")
        imprimir_resultados(resultados_rede_neural, "Rede Neural")
        plotar_acuracia(resultados_rede_neural, "Rede Neural")
        for resultado in resultados_rede_neural:
            plotar_matriz_confusao(resultado[2], resultado[0])
