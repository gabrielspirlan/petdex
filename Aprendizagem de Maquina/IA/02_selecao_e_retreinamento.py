import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

print(" Iniciando Fase 3: Seleção de Atributos e Retreinamento...")

# 1. Carregar Dados
caminho_dados = os.path.join('Datasets Tratados', 'dataset_petdex_final_PT.csv')
if not os.path.exists(caminho_dados):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_dados}. Rode a Fase 1 primeiro.")

df = pd.read_csv(caminho_dados)

# 2. Definir Features (X) e Target (y)
y = df['Marca_Racao']
X = df.drop(columns=['Marca_Racao', 'Dieta_Atual'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Carregar o Modelo Inicial para extrair as Feature Importances
caminho_modelo_inicial = os.path.join('Modelos Gerados', 'modelo_xgboost_completo.pkl')
if not os.path.exists(caminho_modelo_inicial):
    raise FileNotFoundError("Modelo inicial não encontrado. Rode a Fase 2 primeiro.")

modelo_inicial = joblib.load(caminho_modelo_inicial)

# 4. Seleção de Atributos (Feature Selection)
# Vamos usar o SelectFromModel do Scikit-Learn baseado no XGBoost
# Ele vai escolher os atributos cuja importância seja maior ou igual à importância média de todos os atributos
seletor = SelectFromModel(modelo_inicial, prefit=True)

# Transforma a base X para manter só as colunas aprovadas
X_train_reduzido = seletor.transform(X_train)
X_test_reduzido = seletor.transform(X_test)

# Descobrir o nome das colunas escolhidas
colunas_selecionadas = X.columns[seletor.get_support()]

print("\n=========================================")
print(" SELEÇÃO DE ATRIBUTOS (FEATURE SELECTION)")
print("=========================================")
print(f"Total de atributos originais: {X.shape[1]}")
print(f"Total de atributos mantidos: {len(colunas_selecionadas)}")
print(f" Atributos selecionados pela IA: {list(colunas_selecionadas)}")
print("=========================================")

# 5. Retreinamento com a Nova Base
print("\n Retreinando o modelo apenas com os melhores atributos...")
modelo_otimizado = XGBClassifier(random_state=42, eval_metric='mlogloss')
modelo_otimizado.fit(X_train_reduzido, y_train)

# 6. Novas Métricas
y_pred = modelo_otimizado.predict(X_test_reduzido)

acuracia = accuracy_score(y_test, y_pred)
taxa_erro = 1 - acuracia
matriz_conf = confusion_matrix(y_test, y_pred)
relatorio_classificacao = classification_report(y_test, y_pred)

print("\n=========================================")
print(" MÉTRICAS DO MODELO (PÓS SELEÇÃO DE ATRIBUTOS)")
print("=========================================")
print(f" Nova Taxa de Acerto (Accuracy): {acuracia:.4f} ({acuracia * 100:.2f}%)")
print(f" Nova Taxa de Erro: {taxa_erro:.4f} ({taxa_erro * 100:.2f}%)")
print("\n Matriz de Confusão:")
print(matriz_conf)
print("=========================================")

# 7. Salvar Modelo Otimizado e Lista de Features
caminho_modelo_otimizado = os.path.join('Modelos Gerados', 'modelo_xgboost_otimizado.pkl')
joblib.dump(modelo_otimizado, caminho_modelo_otimizado)

caminho_features = os.path.join('Modelos Gerados', 'features_selecionadas.pkl')
joblib.dump(list(colunas_selecionadas), caminho_features)

# Salvar métricas em um arquivo texto
caminho_metricas = os.path.join('Modelos Gerados', 'metricas_otimizadas.txt')
with open(caminho_metricas, 'w', encoding='utf-8') as f:
    f.write("MÉTRICAS DO MODELO - PÓS SELEÇÃO DE ATRIBUTOS\n")
    f.write("==============================================\n")
    f.write(f"Atributos selecionados ({len(colunas_selecionadas)}): {list(colunas_selecionadas)}\n\n")
    f.write(f"Acurácia: {acuracia:.4f}\n")
    f.write(f"Taxa de Erro: {taxa_erro:.4f}\n\n")
    f.write("Matriz de Confusão:\n")
    f.write(str(matriz_conf) + "\n\n")
    f.write("Relatório de Classificação:\n")
    f.write(relatorio_classificacao)

print(f"\n Novo modelo otimizado salvo em: {caminho_modelo_otimizado}")
print(f" Novas métricas salvas em: {caminho_metricas}")
