import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

print("Iniciando Fase 2: Treinamento Inicial do Modelo XGBoost...")

# 1. Carregar Dados
caminho_dados = os.path.join('Datasets Tratados', 'dataset_petdex_final_PT.csv')
if not os.path.exists(caminho_dados):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_dados}. Rode a Fase 1 primeiro.")

df = pd.read_csv(caminho_dados)

# 2. Definir Features (X) e Target (y)
# O modelo vai prever a Marca_Racao (Target).
# A Dieta_Atual não fará parte do X, pois a API receberá apenas os dados do pet para adivinhar a ração.
y = df['Marca_Racao']
X = df.drop(columns=['Marca_Racao', 'Dieta_Atual'])

print(f"Total de registros: {len(df)}")
print(f"Atributos utilizados (Features): {list(X.columns)}")

# 3. Divisão Treino e Teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Treinamento do Modelo XGBoost
print("\n⏳ Treinando o modelo (Isso pode levar alguns segundos)...")
modelo_xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')
modelo_xgb.fit(X_train, y_train)

# 5. Previsões e Métricas
y_pred = modelo_xgb.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
taxa_erro = 1 - acuracia
matriz_conf = confusion_matrix(y_test, y_pred)
relatorio_classificacao = classification_report(y_test, y_pred)

print("\n=========================================")
print("MÉTRICAS DO MODELO (BASE COMPLETA)")
print("=========================================")
print(f"Taxa de Acerto (Accuracy): {acuracia:.4f} ({acuracia * 100:.2f}%)")
print(f"Taxa de Erro: {taxa_erro:.4f} ({taxa_erro * 100:.2f}%)")
print("\n Matriz de Confusão:")
print(matriz_conf)
print("\n Relatório de Classificação (Precision, Recall, F1):")
print(relatorio_classificacao)
print("=========================================")

# 6. Salvar Modelo e Métricas
os.makedirs('Modelos Gerados', exist_ok=True)
caminho_modelo = os.path.join('Modelos Gerados', 'modelo_xgboost_completo.pkl')
joblib.dump(modelo_xgb, caminho_modelo)
print(f"\n💾 Modelo treinado salvo em: {caminho_modelo}")

# Salvar métricas em um arquivo texto para o relatório final
caminho_metricas = os.path.join('Modelos Gerados', 'metricas_iniciais.txt')
with open(caminho_metricas, 'w', encoding='utf-8') as f:
    f.write("MÉTRICAS DO MODELO - BASE COMPLETA (ANTES DA SELEÇÃO DE ATRIBUTOS)\n")
    f.write("=================================================================\n")
    f.write(f"Acurácia: {acuracia:.4f}\n")
    f.write(f"Taxa de Erro: {taxa_erro:.4f}\n\n")
    f.write("Matriz de Confusão:\n")
    f.write(str(matriz_conf) + "\n\n")
    f.write("Relatório de Classificação:\n")
    f.write(relatorio_classificacao)

print(f" Métricas salvas em: {caminho_metricas}")

