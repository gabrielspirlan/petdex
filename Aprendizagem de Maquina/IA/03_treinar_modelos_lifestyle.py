import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBClassifier

print("[INFO] Iniciando Fase 4: Treinamento dos Modelos de Estilo de Vida Saudavel...")

# 1. Carregar Base de Dados Tratada
caminho_dados = os.path.join('Datasets Tratados', 'dataset_petdex_final_PT.csv')
if not os.path.exists(caminho_dados):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_dados}. Rode a Fase 1 primeiro.")

df = pd.read_csv(caminho_dados)

# 2. Filtrar apenas pelos caes saudaveis (Status_Saude == 1, mapeado de 'Sim')
df_saudavel = df[df['Status_Saude'] == 1].copy()
print(f"[INFO] Registros totais: {len(df)}")
print(f"[INFO] Registros de caes saudaveis: {len(df_saudavel)}")

if len(df_saudavel) == 0:
    raise ValueError("A base de dados nao contem caes rotulados como saudaveis (Status_Saude == 1).")

# 3. Treinar Modelo de Regressao de Peso Ideal
# Features: Raca, Sexo, Idade
X_peso = df_saudavel[['Raca', 'Sexo', 'Idade']]
y_peso = df_saudavel['peso_kg']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_peso, y_peso, test_size=0.2, random_state=42)

print("\n[INFO] Treinando modelo de peso ideal (RandomForestRegressor)...")
modelo_peso = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_peso.fit(X_train_p, y_train_p)

mae_peso = mean_absolute_error(y_test_p, modelo_peso.predict(X_test_p))
print(f"[INFO] Erro Medio Absoluto (MAE) do Peso Ideal: {mae_peso:.2f} kg")

# 4. Treinar Modelo de Regressao de Caminhada Diaria Recomendada
# Features: Raca, Sexo, Idade, peso_kg
X_caminhada = df_saudavel[['Raca', 'Sexo', 'Idade', 'peso_kg']]
y_caminhada = df_saudavel['caminhada_diaria_km']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_caminhada, y_caminhada, test_size=0.2, random_state=42)

print("[INFO] Treinando modelo de caminhada recomendada (RandomForestRegressor)...")
modelo_caminhada = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_caminhada.fit(X_train_c, y_train_c)

mae_caminhada = mean_absolute_error(y_test_c, modelo_caminhada.predict(X_test_c))
print(f"[INFO] Erro Medio Absoluto (MAE) da Caminhada: {mae_caminhada:.2f} km")

# 5. Treinar Modelo de Classificacao de Dieta Recomendada
# Features: Raca, Sexo, Idade, peso_kg
X_dieta = df_saudavel[['Raca', 'Sexo', 'Idade', 'peso_kg']]
y_dieta = df_saudavel['Dieta_Atual']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dieta, y_dieta, test_size=0.2, random_state=42)

print("[INFO] Treinando modelo de dieta recomendada (RandomForestClassifier)...")
modelo_dieta = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_dieta.fit(X_train_d, y_train_d)

acc_dieta = accuracy_score(y_test_d, modelo_dieta.predict(X_test_d))
print(f"[INFO] Acuracia do modelo de Dieta: {acc_dieta * 100:.2f}%")

# 6. Treinar Modelo de Classificacao de Nivel de Atividade Recomendado
# Features: Raca, Sexo, Idade, peso_kg
X_atividade = df_saudavel[['Raca', 'Sexo', 'Idade', 'peso_kg']]
y_atividade = df_saudavel['Nivel_Atividade_Pet']

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_atividade, y_atividade, test_size=0.2, random_state=42)

print("[INFO] Treinando modelo de nivel de atividade (RandomForestClassifier)...")
modelo_atividade = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_atividade.fit(X_train_a, y_train_a)

acc_atividade = accuracy_score(y_test_a, modelo_atividade.predict(X_test_a))
print(f"[INFO] Acuracia do nivel de atividade: {acc_atividade * 100:.2f}%")

# 7. Retreinar o modelo principal XGBoost de Marcas com as 4 features oficiais
# Features: Idade, peso_kg, caminhada_diaria_km, calorias_diarias_RER
print("\n[INFO] Retreinando o modelo principal XGBoost de marcas com as 4 features oficiais...")
X_brand = df[['Idade', 'peso_kg', 'caminhada_diaria_km', 'calorias_diarias_RER']]
y_brand = df['Marca_Racao']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_brand, y_brand, test_size=0.2, random_state=42, stratify=y_brand)

modelo_marca = XGBClassifier(random_state=42, eval_metric='mlogloss')
modelo_marca.fit(X_train_b, y_train_b)

acc_brand = accuracy_score(y_test_b, modelo_marca.predict(X_test_b))
print(f"[INFO] Acuracia do Modelo Principal XGBoost (4 features): {acc_brand * 100:.2f}%")

# 8. Salvar todos os novos modelos
diretorio_modelos = 'Modelos Gerados'
os.makedirs(diretorio_modelos, exist_ok=True)

joblib.dump(modelo_peso, os.path.join(diretorio_modelos, 'modelo_peso_ideal.pkl'))
joblib.dump(modelo_caminhada, os.path.join(diretorio_modelos, 'modelo_caminhada_ideal.pkl'))
joblib.dump(modelo_dieta, os.path.join(diretorio_modelos, 'modelo_dieta_ideal.pkl'))
joblib.dump(modelo_atividade, os.path.join(diretorio_modelos, 'modelo_atividade_ideal.pkl'))
joblib.dump(modelo_marca, os.path.join(diretorio_modelos, 'modelo_xgboost_otimizado.pkl'))

print("\n[INFO] Modelos salvos com sucesso na pasta 'Modelos Gerados/'!")

# 9. Copiar tambem para a pasta de producao da API Python (subir 2 niveis)
pasta_producao = os.path.join('..', '..', 'api-python', 'app', 'modelos_ia')
os.makedirs(pasta_producao, exist_ok=True)

joblib.dump(modelo_peso, os.path.join(pasta_producao, 'modelo_peso_ideal.pkl'))
joblib.dump(modelo_caminhada, os.path.join(pasta_producao, 'modelo_caminhada_ideal.pkl'))
joblib.dump(modelo_dieta, os.path.join(pasta_producao, 'modelo_dieta_ideal.pkl'))
joblib.dump(modelo_atividade, os.path.join(pasta_producao, 'modelo_atividade_ideal.pkl'))
joblib.dump(modelo_marca, os.path.join(pasta_producao, 'modelo_xgboost_otimizado.pkl'))

print(f"[INFO] Modelos copiados com sucesso para a pasta de producao da API: '{pasta_producao}'!")
