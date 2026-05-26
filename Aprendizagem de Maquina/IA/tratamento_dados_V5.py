import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

# Nesse arquivo ele basicamente é a "faxina" inicial. Ele pega os dados brutos 
# e os deixa prontos para o Brasil e prepara as colunas categóricas para o Machine Learning.

PATH_BASES = 'Base de Dados'

try:
    df = pd.read_csv(os.path.join(PATH_BASES, 'synthetic_dog_breed_health_data.csv'))

    # --- 1. LIMPEZA E PREENCHIMENTO ---
    df['Breed'] = df['Breed'].fillna('SRD (Sem Raça Definida)')
    
    num_cols = ['Age', 'Weight (lbs)', 'Daily Walk Distance (miles)', 'Hours of Sleep', 'Play Time (hrs)', 'Annual Vet Visits']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = ['Sex', 'Diet', 'Food Brand', 'Medications', 'Seizures', 'Daily Activity Level', 'Owner Activity Level', 'Healthy']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # --- 2. TRADUÇÃO DO CONTEÚDO (LINHAS) ---
    traducoes = {
        'Sex': {'Male': 'Macho', 'Female': 'Fêmea'},
        'Healthy': {'Yes': 'Sim', 'No': 'Não'},
        'Medications': {'Yes': 'Sim', 'No': 'Não'},
        'Seizures': {'Yes': 'Sim', 'No': 'Não'},
        'Daily Activity Level': {
            'None': 'Nenhum', 
            'Low': 'Baixo', 
            'Moderate': 'Moderado', 
            'Active': 'Ativo', 
            'Very Active': 'Muito Ativo'
        },
        'Owner Activity Level': {
            'None': 'Nenhum', 
            'Low': 'Baixo', 
            'Moderate': 'Moderado', 
            'Active': 'Ativo', 
            'Very Active': 'Muito Ativo'
        },
        'Diet': {
            'Wet food': 'Úmida', 
            'Hard food': 'Ração Seca', 
            'Home cooked': 'Caseira', 
            'Special diet': 'Especial'
        }
    }

    for coluna, mapa in traducoes.items():
        if coluna in df.columns:
            df[coluna] = df[coluna].map(mapa).fillna(df[coluna])

    # --- 3. ENGENHARIA E CONVERSÕES ---
    df['peso_kg'] = (df['Weight (lbs)'] * 0.453592).round(2)
    df['caminhada_diaria_km'] = (df['Daily Walk Distance (miles)'] * 1.60934).round(2)
    
    def get_size(w):
        if w <= 10: return 'Pequeno'
        elif w <= 25: return 'Médio'
        else: return 'Grande'
    df['porte_animal'] = df['peso_kg'].apply(get_size)
    
    df['calorias_diarias_RER'] = (70 * (df['peso_kg'] ** 0.75)).round(2)

    # --- 4. RENOMEAÇÃO ---
    mapeamento_colunas = {
        'Breed': 'Raca',
        'Sex': 'Sexo',
        'Age': 'Idade',
        'Diet': 'Dieta_Atual',
        'Food Brand': 'Marca_Racao',
        'Medications': 'Medicamentos',
        'Seizures': 'Convulsoes',
        'Hours of Sleep': 'Horas_Sono',
        'Play Time (hrs)': 'Tempo_Brincadeira_Horas',
        'Daily Activity Level': 'Nivel_Atividade_Pet',
        'Owner Activity Level': 'Nivel_Atividade_Dono',
        'Annual Vet Visits': 'Visitas_Anuais_Veterinario',
        'Healthy': 'Status_Saude'
    }
    
    df_pt = df.rename(columns=mapeamento_colunas)
    
    # Remover apenas as originais em inglês de peso e distância, mantendo todo o resto
    if 'Weight (lbs)' in df_pt.columns: df_pt = df_pt.drop(columns=['Weight (lbs)'])
    if 'Daily Walk Distance (miles)' in df_pt.columns: df_pt = df_pt.drop(columns=['Daily Walk Distance (miles)'])
    
    df_final = df_pt.copy()

    # --- 5. ENCODING PARA MACHINE LEARNING ---
    # Modelos avançados precisam de números. Vamos codificar as colunas de texto (categóricas).
    colunas_categoricas = df_final.select_dtypes(include=['object']).columns.tolist()
                           
    encoders = {}
    
    for col in colunas_categoricas:
        if col in df_final.columns:
            le = LabelEncoder()
            # Transforma texto em número
            df_final[col] = le.fit_transform(df_final[col].astype(str))
            encoders[col] = le
            
    # Salvar os encoders em disco para usarmos na API posteriormente
    os.makedirs('Modelos Gerados', exist_ok=True)
    joblib.dump(encoders, os.path.join('Modelos Gerados', 'label_encoders.pkl'))
    print("✅ LabelEncoders salvos em 'Modelos Gerados/label_encoders.pkl'.")

    # --- 6. SALVANDO BASE FINAL ---
    os.makedirs('Datasets Tratados', exist_ok=True)
    caminho_final = os.path.join('Datasets Tratados', 'dataset_petdex_final_PT.csv')
    df_final.to_csv(caminho_final, index=False, encoding='utf-8-sig')
    print(f"✅ Base de dados tratada salva em '{caminho_final}'.")


except Exception as e:
    print(f"❌ Erro: {e}")