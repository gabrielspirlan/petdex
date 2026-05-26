import json
import os
import joblib
import pandas as pd
from datetime import datetime, timezone
import math

# Caminho absoluto para a pasta onde salvamos os artefatos da IA
PASTA_MODELOS = os.path.join(os.path.dirname(__file__), '..', 'modelos_ia')

def carregar_arquivos_ia():
    try:
        modelo = joblib.load(os.path.join(PASTA_MODELOS, 'modelo_xgboost_otimizado.pkl'))
        encoders = joblib.load(os.path.join(PASTA_MODELOS, 'label_encoders.pkl'))
        
        # Carrega os novos modelos de Estilo de Vida Saudável
        modelo_caminhada = joblib.load(os.path.join(PASTA_MODELOS, 'modelo_caminhada_ideal.pkl'))
        modelo_dieta = joblib.load(os.path.join(PASTA_MODELOS, 'modelo_dieta_ideal.pkl'))
        modelo_atividade = joblib.load(os.path.join(PASTA_MODELOS, 'modelo_atividade_ideal.pkl'))
        
        with open(os.path.join(PASTA_MODELOS, 'db-food.json'), 'r', encoding='utf-8') as f:
            catalogo = json.load(f)
            
        return {
            "modelo_marca": modelo,
            "encoders": encoders,
            "modelo_caminhada": modelo_caminhada,
            "modelo_dieta": modelo_dieta,
            "modelo_atividade": modelo_atividade,
            "catalogo": catalogo
        }
    except Exception as e:
        print(f"Erro ao carregar arquivos da IA: {e}")
        return None

def calcular_idade(data_nascimento_str):
    if not data_nascimento_str:
        return None
    try:
        data_nascimento = datetime.fromisoformat(data_nascimento_str.replace("Z", "+00:00"))
        if data_nascimento.tzinfo is None:
            data_nascimento = data_nascimento.replace(tzinfo=timezone.utc)
        hoje = datetime.now(timezone.utc)
        if hoje < data_nascimento:
            return None
        anos = (hoje - data_nascimento).total_seconds() / (86400.0 * 365.2425)
        return max(1, int(math.floor(anos + 0.5)))
    except:
        return None

def gerar_sugestao_nutricional(dados_animal, peso_ideal=None):
    """
    Usa predições em cadeia com modelos de Machine Learning para prever
    as metas saudáveis de estilo de vida, e a marca ideal de ração,
    utilizando o peso ideal recebido como parâmetro.
    """
    if peso_ideal is None:
        raise ValueError("O peso ideal recomendado do animal é obrigatório.")
    try:
        peso_ideal = float(peso_ideal)
        if peso_ideal <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("Peso ideal recomendado inválido. Deve ser um número maior que zero.")

    # Carrega os modelos
    artefatos = carregar_arquivos_ia()
    if not artefatos:
        raise ValueError("Arquivos do modelo de IA não foram carregados corretamente no servidor.")

    modelo_marca = artefatos["modelo_marca"]
    encoders = artefatos["encoders"]
    modelo_caminhada = artefatos["modelo_caminhada"]
    modelo_dieta = artefatos["modelo_dieta"]
    modelo_atividade = artefatos["modelo_atividade"]
    catalogo = artefatos["catalogo"]

    # 1. VALIDAÇÃO E EXTRAÇÃO DOS INPUTS OBRIGATÓRIOS
    # Peso Atual (Usando apenas o parâmetro oficial 'peso' retornado pela API Java)
    peso_val = dados_animal.get("peso")
    if peso_val is None:
        raise ValueError("O peso do animal é obrigatório para gerar a recomendação.")
    try:
        peso_kg = float(peso_val)
        if peso_kg <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("Peso do animal inválido. Deve ser um número maior que zero.")

    # Idade
    data_nasc = dados_animal.get("dataNascimento")
    if not data_nasc:
        raise ValueError("A data de nascimento do animal é obrigatória.")
    idade = calcular_idade(data_nasc)
    if idade is None:
        raise ValueError("Idade do animal inválida ou não pôde ser calculada a partir da data de nascimento.")

    # Caminhada Diária Atual (KM) - Sem erro caso ausente (mantém fallback 0.0)
    caminhada_val = dados_animal.get("caminhada_diaria_km") or dados_animal.get("caminhadaDiariaKm") or dados_animal.get("caminhada_diaria") or dados_animal.get("caminhadaDiaria")
    if caminhada_val is not None:
        try:
            caminhada_diaria_km = float(caminhada_val)
            if caminhada_diaria_km < 0:
                caminhada_diaria_km = 0.0
        except ValueError:
            caminhada_diaria_km = 0.0
    else:
        caminhada_diaria_km = 0.0

    # 2. CODIFICAÇÃO DAS CATEGORIAS DO ANIMAL
    le_raca = encoders['Raca']
    le_sexo = encoders['Sexo']

    # Normalização da raça usando racaNome
    raca_crua = dados_animal.get("racaNome") or "SRD (Sem Raça Definida)"
    raca_nome = "SRD (Sem Raça Definida)"
    for classe in le_raca.classes_:
        if raca_crua.strip().lower() == classe.strip().lower():
            raca_nome = classe
            break
    
    # Normalização do sexo
    sexo_cru = dados_animal.get("sexo") or dados_animal.get("sex") or "Macho"
    if (sexo_cru or "").lower() in ['f', 'femea', 'fêmea', 'female']:
        sexo_nome = 'Fêmea'
    else:
        sexo_nome = 'Macho'

    raca_encoded = le_raca.transform([raca_nome])[0]
    sexo_encoded = le_sexo.transform([sexo_nome])[0]

    # 3. FLUXO DE INFERÊNCIAS EM CADEIA (LIFESTYLE SAUDÁVEL)
    try:
        # A. Peso Ideal recebido como parâmetro
        peso_ideal = round(peso_ideal, 2)

        # B. Cálculo de Calorias RER com base no Peso Ideal
        calorias_diarias_RER = round(70 * (peso_ideal ** 0.75), 2)

        # C. Predição da Meta de Caminhada Recomendada (com base no Peso Ideal)
        X_caminhada = pd.DataFrame([{
            'Raca': raca_encoded,
            'Sexo': sexo_encoded,
            'Idade': idade,
            'peso_kg': peso_ideal
        }])
        caminhada_diaria_km_meta = round(float(modelo_caminhada.predict(X_caminhada)[0]), 2)

        # D. Predição da Dieta Recomendada
        X_dieta = pd.DataFrame([{
            'Raca': raca_encoded,
            'Sexo': sexo_encoded,
            'Idade': idade,
            'peso_kg': peso_ideal
        }])
        dieta_pred_num = modelo_dieta.predict(X_dieta)[0]
        dieta_recomendada = encoders['Dieta_Atual'].inverse_transform([dieta_pred_num])[0]

        # E. Predição de Nível de Atividade Recomendado
        X_atividade = pd.DataFrame([{
            'Raca': raca_encoded,
            'Sexo': sexo_encoded,
            'Idade': idade,
            'peso_kg': peso_ideal
        }])
        atividade_pred_num = modelo_atividade.predict(X_atividade)[0]
        atividade_recomendada = encoders['Nivel_Atividade_Pet'].inverse_transform([atividade_pred_num])[0]

        # F. Predição da Marca de Ração Ideal (XGBoost) usando as metas saudáveis
        X_brand = pd.DataFrame([{
            'Idade': idade,
            'peso_kg': peso_ideal,
            'caminhada_diaria_km': caminhada_diaria_km_meta,
            'calorias_diarias_RER': calorias_diarias_RER
        }])
        predicao_brand_num = modelo_marca.predict(X_brand)[0]
        marca_prevista_nome = encoders['Marca_Racao'].inverse_transform([predicao_brand_num])[0]

    except Exception as e:
        raise ValueError(f"Não foi possível obter a recomendação da IA devido a uma falha nos modelos: {str(e)}")

    # 4. DIAGNÓSTICO CORPORAL COMPARATIVO
    if peso_kg > (peso_ideal * 1.15):
        status_corpo = 'Sobrepeso'
    elif peso_kg < (peso_ideal * 0.85):
        status_corpo = 'Abaixo do Peso'
    else:
        status_corpo = 'Peso Ideal'

    # 5. FILTRAR CATÁLOGO DE PRODUTOS COM BASE NA MARCA, PORTE E STATUS CORPORAL
    # O porte é calculado dinamicamente com base no peso real do animal
    if peso_kg <= 10:
        porte_pet_original = 'Pequeno'
    elif peso_kg <= 25:
        porte_pet_original = 'Médio'
    else:
        porte_pet_original = 'Grande'

    mapa_porte = {'Pequeno': 'Small', 'Médio': 'Medium', 'Grande': 'Large'}
    porte_busca = mapa_porte.get(porte_pet_original, 'All')

    sugestoes = []
    for produto in catalogo:
        match_marca = (produto.get('brand') == marca_prevista_nome)
        match_porte = (produto.get('animalSize') == "All" or produto.get('animalSize') == porte_busca)
        
        # Filtro de nutrição baseada no status corporal
        match_nutricao = False
        if status_corpo == 'Sobrepeso':
            match_nutricao = produto.get('condition') in ['Overweight', 'Weight Management', 'Weight Care', 'Active/Weight Management']
        elif status_corpo == 'Abaixo do Peso':
            match_nutricao = produto.get('condition') is None or "Puppy" in produto.get('name', '')
        else:
            match_nutricao = produto.get('condition') in [None, 'Everyday Health']

        if match_marca and match_porte and match_nutricao:
            sugestoes.append(produto['name'])

    # Fallbacks de relaxamento de filtros se nenhum for encontrado
    if not sugestoes:
        for produto in catalogo:
            if produto.get('brand') == marca_prevista_nome and (produto.get('animalSize') == "All" or produto.get('animalSize') == porte_busca):
                sugestoes.append(produto['name'])
    if not sugestoes:
        for produto in catalogo:
            if produto.get('brand') == marca_prevista_nome:
                sugestoes.append(produto['name'])
    if not sugestoes:
        sugestoes = [f"Ração recomendada da marca {marca_prevista_nome}"]

    # 6. JUSTIFICATIVA E METAS DE ESTILO DE VIDA
    if status_corpo == 'Sobrepeso':
        justificativa = (
            f"O pet esta com sobrepeso (Peso Atual: {peso_kg:.1f} kg vs. Peso Ideal: {peso_ideal:.1f} kg). "
            f"Recomenda-se reduzir a ingestao calorica diaria para {calorias_diarias_RER} kcal e "
            f"elevar a atividade fisica do animal para a meta '{atividade_recomendada}'."
        )
    elif status_corpo == 'Abaixo do Peso':
        justificativa = (
            f"O pet esta abaixo do peso (Peso Atual: {peso_kg:.1f} kg vs. Peso Ideal: {peso_ideal:.1f} kg). "
            f"Recomenda-se adotar uma dieta do tipo '{dieta_recomendada}' de alto teor energetico "
            f"e limitar atividades exaustivas temporariamente para acumulo de massa."
        )
    else:
        justificativa = (
            f"Parabens! O pet esta na faixa de peso ideal ({peso_kg:.1f} kg). "
            f"Mantenha a dieta atual equilibrada e siga a rotina de exercicios para preservar a saude."
        )

    return {
        "status_corporal": status_corpo,
        "peso_atual": peso_kg,
        "peso_referencia": peso_ideal,
        "recomendacoes": sugestoes[:2],
        "marca_prevista": marca_prevista_nome,
        "recomendacoes_estilo_vida": {
            "caminhada_diaria_km_meta": caminhada_diaria_km_meta,
            "nivel_atividade_meta": atividade_recomendada,
            "tempo_brincadeira_horas_meta": 1.5 if status_corpo == 'Sobrepeso' else 1.0,
            "tipo_dieta_meta": dieta_recomendada,
            "justificativa": justificativa
        }
    }