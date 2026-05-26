import sys
import os

# Adiciona o caminho da pasta app ao path do Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services import recomendacao_ia

def verificar_status_corporal(resultado):
    ideal = resultado['peso_referencia']
    atual = resultado['peso_atual']
    if atual > ideal * 1.15:
        return 'Sobrepeso'
    elif atual < ideal * 0.85:
        return 'Abaixo do Peso'
    else:
        return 'Peso Ideal'

def testar_casos():
    print("=" * 60)
    print("[INFO] Iniciando testes de recomendacao nutricional e lifestyle por IA...")
    print("=" * 60)

    # Caso 1: Cão em Sobrepeso (Labrador Retriever, ideal 23.26kg, atual 45kg)
    print("\n[TESTE 1] Testando cao com Sobrepeso (Labrador de 45kg)")
    pet_sobrepeso = {
        "racaNome": "Labrador Retriever",
        "sexo": "Macho",
        "dataNascimento": "2021-05-20T00:00:00Z",
        "peso": 45.0,
        "porte": "Grande",
        "caminhada_diaria_km": 1.0
    }
    res1 = recomendacao_ia.gerar_sugestao_nutricional(pet_sobrepeso, peso_ideal=23.26)
    print(f"  - Diagnostico corporal: {res1['status_corporal']}")
    print(f"  - Peso atual: {res1['peso_atual']} kg")
    print(f"  - Peso ideal previsto: {res1['peso_referencia']} kg")
    print(f"  - Sugestoes de Racao: {res1['recomendacoes']}")
    print(f"  - Meta de Caminhada: {res1['recomendacoes_estilo_vida']['caminhada_diaria_km_meta']} km")
    print(f"  - Meta de Dieta: {res1['recomendacoes_estilo_vida']['tipo_dieta_meta']}")
    print(f"  - Meta de Atividade: {res1['recomendacoes_estilo_vida']['nivel_atividade_meta']}")
    print(f"  - Justificativa: {res1['recomendacoes_estilo_vida']['justificativa']}")
    
    assert res1['status_corporal'] == verificar_status_corporal(res1), "Erro no status corporal do Caso 1"
    assert len(res1['recomendacoes']) > 0, "Deveria ter recomendacoes de racao"
    
    # Caso 2: Cão Abaixo do Peso (Golden Retriever, atual 12kg)
    print("\n[TESTE 2] Testando cao Abaixo do Peso (Golden de 12kg)")
    pet_abaixo = {
        "racaNome": "Golden Retriever",
        "sexo": "Fêmea",
        "dataNascimento": "2023-01-10T00:00:00Z",
        "peso": 12.0,
        "porte": "Grande",
        "caminhada_diaria_km": 3.0
    }
    res2 = recomendacao_ia.gerar_sugestao_nutricional(pet_abaixo, peso_ideal=21.7)
    print(f"  - Diagnostico corporal: {res2['status_corporal']}")
    print(f"  - Peso atual: {res2['peso_atual']} kg")
    print(f"  - Peso ideal previsto: {res2['peso_referencia']} kg")
    print(f"  - Sugestoes de Racao: {res2['recomendacoes']}")
    print(f"  - Meta de Caminhada: {res2['recomendacoes_estilo_vida']['caminhada_diaria_km_meta']} km")
    print(f"  - Meta de Dieta: {res2['recomendacoes_estilo_vida']['tipo_dieta_meta']}")
    print(f"  - Justificativa: {res2['recomendacoes_estilo_vida']['justificativa']}")
    
    assert res2['status_corporal'] == verificar_status_corporal(res2), "Erro no status corporal do Caso 2"

    # Caso 3: Cão em Peso Ideal (Beagle, atual 21.8kg)
    print("\n[TESTE 3] Testando cao com Peso Ideal (Beagle)")
    pet_ideal = {
        "racaNome": "Beagle",
        "sexo": "Macho",
        "dataNascimento": "2020-03-15T00:00:00Z",
        "peso": 21.8,
        "porte": "Médio",
        "caminhada_diaria_km": 2.0
    }
    res3 = recomendacao_ia.gerar_sugestao_nutricional(pet_ideal, peso_ideal=23.38)
    print(f"  - Diagnostico corporal: {res3['status_corporal']}")
    print(f"  - Peso atual: {res3['peso_atual']} kg")
    print(f"  - Peso ideal previsto: {res3['peso_referencia']} kg")
    print(f"  - Sugestoes de Racao: {res3['recomendacoes']}")
    print(f"  - Meta de Caminhada: {res3['recomendacoes_estilo_vida']['caminhada_diaria_km_meta']} km")
    print(f"  - Justificativa: {res3['recomendacoes_estilo_vida']['justificativa']}")
    
    assert res3['status_corporal'] == verificar_status_corporal(res3), "Erro no status corporal do Caso 3"

    # Caso 4: SRD por porte (SRD Pequeno de 12kg -> Deveria ser sobrepeso, pois ideal pra Pequeno é ~7.0kg)
    print("\n[TESTE 4] Testando cao SRD Pequeno de 12kg")
    pet_srd = {
        "racaNome": "SRD (Sem Raça Definida)",
        "sexo": "Fêmea",
        "dataNascimento": "2022-09-01T00:00:00Z",
        "peso": 12.0,
        "porte": "Pequeno",
        "caminhada_diaria_km": 1.5
    }
    res4 = recomendacao_ia.gerar_sugestao_nutricional(pet_srd, peso_ideal=7.0)
    print(f"  - Diagnostico corporal: {res4['status_corporal']}")
    print(f"  - Peso atual: {res4['peso_atual']} kg")
    print(f"  - Peso ideal previsto: {res4['peso_referencia']} kg")
    print(f"  - Sugestoes de Racao: {res4['recomendacoes']}")
    print(f"  - Meta de Caminhada: {res4['recomendacoes_estilo_vida']['caminhada_diaria_km_meta']} km")
    print(f"  - Justificativa: {res4['recomendacoes_estilo_vida']['justificativa']}")
    
    assert res4['status_corporal'] == verificar_status_corporal(res4), "Erro no status corporal do Caso 4"

    # Caso 5: Testes de Validação e Falhas Esperadas
    print("\n[TESTE 5] Testando validacoes de erros esperados")
    
    # Sem peso
    try:
        recomendacao_ia.gerar_sugestao_nutricional({"racaNome": "Poodle"}, peso_ideal=5.0)
        assert False, "Deveria falhar sem peso"
    except ValueError as e:
        print(f"  - Falhou corretamente (sem peso): {e}")

    # Sem data de nascimento
    try:
        recomendacao_ia.gerar_sugestao_nutricional({"racaNome": "Poodle", "peso": 8.0}, peso_ideal=5.0)
        assert False, "Deveria falhar sem data de nascimento"
    except ValueError as e:
        print(f"  - Falhou corretamente (sem nascimento): {e}")

    # Sem caminhada diária (agora é opcional com fallback)
    try:
        res_sem_caminhada = recomendacao_ia.gerar_sugestao_nutricional({
            "racaNome": "Poodle",
            "peso": 8.0,
            "dataNascimento": "2020-01-01T00:00:00Z"
        }, peso_ideal=5.0)
        print("  - Passou corretamente (sem caminhada diária agora é opcional com fallback)")
    except ValueError as e:
        assert False, f"Não deveria falhar sem caminhada diária: {e}"

    # Sem peso ideal
    try:
        recomendacao_ia.gerar_sugestao_nutricional({
            "racaNome": "Poodle",
            "peso": 8.0,
            "dataNascimento": "2020-01-01T00:00:00Z"
        }, peso_ideal=None)
        assert False, "Deveria falhar sem peso ideal"
    except ValueError as e:
        print(f"  - Falhou corretamente (sem peso ideal): {e}")

    # Peso ideal inválido
    try:
        recomendacao_ia.gerar_sugestao_nutricional({
            "racaNome": "Poodle",
            "peso": 8.0,
            "dataNascimento": "2020-01-01T00:00:00Z"
        }, peso_ideal=-2.0)
        assert False, "Deveria falhar com peso ideal negativo"
    except ValueError as e:
        print(f"  - Falhou corretamente (peso ideal inválido): {e}")

    print("\n" + "=" * 60)
    print("[INFO] Todos os testes de recomendacao foram concluidos com sucesso!")
    print("=" * 60)

if __name__ == "__main__":
    testar_casos()
