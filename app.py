"""
Dashboard de Positividade Toxicológica
Synvia Group - Análise de Laboratórios
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta, date
from pymongo import MongoClient
from bson import ObjectId
import hashlib
import json
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Aumentar limite do Pandas Styler para tabelas grandes
pd.set_option("styler.render.max_elements", 5000000)

# ============================================
# CONFIGURAÇÃO DE PERÍODO PADRÃO
# ============================================
def get_period_by_days(days: int):
    """
    Retorna o período baseado na quantidade de dias.
    """
    end_date = datetime.now().replace(hour=23, minute=59, second=59)
    start_date = (datetime.now() - timedelta(days=days)).replace(hour=0, minute=0, second=0)
    return start_date, end_date


def get_default_period():
    """
    Retorna o período padrão para consultas: últimos 30 dias.
    """
    return get_period_by_days(30)


def get_selected_period():
    """
    Retorna o período selecionado pelo usuário na sessão.
    Se não houver seleção, retorna o período padrão (30 dias).
    """
    if 'periodo_inicio' in st.session_state and 'periodo_fim' in st.session_state:
        start = st.session_state.periodo_inicio
        end = st.session_state.periodo_fim
        # Converter date para datetime se necessário
        if isinstance(start, date) and not isinstance(start, datetime):
            start = datetime.combine(start, datetime.min.time())
        if isinstance(end, date) and not isinstance(end, datetime):
            end = datetime.combine(end, datetime.max.time().replace(microsecond=0))
        return start, end
    return get_default_period()


def init_period_session():
    """
    Inicializa as variáveis de período na sessão.
    """
    if 'periodo_tipo' not in st.session_state:
        st.session_state.periodo_tipo = "30 dias"
    if 'periodo_inicio' not in st.session_state or 'periodo_fim' not in st.session_state:
        start, end = get_default_period()
        st.session_state.periodo_inicio = start
        st.session_state.periodo_fim = end


# Período padrão global (usado apenas para cache inicial)
DEFAULT_START_DATE, DEFAULT_END_DATE = get_default_period()

# Configuração da página - DEVE ser a primeira chamada Streamlit
st.set_page_config(
    page_title="Dashboard Toxicologia | Synvia",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports após configuração da página
from auth_microsoft import (
    MicrosoftAuth,
    AuthManager,
    create_login_page,
    create_user_header
)


# ============================================
# SISTEMA DE CACHE POR SESSÃO
# ============================================

def init_session_cache():
    """Inicializa o sistema de cache na sessão"""
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    if 'cache_keys' not in st.session_state:
        st.session_state.cache_keys = {}
    if 'base_data_loaded' not in st.session_state:
        st.session_state.base_data_loaded = False


def generate_cache_key(*args) -> str:
    """Gera uma chave de cache única baseada nos parâmetros"""
    # Converter argumentos para string de forma segura
    def serialize(obj):
        if obj is None:
            return "None"
        if isinstance(obj, (list, set)):
            return json.dumps(sorted([str(x) for x in obj]))
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, ObjectId):
            return str(obj)
        return str(obj)

    key_parts = [serialize(arg) for arg in args]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cached_data(cache_name: str, key: str):
    """Recupera dados do cache da sessão"""
    init_session_cache()
    cache = st.session_state.data_cache.get(cache_name, {})
    return cache.get(key)


def set_cached_data(cache_name: str, key: str, data):
    """Armazena dados no cache da sessão"""
    init_session_cache()
    if cache_name not in st.session_state.data_cache:
        st.session_state.data_cache[cache_name] = {}
    st.session_state.data_cache[cache_name][key] = data


def clear_cache(cache_name: str = None):
    """Limpa o cache (específico ou todo)"""
    init_session_cache()
    if cache_name:
        st.session_state.data_cache.pop(cache_name, None)
    else:
        st.session_state.data_cache = {}


def run_parallel_tasks(tasks: dict, max_workers: int = 5) -> dict:
    """
    Executa múltiplas funções em paralelo usando ThreadPoolExecutor.

    Args:
        tasks: dict com {nome_task: (funcao, args, kwargs)}
               ou {nome_task: (funcao, args)}
               ou {nome_task: funcao}
        max_workers: número máximo de threads

    Returns:
        dict com {nome_task: resultado}
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {}

        for name, task_info in tasks.items():
            if callable(task_info):
                # Apenas função sem argumentos
                future = executor.submit(task_info)
            elif len(task_info) == 2:
                func, args = task_info
                future = executor.submit(func, *args)
            else:
                func, args, kwargs = task_info
                future = executor.submit(func, *args, **kwargs)

            future_to_name[future] = name

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = None
                st.error(f"Erro ao carregar {name}: {e}")

    return results


def load_base_data():
    """
    Carrega dados básicos que são usados em múltiplas páginas.
    Chamado uma vez no início da sessão.
    Usa processamento paralelo para carregar mais rápido.
    Pré-carrega TODOS os dados do período padrão (últimos 30 dias) para acelerar consultas subsequentes.
    """
    init_session_cache()

    if st.session_state.base_data_loaded:
        return

    # Garantir índices (só executa se necessário)
    ensure_indexes()

    # Pré-carregar TODOS os dados em paralelo (popula o cache do Streamlit)
    with st.spinner("Carregando dados... (primeira execução pode demorar)"):
        preload_all_data()

    st.session_state.base_data_loaded = True

# ============================================
# CONEXÃO MONGODB
# ============================================

@st.cache_resource
def get_mongo_client():
    """Conecta ao MongoDB usando connection string do secrets"""
    connection_string = st.secrets["mongodb"]["MONGO_CONNECTION_STRING"]
    client = MongoClient(
        connection_string,
        maxPoolSize=50,
        minPoolSize=10,
        maxIdleTimeMS=30000,
        connectTimeoutMS=5000,
        serverSelectionTimeoutMS=5000
    )
    return client

@st.cache_resource
def get_database():
    """Retorna o banco de dados ctox com conexão otimizada"""
    client = get_mongo_client()
    return client["ctox"]

def get_collection(collection_name: str):
    """Retorna uma collection do banco ctox"""
    db = get_database()
    return db[collection_name]


def ensure_indexes():
    """Cria índices para otimizar queries - executar uma vez"""
    try:
        db = get_database()

        # Índices para lots
        db.lots.create_index([("analysisType", 1), ("createdAt", 1)])
        db.lots.create_index([("code", 1)])
        db.lots.create_index([("createdAt", 1)])

        # Índices para results
        db.results.create_index([("_lot", 1)])

        # Índices para gatherings
        db.gatherings.create_index([("_laboratory", 1)])
        db.gatherings.create_index([("_chainOfCustody", 1)])
        db.gatherings.create_index([("createdAt", 1)])

        # Índices para chainofcustodies
        db.chainofcustodies.create_index([("createdAt", 1)])
        db.chainofcustodies.create_index([("sample.code", 1)])

    except Exception:
        pass  # Índices já existem ou sem permissão


def test_connection() -> bool:
    """Testa a conexão com o MongoDB"""
    try:
        client = get_mongo_client()
        client.admin.command('ping')
        return True
    except:
        return False


# ============================================
# CACHE DE DADOS PRÉ-CARREGADOS
# ============================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_all_lots(start_date: datetime = None, end_date: datetime = None):
    """
    Pré-carrega TODOS os lotes do período selecionado.
    Retorna dict {lot_code: {analysisType, createdAt, _samples, month}}
    """
    try:
        lots_collection = get_collection("lots")
        if start_date is None or end_date is None:
            start_date, end_date = get_default_period()

        lots = list(lots_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"code": 1, "analysisType": 1, "createdAt": 1, "_samples": 1}
        ))

        lots_dict = {}
        for lot in lots:
            code = lot.get('code')
            if code:
                lots_dict[code] = {
                    "analysisType": lot.get('analysisType'),
                    "createdAt": lot.get('createdAt'),
                    "month": lot.get('createdAt').month if lot.get('createdAt') else None,
                    "_samples": set(lot.get('_samples', []))
                }

        return lots_dict
    except Exception as e:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def get_all_results(start_date: datetime = None, end_date: datetime = None):
    """
    Pré-carrega TODOS os resultados do período selecionado.
    Retorna dict {lot_code: [{_sample, positive, compounds}]}
    """
    try:
        # Primeiro pegar os códigos dos lotes do período
        lots_dict = get_all_lots(start_date, end_date)
        lot_codes = list(lots_dict.keys())

        if not lot_codes:
            return {}

        # Buscar todos os resultados de uma vez
        results_collection = get_collection("results")
        results = list(results_collection.find(
            {"_lot": {"$in": lot_codes}},
            {"_lot": 1, "samples": 1}
        ))

        results_dict = {}
        for result in results:
            lot_code = result.get('_lot')
            if lot_code not in results_dict:
                results_dict[lot_code] = []

            for sample in result.get('samples', []):
                results_dict[lot_code].append({
                    "_sample": sample.get('_sample'),
                    "positive": sample.get('positive', False),
                    "compounds": sample.get('compounds', [])
                })

        return results_dict
    except Exception as e:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def get_all_gatherings(start_date: datetime = None, end_date: datetime = None):
    """
    Pré-carrega mapeamento de chainOfCustody -> laboratory do período selecionado.
    Retorna dict com chain_to_lab, chain_to_purpose e chain_to_date
    """
    try:
        gatherings_collection = get_collection("gatherings")
        if start_date is None or end_date is None:
            start_date, end_date = get_default_period()

        gatherings = list(gatherings_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"_chainOfCustody": 1, "_laboratory": 1, "purpose.type": 1, "createdAt": 1}
        ))

        chain_to_lab = {}
        chain_to_purpose = {}
        chain_to_date = {}

        for g in gatherings:
            chain_id = g.get('_chainOfCustody')
            lab_id = g.get('_laboratory')
            purpose = g.get('purpose', {}).get('type')
            created_at = g.get('createdAt')

            if chain_id:
                if lab_id:
                    chain_to_lab[chain_id] = str(lab_id) if isinstance(lab_id, ObjectId) else lab_id
                if purpose:
                    chain_to_purpose[chain_id] = purpose
                if created_at:
                    chain_to_date[chain_id] = created_at

        return {"chain_to_lab": chain_to_lab, "chain_to_purpose": chain_to_purpose, "chain_to_date": chain_to_date}
    except Exception as e:
        return {"chain_to_lab": {}, "chain_to_purpose": {}, "chain_to_date": {}}


@st.cache_data(ttl=1800, show_spinner=False)
def get_chain_to_sample_map(start_date: datetime = None, end_date: datetime = None):
    """
    Pré-carrega mapeamento chain_id -> sample.code do período selecionado.
    """
    try:
        chains_collection = get_collection("chainofcustodies")
        if start_date is None or end_date is None:
            start_date, end_date = get_default_period()

        chains = list(chains_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"_id": 1, "sample.code": 1}
        ))

        chain_to_code = {}
        for chain in chains:
            chain_id = chain.get('_id')
            sample_code = chain.get('sample', {}).get('code')
            if chain_id and sample_code:
                chain_to_code[chain_id] = sample_code

        return chain_to_code
    except Exception as e:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def get_renach_data_cached(start_date: datetime = None, end_date: datetime = None):
    """
    Pré-carrega dados de RENACH para todas as chains do período selecionado.
    Retorna dict {chain_id: {"renach": bool, "date": datetime}}
    """
    try:
        chains_collection = get_collection("chainofcustodies")
        if start_date is None or end_date is None:
            start_date, end_date = get_default_period()

        chains = list(chains_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"_id": 1, "analysisStatus.renachEnable": 1, "createdAt": 1}
        ))

        renach_data = {}
        for chain in chains:
            chain_id = chain.get('_id')
            analysis_status = chain.get('analysisStatus', {})
            renach_enable = analysis_status.get('renachEnable', False)
            created_at = chain.get('createdAt')

            renach_data[chain_id] = {
                "renach": renach_enable,
                "date": created_at
            }

        return renach_data
    except Exception as e:
        return {}


def preload_all_data():
    """
    Pré-carrega todos os dados necessários em paralelo.
    Chamado uma vez no início para popular o cache.
    Usa o período selecionado pelo usuário.
    """
    # Obter período selecionado
    start_date, end_date = get_selected_period()

    # Funções com período
    def load_lots():
        return get_all_lots(start_date, end_date)

    def load_results():
        return get_all_results(start_date, end_date)

    def load_gatherings():
        return get_all_gatherings(start_date, end_date)

    def load_chain_map():
        return get_chain_to_sample_map(start_date, end_date)

    def load_renach():
        return get_renach_data_cached(start_date, end_date)

    tasks = {
        "lots": load_lots,
        "results": load_results,
        "gatherings": load_gatherings,
        "chain_map": load_chain_map,
        "renach": load_renach,
        "compounds": get_compounds_map,
        "labs_map": get_laboratories_map,
        "labs_address": get_laboratories_with_address,
    }
    return run_parallel_tasks(tasks, max_workers=8)


# ============================================
# COMPOUNDS - Busca do banco de dados
# ============================================
@st.cache_data(ttl=3600)
def get_compounds_map() -> dict:
    """
    Busca todas as substâncias da collection compounds
    Retorna um dicionário {_id: name}
    """
    try:
        compounds_collection = get_collection("compounds")
        compounds = list(compounds_collection.find({}, {"_id": 1, "name": 1}))

        compounds_map = {}
        for compound in compounds:
            compound_id = compound.get('_id')
            if isinstance(compound_id, ObjectId):
                compound_id = str(compound_id)
            compounds_map[compound_id] = compound.get('name', 'Desconhecido')

        return compounds_map
    except Exception as e:
        st.error(f"Erro ao buscar compounds: {e}")
        return {}


@st.cache_data(ttl=3600)
def get_laboratories_map() -> dict:
    """
    Busca todos os laboratórios da collection laboratories
    Retorna um dicionário {_id: fantasyName}
    """
    try:
        laboratories_collection = get_collection("laboratories")
        laboratories = list(laboratories_collection.find({}, {"_id": 1, "fantasyName": 1, "legalName": 1}))

        laboratories_map = {}
        for lab in laboratories:
            lab_id = lab.get('_id')
            if isinstance(lab_id, ObjectId):
                lab_id = str(lab_id)
            # Usar fantasyName, ou legalName como fallback
            name = lab.get('fantasyName') or lab.get('legalName') or 'Desconhecido'
            laboratories_map[lab_id] = name

        return laboratories_map
    except Exception as e:
        st.error(f"Erro ao buscar laboratories: {e}")
        return {}


@st.cache_data(ttl=3600)
def get_laboratories_with_address() -> list:
    """
    Busca todos os laboratórios com CNPJ e endereço completo.
    Retorna uma lista de dicionários com:
    - id: ObjectId do laboratório
    - name: fantasyName ou legalName
    - cnpj: CNPJ do laboratório
    - city: cidade (de address.city)
    - state: estado (de address.state)
    """
    try:
        laboratories_collection = get_collection("laboratories")
        laboratories = list(laboratories_collection.find(
            {},
            {"_id": 1, "fantasyName": 1, "legalName": 1, "cnpj": 1, "address": 1}
        ))

        labs_list = []
        for lab in laboratories:
            lab_id = lab.get('_id')
            if isinstance(lab_id, ObjectId):
                lab_id = str(lab_id)

            name = lab.get('fantasyName') or lab.get('legalName') or 'Desconhecido'
            cnpj = lab.get('cnpj', '')
            address = lab.get('address', {})

            # Garantir que city e state são strings (não dicionários)
            city = address.get('city', '') if address else ''
            state = address.get('state', '') if address else ''

            # Se city ou state forem dicionários, extrair valor ou usar string vazia
            if isinstance(city, dict):
                city = city.get('name', '') or ''
            if isinstance(state, dict):
                state = state.get('name', '') or state.get('uf', '') or ''

            labs_list.append({
                "id": lab_id,
                "name": name,
                "cnpj": cnpj,
                "city": str(city) if city else '',
                "state": str(state) if state else ''
            })

        return labs_list
    except Exception as e:
        st.error(f"Erro ao buscar laboratories: {e}")
        return []


@st.cache_data(ttl=3600)
def get_unique_states() -> list:
    """
    Retorna lista de estados únicos dos laboratórios.
    """
    labs = get_laboratories_with_address()
    states = set()
    for lab in labs:
        if lab.get('state'):
            states.add(lab['state'])
    return sorted(list(states))


@st.cache_data(ttl=3600)
def get_cities_by_state(state: str = None) -> list:
    """
    Retorna lista de cidades únicas dos laboratórios.
    Se state for fornecido, filtra por estado.
    """
    labs = get_laboratories_with_address()
    cities = set()
    for lab in labs:
        if lab.get('city'):
            if state is None or lab.get('state') == state:
                cities.add(lab['city'])
    return sorted(list(cities))


# Mapeamento de finalidades (purpose.type no banco -> nome exibido)
PURPOSE_MAP = {
    "clt": "CLT",
    "cltCnh": "CLT + CNH",
    "civilService": "Concurso Público",
}


def get_chain_to_sample_code_map(chain_ids: set) -> dict:
    """
    Busca o mapeamento de chainofcustody._id (ObjectId) para sample.code (número).
    USA CACHE PRÉ-CARREGADO para performance.
    """
    if not chain_ids:
        return {}

    try:
        # Usar dados pré-carregados
        full_map = get_chain_to_sample_map()

        # Filtrar apenas os chain_ids solicitados
        return {cid: code for cid, code in full_map.items() if cid in chain_ids}
    except Exception as e:
        st.error(f"Erro ao buscar mapeamento chain->sample.code: {e}")
        return {}


@st.cache_data(ttl=300)
def get_filtered_samples(laboratory_id: str = None, purpose_type: str = None) -> tuple:
    """
    Busca os IDs das amostras filtrados por laboratório e/ou finalidade.
    USA CACHE PRÉ-CARREGADO para performance.
    """
    if not laboratory_id and not purpose_type:
        return None, None

    try:
        # Usar dados pré-carregados com período selecionado
        start_date, end_date = get_selected_period()
        gatherings_data = get_all_gatherings(start_date, end_date)
        chain_to_lab = gatherings_data.get("chain_to_lab", {})
        chain_to_purpose = gatherings_data.get("chain_to_purpose", {})
        chain_to_code = get_chain_to_sample_map(start_date, end_date)

        chain_ids = set()

        for chain_id, lab_id in chain_to_lab.items():
            # Filtro de laboratório
            if laboratory_id and lab_id != laboratory_id:
                continue

            # Filtro de finalidade
            if purpose_type and chain_to_purpose.get(chain_id) != purpose_type:
                continue

            chain_ids.add(chain_id)

        if not chain_ids:
            return set(), set()

        # Buscar sample codes
        sample_codes = {chain_to_code[cid] for cid in chain_ids if cid in chain_to_code}

        return chain_ids, sample_codes

    except Exception as e:
        st.error(f"Erro ao buscar amostras filtradas: {e}")
        return set(), set()


def get_filtered_samples_advanced(
    laboratory_ids: list = None,
    purpose_type: str = None,
    renach_status: str = None,
    state: str = None,
    city: str = None,
    start_date: datetime = None,
    end_date: datetime = None
) -> tuple:
    """
    Busca amostras com filtros avançados:
    - laboratory_ids: lista de IDs de laboratórios (seleção múltipla)
    - purpose_type: tipo de finalidade (periodic, categoryChange, hiring, renovation, resignation)
    - renach_status: "sim" ou "nao"
    - state: estado do laboratório
    - city: cidade do laboratório
    - start_date/end_date: período de análise

    Retorna tupla (chain_ids, sample_codes) ou (None, None) se nenhum filtro.
    - chain_ids: ObjectIds das chainofcustodies (para lots._samples)
    - sample_codes: códigos numéricos das amostras (para results.samples._sample)
    Usa cache de sessão para evitar recarregamentos.
    """
    # Se nenhum filtro, retorna None
    has_filter = any([laboratory_ids, purpose_type, renach_status, state, city])
    if not has_filter:
        return None, None

    # Verificar cache de sessão
    cache_key = generate_cache_key(
        "filtered_samples_advanced", laboratory_ids, purpose_type,
        renach_status, state, city, start_date, end_date
    )
    cached = get_cached_data("filtered_samples_advanced", cache_key)
    if cached is not None:
        return cached

    try:
        # Se tiver filtro geográfico mas não de laboratório, buscar labs pelo endereço
        effective_lab_ids = laboratory_ids

        if (state or city) and not laboratory_ids:
            labs = get_laboratories_with_address()
            filtered_labs = []
            for lab in labs:
                if state and lab.get('state') != state:
                    continue
                if city and lab.get('city') != city:
                    continue
                filtered_labs.append(lab['id'])

            if filtered_labs:
                effective_lab_ids = filtered_labs
            else:
                return set(), set()  # Nenhum lab encontrado com esse filtro

        # Buscar gatherings com filtros
        gatherings_collection = get_collection("gatherings")
        query = {}

        if effective_lab_ids:
            if len(effective_lab_ids) == 1:
                query["_laboratory"] = ObjectId(effective_lab_ids[0])
            else:
                query["_laboratory"] = {"$in": [ObjectId(lid) for lid in effective_lab_ids]}

        if purpose_type:
            query["purpose.type"] = purpose_type

        gatherings = list(gatherings_collection.find(query, {"_chainOfCustody": 1}))

        if not gatherings:
            return set(), set()

        chain_ids = set()
        for g in gatherings:
            chain_id = g.get('_chainOfCustody')
            if chain_id:
                chain_ids.add(chain_id)

        # Filtrar por RENACH se necessário
        if renach_status is not None and chain_ids:
            chains_collection = get_collection("chainofcustodies")
            filtered_chain_ids = set()

            # Processar em lotes para evitar limite de BSON
            chain_ids_list = list(chain_ids)
            batch_size = 5000

            for i in range(0, len(chain_ids_list), batch_size):
                batch = chain_ids_list[i:i + batch_size]

                renach_query = {
                    "_id": {"$in": batch},
                    "analysisStatus.renachEnable": renach_status == "sim"
                }

                # Adicionar filtro de período se fornecido
                if start_date and end_date:
                    renach_query["createdAt"] = {"$gte": start_date, "$lte": end_date}

                renach_chains = list(chains_collection.find(renach_query, {"_id": 1}))
                filtered_chain_ids.update(c["_id"] for c in renach_chains)

            chain_ids = filtered_chain_ids

        if not chain_ids:
            result = (set(), set())
            set_cached_data("filtered_samples_advanced", cache_key, result)
            return result

        # Buscar os sample.code correspondentes a cada chainofcustody
        chain_to_code_map = get_chain_to_sample_code_map(chain_ids)

        # Criar set de sample.codes
        sample_codes = set(chain_to_code_map.values())

        # Salvar no cache de sessão
        result = (chain_ids, sample_codes)
        set_cached_data("filtered_samples_advanced", cache_key, result)
        return result

    except Exception as e:
        st.error(f"Erro ao buscar amostras filtradas: {e}")
        return set(), set()


# ============================================
# CSS
# ============================================
GLOBAL_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #E8E8E8;
    }
</style>
"""


# ============================================
# MAIN
# ============================================

def main():
    """Função principal"""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Inicializar sistema de cache por sessão
    init_session_cache()

    # Autenticação
    try:
        auth = MicrosoftAuth()
    except Exception as e:
        st.error(f"Erro ao inicializar autenticacao: {e}")
        st.stop()

    if not create_login_page(auth):
        st.stop()

    AuthManager.check_and_refresh_token(auth)
    create_user_header()

    # Carregar dados base na primeira execução
    load_base_data()

    # Sidebar - Navegação
    with st.sidebar:
        st.markdown("---")

        if "pagina_atual" not in st.session_state:
            st.session_state.pagina_atual = "🏠 Visão Geral"

        paginas = [
            "🏠 Visão Geral",
            "🧪 Substâncias",
            "🗺️ Mapa Geográfico",
            "📈 Análise Temporal",
            "📋 Tabela Detalhada",
            "🔍 Auditoria",
            "🏢 Rede"
        ]

        for pag in paginas:
            tipo_botao = "primary" if st.session_state.pagina_atual == pag else "secondary"
            if st.button(pag, key=f"nav_{pag}", use_container_width=True, type=tipo_botao):
                st.session_state.pagina_anterior = st.session_state.pagina_atual
                st.session_state.pagina_atual = pag
                st.session_state.carregando_pagina = True
                st.rerun()

        pagina = st.session_state.pagina_atual

        # Inicializar variáveis de controle de carregamento
        if 'carregando_pagina' not in st.session_state:
            st.session_state.carregando_pagina = False
        if 'pagina_anterior' not in st.session_state:
            st.session_state.pagina_anterior = None

        st.markdown("---")

        # ========== FILTRO DE PERÍODO GLOBAL ==========
        st.markdown("### 📅 Período")
        init_period_session()

        periodo_opcoes = ["30 dias", "60 dias", "90 dias", "Personalizado"]

        periodo_selecionado = st.radio(
            "Selecione o período:",
            periodo_opcoes,
            index=periodo_opcoes.index(st.session_state.periodo_tipo) if st.session_state.periodo_tipo in periodo_opcoes else 0,
            key="radio_periodo",
            horizontal=False
        )

        # Verificar se mudou o tipo de período
        periodo_mudou = False
        if periodo_selecionado != st.session_state.periodo_tipo:
            st.session_state.periodo_tipo = periodo_selecionado
            periodo_mudou = True

        if periodo_selecionado == "Personalizado":
            col_dt1, col_dt2 = st.columns(2)
            with col_dt1:
                data_inicio = st.date_input(
                    "Início",
                    value=st.session_state.periodo_inicio if isinstance(st.session_state.periodo_inicio, date) else st.session_state.periodo_inicio.date(),
                    key="input_data_inicio",
                    format="DD/MM/YYYY"
                )
            with col_dt2:
                data_fim = st.date_input(
                    "Fim",
                    value=st.session_state.periodo_fim if isinstance(st.session_state.periodo_fim, date) else st.session_state.periodo_fim.date(),
                    key="input_data_fim",
                    format="DD/MM/YYYY"
                )

            # Atualizar sessão com datas personalizadas
            new_start = datetime.combine(data_inicio, datetime.min.time())
            new_end = datetime.combine(data_fim, datetime.max.time().replace(microsecond=0))

            if new_start != st.session_state.periodo_inicio or new_end != st.session_state.periodo_fim:
                st.session_state.periodo_inicio = new_start
                st.session_state.periodo_fim = new_end
                periodo_mudou = True
        else:
            # Calcular período baseado nos dias selecionados
            dias = int(periodo_selecionado.split()[0])
            new_start, new_end = get_period_by_days(dias)

            if periodo_mudou:
                st.session_state.periodo_inicio = new_start
                st.session_state.periodo_fim = new_end

        # Mostrar período atual
        periodo_inicio, periodo_fim = get_selected_period()
        st.caption(f"📆 {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')}")

        # Limpar cache se período mudou
        if periodo_mudou:
            clear_cache()
            st.session_state.base_data_loaded = False
            st.rerun()

        st.markdown("---")

        with st.expander("⚙️ Status do Sistema", expanded=False):
            if test_connection():
                st.success("MongoDB conectado")
            else:
                st.error("MongoDB desconectado")

            # Botão para limpar cache e recarregar dados
            if st.button("Atualizar Dados", key="btn_refresh_cache", use_container_width=True):
                clear_cache()
                st.session_state.base_data_loaded = False
                st.rerun()

    # Roteamento com tela de carregamento
    if st.session_state.get('carregando_pagina', False):
        # Mostrar card de carregamento
        st.markdown("""
            <style>
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                .loading-spinner {
                    animation: pulse 1.5s ease-in-out infinite;
                }
            </style>
            <div style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 70vh;
            ">
                <div style="
                    background: white;
                    border-radius: 16px;
                    padding: 50px 80px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                    text-align: center;
                    border: 1px solid #e0e0e0;
                ">
                    <div class="loading-spinner" style="font-size: 60px; margin-bottom: 25px;">
                        🔄
                    </div>
                    <h2 style="
                        color: #1A1A2E;
                        margin: 0 0 15px 0;
                        font-size: 24px;
                        font-weight: 600;
                    ">
                        Carregando dados...
                    </h2>
                    <p style="
                        color: #666;
                        margin: 0;
                        font-size: 16px;
                        line-height: 1.5;
                    ">
                        Isso pode levar alguns minutos.<br>
                        Por favor, aguarde.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Resetar flag de carregamento após renderizar
        st.session_state.carregando_pagina = False
        st.rerun()
    else:
        # Renderizar página normalmente
        if pagina == "🏠 Visão Geral":
            render_visao_geral()
        elif pagina == "🧪 Substâncias":
            render_substancias()
        elif pagina == "🗺️ Mapa Geográfico":
            render_mapa()
        elif pagina == "📈 Análise Temporal":
            render_temporal()
        elif pagina == "📋 Tabela Detalhada":
            render_tabela_detalhada()
        elif pagina == "🔍 Auditoria":
            render_auditoria()
        elif pagina == "🏢 Rede":
            render_rede()


# ============================================
# PÁGINAS
# ============================================

def render_tabela_substancias():
    """
    Tabela de Substâncias - Amostras do período selecionado
    """


    with st.spinner("Carregando dados..."):
        df = get_substance_data()

    if df.empty:
        st.warning("⚠️ Nenhum dado encontrado")
        return

    # Colunas de substâncias (excluindo colunas de metadados)
    substance_cols = [col for col in df.columns if col not in ['Data', 'Lote', 'Tipo de Lote', 'Amostra']]

    # ============================================
    # ESTATÍSTICAS (do dataframe total)
    # ============================================
    st.markdown("### 📊 Estatísticas")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Amostras", f"{len(df):,}")

    with col2:
        total_lotes = df['Lote'].nunique()
        st.metric("Total de Lotes", f"{total_lotes:,}")

    with col3:
        amostras_positivas = df[df[substance_cols].apply(lambda row: (row == 'Positivo').any(), axis=1)].shape[0]
        st.metric("Amostras Positivas", f"{amostras_positivas:,}")

    with col4:
        taxa = (amostras_positivas / len(df) * 100) if len(df) > 0 else 0
        st.metric("Taxa de Positividade", f"{taxa:.2f}%")

    st.markdown("---")

    # ============================================
    # FILTROS
    # ============================================
    st.markdown("### 🔍 Filtros")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Filtro por Lote
        filtro_lote = st.text_input("Lote", placeholder="Digite o código do lote...")

    with col2:
        # Filtro por Código de Amostra
        filtro_amostra = st.text_input("Código da Amostra", placeholder="Digite o código...")

    with col3:
        # Filtro Substâncias Positivas (Sim/Não)
        filtro_positivas = st.selectbox("Substâncias Positivas", ["Todos", "Sim", "Não"])

    with col4:
        # Filtro por Substância Específica Positiva
        compounds_map = get_compounds_map()
        substancias_opcoes = ['Todas'] + list(compounds_map.values())
        filtro_substancia = st.selectbox("Buscar Substância Positiva", substancias_opcoes)

    # Aplicar filtros
    df_filtrado = df.copy()

    # Filtro por Lote
    if filtro_lote:
        df_filtrado = df_filtrado[df_filtrado['Lote'].str.contains(filtro_lote, case=False, na=False)]

    # Filtro por Código de Amostra
    if filtro_amostra:
        df_filtrado = df_filtrado[df_filtrado['Amostra'].str.contains(filtro_amostra, case=False, na=False)]

    # Filtro Substâncias Positivas
    if filtro_positivas == 'Sim':
        # Amostras com pelo menos uma substância positiva
        df_filtrado = df_filtrado[df_filtrado[substance_cols].apply(lambda row: (row == 'Positivo').any(), axis=1)]
    elif filtro_positivas == 'Não':
        # Amostras com todas substâncias negativas
        df_filtrado = df_filtrado[df_filtrado[substance_cols].apply(lambda row: (row == 'Negativo').all(), axis=1)]

    # Filtro por Substância Específica Positiva
    if filtro_substancia != 'Todas':
        df_filtrado = df_filtrado[df_filtrado[filtro_substancia] == 'Positivo']

    st.markdown("---")

    # ============================================
    # TABELA
    # ============================================
    st.markdown("### 📋 Dados")
    st.dataframe(df_filtrado, use_container_width=True, hide_index=True, height=600)

    # Controles abaixo da tabela
    col1, col2 = st.columns([2, 1])

    with col1:
        st.caption(f"Total: {len(df_filtrado):,} registros")

    with col2:
        csv = df_filtrado.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("⬇️ Download CSV", csv, "substancias.csv", "text/csv", use_container_width=True)


def get_substance_data() -> pd.DataFrame:
    """
    Busca dados de substâncias - amostras do período selecionado
    Retorna uma linha por amostra, com uma coluna para cada substância
    Usa cache de sessão para evitar recarregamentos.
    """
    # Obter período selecionado
    start_date, end_date = get_selected_period()

    # Verificar cache de sessão (incluindo período na chave)
    cache_key = generate_cache_key("substance_data", start_date, end_date)
    cached = get_cached_data("substance_data", cache_key)
    if cached is not None:
        return cached

    try:
        # 1. Buscar lotes do período (incluindo tipo de análise)
        lots_collection = get_collection("lots")

        lots_period = list(lots_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"code": 1, "analysisType": 1}
        ))

        if not lots_period:
            return pd.DataFrame()

        # 2. Extrair códigos dos lotes e criar mapeamento para tipo de análise
        lot_codes = []
        lot_type_map = {}
        analysis_type_names = {
            "screening": "Triagem",
            "confirmatory": "Confirmatório",
            "confirmatoryTHC": "Confirmatório THC"
        }

        for lot in lots_period:
            code = lot.get('code')
            if code:
                lot_codes.append(code)
                analysis_type = lot.get('analysisType', '')
                lot_type_map[code] = analysis_type_names.get(analysis_type, analysis_type or 'N/A')

        # 3. Buscar mapeamento de compounds do banco (fazer antes para usar no processamento)
        compounds_map = get_compounds_map()

        # 4. Função para processar um lote de results
        def process_substance_batch(batch_lot_codes):
            try:
                client = get_mongo_client()
                db = client["ctox"]
                results_collection = db["results"]

                results = list(results_collection.find(
                    {"_lot": {"$in": batch_lot_codes}},
                    {"_lot": 1, "samples": 1, "createdAt": 1}
                ))

                batch_rows = []
                for result in results:
                    lot_code = result.get('_lot', 'N/A')
                    created_at = result.get('createdAt')

                    # Converter UTC para UTC-3 (horário de Brasília)
                    if created_at:
                        created_at_brt = created_at - timedelta(hours=3)
                        data_str = created_at_brt.strftime('%d/%m/%Y')
                    else:
                        data_str = 'N/A'

                    # Obter tipo de lote
                    tipo_lote = lot_type_map.get(lot_code, 'N/A')

                    for sample in result.get('samples', []):
                        sample_code = sample.get('_sample', 'N/A')

                        # Criar linha base
                        row = {
                            'Data': data_str,
                            'Lote': lot_code,
                            'Tipo de Lote': tipo_lote,
                            'Amostra': sample_code
                        }

                        # Inicializar todas as substâncias como Negativo
                        for substance_name in compounds_map.values():
                            row[substance_name] = 'Negativo'

                        # Preencher resultados das substâncias
                        for compound in sample.get('compounds', []):
                            compound_id = compound.get('_compound')
                            is_positive = compound.get('positive', False)

                            # Converter ObjectId para string
                            if isinstance(compound_id, ObjectId):
                                compound_id_str = str(compound_id)
                            else:
                                compound_id_str = str(compound_id) if compound_id else None

                            # Nome da substância (busca no mapeamento do banco)
                            substance_name = compounds_map.get(compound_id_str)
                            if substance_name:
                                row[substance_name] = 'Positivo' if is_positive else 'Negativo'

                        batch_rows.append(row)

                return batch_rows
            except Exception as e:
                return []

        # 5. Dividir em lotes para processamento paralelo
        batch_size = 500
        batches = [lot_codes[i:i + batch_size] for i in range(0, len(lot_codes), batch_size)]

        all_rows = []

        # Processar lotes em paralelo
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {executor.submit(process_substance_batch, batch): batch for batch in batches}

            for future in as_completed(future_to_batch):
                try:
                    batch_rows = future.result()
                    if batch_rows:
                        all_rows.extend(batch_rows)
                except Exception as e:
                    pass

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # Salvar no cache de sessão
        set_cached_data("substance_data", cache_key, df)

        return df

    except Exception as e:
        st.error(f"Erro: {e}")
        return pd.DataFrame()


def count_results_by_type(analysis_type: str, laboratory_id: str = None, month: int = None, purpose_type: str = None) -> dict:
    """
    Função otimizada que usa dados pré-carregados para contar resultados.
    Funciona para qualquer tipo de análise: screening, confirmatory, confirmatoryTHC.
    """
    cache_key = generate_cache_key(f"count_{analysis_type}", laboratory_id, month, purpose_type)
    cached = get_cached_data(f"count_{analysis_type}", cache_key)
    if cached is not None:
        return cached

    try:
        # Usar dados pré-carregados com período selecionado
        start_date, end_date = get_selected_period()
        lots_dict = get_all_lots(start_date, end_date)
        results_dict = get_all_results(start_date, end_date)

        # Filtrar amostras se necessário
        allowed_chain_ids, allowed_sample_codes = get_filtered_samples(laboratory_id, purpose_type)

        positivo = 0
        negativo = 0

        for lot_code, lot_data in lots_dict.items():
            # Filtrar por tipo de análise
            if lot_data["analysisType"] != analysis_type:
                continue

            # Filtrar por mês
            if month and lot_data.get("month") != month:
                continue

            # Filtrar por laboratório/finalidade
            if allowed_chain_ids is not None:
                lot_samples = lot_data.get("_samples", set())
                if not lot_samples.intersection(allowed_chain_ids):
                    continue

            # Contar resultados deste lote
            lot_results = results_dict.get(lot_code, [])
            for sample in lot_results:
                sample_code = sample.get("_sample")

                # Filtrar por sample_code se necessário
                if allowed_sample_codes is not None and sample_code not in allowed_sample_codes:
                    continue

                if sample.get("positive", False):
                    positivo += 1
                else:
                    negativo += 1

        result_data = {"positivo": positivo, "negativo": negativo}
        set_cached_data(f"count_{analysis_type}", cache_key, result_data)
        return result_data

    except Exception as e:
        st.error(f"Erro ao contar resultados {analysis_type}: {e}")
        return {"positivo": 0, "negativo": 0}


def get_triagem_data(laboratory_id: str = None, month: int = None, purpose_type: str = None) -> dict:
    """
    Busca dados de Triagem (screening) - USA DADOS PRÉ-CARREGADOS.
    """
    return count_results_by_type("screening", laboratory_id, month, purpose_type)


def get_confirmatorio_data(laboratory_id: str = None, month: int = None, purpose_type: str = None) -> dict:
    """
    Busca dados de Confirmatório (confirmatory) - USA DADOS PRÉ-CARREGADOS.
    """
    return count_results_by_type("confirmatory", laboratory_id, month, purpose_type)


def get_confirmatorio_thc_data(laboratory_id: str = None, month: int = None, purpose_type: str = None) -> dict:
    """
    Busca dados de Confirmatório THC (confirmatoryTHC) - USA DADOS PRÉ-CARREGADOS.
    """
    return count_results_by_type("confirmatoryTHC", laboratory_id, month, purpose_type)


def get_total_samples(laboratory_id: str = None, month: int = None, purpose_type: str = None) -> int:
    """
    Conta o total de amostras em chainofcustodies aplicando os filtros.
    Padrão: últimos 30 dias.
    """
    try:
        # Usar período padrão (últimos 30 dias)
        start_date, end_date = get_default_period()

        chains_collection = get_collection("chainofcustodies")

        # Se tiver filtros de laboratório ou finalidade
        # Usar apenas chain_ids (ObjectIds) para filtrar chainofcustodies
        allowed_chain_ids, _ = get_filtered_samples(laboratory_id, purpose_type)

        if allowed_chain_ids is not None:
            # Processar em lotes para evitar limite de BSON
            chain_ids_list = list(allowed_chain_ids)
            batch_size = 5000
            total_count = 0

            for i in range(0, len(chain_ids_list), batch_size):
                batch = chain_ids_list[i:i + batch_size]
                query = {
                    "createdAt": {"$gte": start_date, "$lte": end_date},
                    "_id": {"$in": batch}
                }
                total_count += chains_collection.count_documents(query)

            return total_count
        else:
            query = {
                "createdAt": {"$gte": start_date, "$lte": end_date}
            }
            return chains_collection.count_documents(query)

    except Exception as e:
        st.error(f"Erro ao contar amostras: {e}")
        return 0


def get_renach_data(laboratory_id: str = None, month: int = None, purpose_type: str = None) -> dict:
    """
    Busca dados de RENACH - USA DADOS PRÉ-CARREGADOS (últimos 30 dias).
    """
    cache_key = generate_cache_key("renach", laboratory_id, month, purpose_type)
    cached = get_cached_data("renach_data", cache_key)
    if cached is not None:
        return cached

    try:
        # Usar dados pré-carregados com período selecionado
        start_date, end_date = get_selected_period()
        renach_all = get_renach_data_cached(start_date, end_date)
        allowed_chain_ids, _ = get_filtered_samples(laboratory_id, purpose_type)

        # Filtrar por mês se especificado
        # Calcular período do mês se especificado
        month_start = None
        month_end = None
        if month:
            year = datetime.now().year
            month_start = datetime(year, month, 1)
            if month == 12:
                month_end = datetime(year, 12, 31, 23, 59, 59)
            else:
                month_end = datetime(year, month + 1, 1) - timedelta(seconds=1)

        no_renach = 0
        fora_renach = 0

        for chain_id, data in renach_all.items():
            # Filtrar por laboratório/finalidade
            if allowed_chain_ids is not None and chain_id not in allowed_chain_ids:
                continue

            # Filtrar por mês se especificado
            if month_start and month_end:
                chain_date = data.get("date")
                if chain_date:
                    if chain_date < month_start or chain_date > month_end:
                        continue

            if data.get("renach", False):
                no_renach += 1
            else:
                fora_renach += 1

        result_data = {"no_renach": no_renach, "fora_renach": fora_renach}
        set_cached_data("renach_data", cache_key, result_data)
        return result_data

    except Exception as e:
        st.error(f"Erro ao buscar dados de RENACH: {e}")
        return {"no_renach": 0, "fora_renach": 0}


def get_samples_by_purpose(laboratory_id: str = None, month: int = None) -> dict:
    """
    Busca contagem de amostras por finalidade - USA DADOS PRÉ-CARREGADOS (últimos 30 dias).
    """
    cache_key = generate_cache_key("samples_purpose", laboratory_id, month)
    cached = get_cached_data("samples_purpose_data", cache_key)
    if cached is not None:
        return cached

    try:
        # Usar dados pré-carregados com período selecionado
        start_date, end_date = get_selected_period()
        gatherings_data = get_all_gatherings(start_date, end_date)
        chain_to_lab = gatherings_data.get("chain_to_lab", {})
        chain_to_purpose = gatherings_data.get("chain_to_purpose", {})
        chain_to_date = gatherings_data.get("chain_to_date", {})

        # Calcular período do mês se especificado
        month_start = None
        month_end = None
        if month:
            year = datetime.now().year
            month_start = datetime(year, month, 1)
            if month == 12:
                month_end = datetime(year, 12, 31, 23, 59, 59)
            else:
                month_end = datetime(year, month + 1, 1) - timedelta(seconds=1)

        # Mapear tipos de finalidade para nomes legíveis
        purpose_names = {
            "clt": "CLT",
            "cltCnh": "CLT + CNH",
            "civilService": "Concurso Público",
            "periodic": "Periódico",
            "categoryChange": "Mudança de Categoria",
            "hiring": "Admissão",
            "renovation": "Renovação",
            "resignation": "Demissão"
        }

        purpose_counts = {}

        for chain_id, purpose_type in chain_to_purpose.items():
            # Filtrar por laboratório
            if laboratory_id:
                lab_id = chain_to_lab.get(chain_id)
                if lab_id != laboratory_id:
                    continue

            # Filtrar por mês se especificado
            if month_start and month_end:
                chain_date = chain_to_date.get(chain_id)
                if chain_date:
                    if chain_date < month_start or chain_date > month_end:
                        continue

            purpose_name = purpose_names.get(purpose_type, purpose_type or 'Não informado')

            if purpose_name not in purpose_counts:
                purpose_counts[purpose_name] = 0
            purpose_counts[purpose_name] += 1

        set_cached_data("samples_purpose_data", cache_key, purpose_counts)
        return purpose_counts

    except Exception as e:
        st.error(f"Erro ao buscar dados por finalidade: {e}")
        return {}


def render_visao_geral():
    # Placeholder para título (será preenchido após carregar dados)
    titulo_placeholder = st.empty()

    # Mostrar período selecionado
    periodo_inicio, periodo_fim = get_selected_period()
    st.info(f"📅 **Período:** {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')} *(altere na sidebar)*")

    # Filtros
    st.markdown("### 🔍 Filtros")

    col_filtro1, col_filtro2, col_filtro3 = st.columns(3)

    # Filtro de Mês
    with col_filtro1:
        meses = {
            "Todos": None,
            "Janeiro": 1,
            "Fevereiro": 2,
            "Março": 3,
            "Abril": 4,
            "Maio": 5,
            "Junho": 6,
            "Julho": 7,
            "Agosto": 8,
            "Setembro": 9,
            "Outubro": 10,
            "Novembro": 11,
            "Dezembro": 12
        }
        selected_mes_name = st.selectbox(
            "Mês",
            options=list(meses.keys()),
            index=0
        )
        selected_month = meses[selected_mes_name]

    # Filtro de Finalidade
    with col_filtro2:
        finalidades = {
            "Todas": None,
            "CLT": "clt",
            "CLT + CNH": "cltCnh",
            "Concurso Público": "civilService",
        }
        selected_finalidade_name = st.selectbox(
            "Finalidade da Amostra",
            options=list(finalidades.keys()),
            index=0
        )
        selected_purpose = finalidades[selected_finalidade_name]

    # Filtro de Laboratório
    with col_filtro3:
        laboratories_map = get_laboratories_map()
        lab_options = {"Todos": None}
        lab_options.update({name: lab_id for lab_id, name in laboratories_map.items()})

        selected_lab_name = st.selectbox(
            "Laboratório (PCL)",
            options=list(lab_options.keys()),
            index=0
        )
        selected_lab_id = lab_options[selected_lab_name]

    st.markdown("---")

    with st.spinner("Carregando dados..."):
        triagem_data = get_triagem_data(selected_lab_id, selected_month, selected_purpose)
        confirmatorio_data = get_confirmatorio_data(selected_lab_id, selected_month, selected_purpose)
        confirmatorio_thc_data = get_confirmatorio_thc_data(selected_lab_id, selected_month, selected_purpose)
        renach_data = get_renach_data(selected_lab_id, selected_month, selected_purpose)
        purpose_data = get_samples_by_purpose(selected_lab_id, selected_month)

    # Contagem de amostras (soma de todos os tipos de análise)
    total_amostras = (
        triagem_data["positivo"] + triagem_data["negativo"] +
        confirmatorio_data["positivo"] + confirmatorio_data["negativo"] +
        confirmatorio_thc_data["positivo"] + confirmatorio_thc_data["negativo"]
    )

    # Preencher título com contagem ao lado
    with titulo_placeholder.container():
        col_titulo, col_contagem = st.columns([3, 1])
        with col_titulo:
            st.title("🏠 Visão Geral")
        with col_contagem:
            st.metric("Amostras", f"{total_amostras:,}")


    # Grid 2x2 para os gráficos
    col1, col2 = st.columns(2)

    # Gráfico de Triagem (canto superior esquerdo)
    with col1:
        st.subheader("Triagem")

        total_triagem = triagem_data["positivo"] + triagem_data["negativo"]

        if total_triagem > 0:
            # Calcular percentuais
            pct_negativo = (triagem_data["negativo"] / total_triagem) * 100
            pct_positivo = (triagem_data["positivo"] / total_triagem) * 100

            # Criar dataframe para o gráfico
            df_triagem = pd.DataFrame({
                'Status': ['Negativo', 'Positivo'],
                'Quantidade': [triagem_data["negativo"], triagem_data["positivo"]],
                'Percentual': [pct_negativo, pct_positivo]
            })

            # Criar gráfico de pizza
            fig = px.pie(
                df_triagem,
                values='Quantidade',
                names='Status',
                color='Status',
                color_discrete_map={'Negativo': '#00CED1', 'Positivo': '#1A1A2E'},
                hole=0.4
            )

            fig.update_traces(
                textposition='inside',
                textinfo='value+percent',
                texttemplate='%{value:,.0f}<br>(%{percent:.2%})',
                textfont_size=12
            )

            fig.update_layout(
                showlegend=True,
                legend_title_text='Substâncias Detectadas',
                height=400,
                margin=dict(t=30, b=30, l=50, r=50)
            )

            st.plotly_chart(fig, use_container_width=True, key="chart_visao_triagem_pizza")
        else:
            st.warning("Nenhum dado de triagem encontrado")

    # Gráfico de Quantidade de Amostras por Status de Positividade (soma de todos os tipos)
    with col2:
        st.subheader("Amostras por Resultado")

        # Somar todos os tipos de análises
        total_positivo = triagem_data["positivo"] + confirmatorio_data["positivo"] + confirmatorio_thc_data["positivo"]
        total_negativo = triagem_data["negativo"] + confirmatorio_data["negativo"] + confirmatorio_thc_data["negativo"]
        total_geral = total_positivo + total_negativo

        if total_geral > 0:
            # Calcular percentuais
            pct_negativo = (total_negativo / total_geral) * 100
            pct_positivo = (total_positivo / total_geral) * 100

            # Criar dataframe para o gráfico
            df_total = pd.DataFrame({
                'Status': ['Negativo', 'Positivo'],
                'Quantidade': [total_negativo, total_positivo],
                'Percentual': [pct_negativo, pct_positivo]
            })

            # Criar gráfico de pizza
            fig_total = px.pie(
                df_total,
                values='Quantidade',
                names='Status',
                color='Status',
                color_discrete_map={'Negativo': '#00CED1', 'Positivo': '#1A1A2E'},
                hole=0.4
            )

            fig_total.update_traces(
                textposition='inside',
                textinfo='value+percent',
                texttemplate='%{value:,.0f}<br>(%{percent:.2%})',
                textfont_size=12
            )

            fig_total.update_layout(
                showlegend=True,
                legend_title_text='Substâncias Detectadas',
                height=400,
                margin=dict(t=30, b=30, l=50, r=50)
            )

            st.plotly_chart(fig_total, use_container_width=True, key="chart_visao_triagem")
        else:
            st.warning("Nenhum dado encontrado")

    col3, col4 = st.columns(2)

    # Gráfico de Confirmatório (canto inferior esquerdo)
    with col3:
        st.subheader("Confirmatório")

        total_confirmatorio = confirmatorio_data["positivo"] + confirmatorio_data["negativo"]

        if total_confirmatorio > 0:
            # Calcular percentuais
            pct_negativo = (confirmatorio_data["negativo"] / total_confirmatorio) * 100
            pct_positivo = (confirmatorio_data["positivo"] / total_confirmatorio) * 100

            # Criar dataframe para o gráfico
            df_confirmatorio = pd.DataFrame({
                'Status': ['Negativo', 'Positivo'],
                'Quantidade': [confirmatorio_data["negativo"], confirmatorio_data["positivo"]],
                'Percentual': [pct_negativo, pct_positivo]
            })

            # Criar gráfico de pizza
            fig_conf = px.pie(
                df_confirmatorio,
                values='Quantidade',
                names='Status',
                color='Status',
                color_discrete_map={'Negativo': '#00CED1', 'Positivo': '#1A1A2E'},
                hole=0.4
            )

            fig_conf.update_traces(
                textposition='inside',
                textinfo='value+percent',
                texttemplate='%{value:,.0f}<br>(%{percent:.2%})',
                textfont_size=12
            )

            fig_conf.update_layout(
                showlegend=True,
                legend_title_text='Substâncias Detectadas',
                height=400,
                margin=dict(t=30, b=30, l=50, r=50)
            )

            st.plotly_chart(fig_conf, use_container_width=True, key="chart_visao_confirmatorio")
        else:
            st.warning("Nenhum dado de confirmatório encontrado")

    # Gráfico de Confirmatório THC (canto inferior direito)
    with col4:
        st.subheader("Confirmatório THC")

        total_confirmatorio_thc = confirmatorio_thc_data["positivo"] + confirmatorio_thc_data["negativo"]

        if total_confirmatorio_thc > 0:
            # Calcular percentuais
            pct_negativo = (confirmatorio_thc_data["negativo"] / total_confirmatorio_thc) * 100
            pct_positivo = (confirmatorio_thc_data["positivo"] / total_confirmatorio_thc) * 100

            # Criar dataframe para o gráfico
            df_confirmatorio_thc = pd.DataFrame({
                'Status': ['Negativo', 'Positivo'],
                'Quantidade': [confirmatorio_thc_data["negativo"], confirmatorio_thc_data["positivo"]],
                'Percentual': [pct_negativo, pct_positivo]
            })

            # Criar gráfico de pizza
            fig_thc = px.pie(
                df_confirmatorio_thc,
                values='Quantidade',
                names='Status',
                color='Status',
                color_discrete_map={'Negativo': '#00CED1', 'Positivo': '#1A1A2E'},
                hole=0.4
            )

            fig_thc.update_traces(
                textposition='inside',
                textinfo='value+percent',
                texttemplate='%{value:,.0f}<br>(%{percent:.2%})',
                textfont_size=12
            )

            fig_thc.update_layout(
                showlegend=True,
                legend_title_text='Substâncias Detectadas',
                height=400,
                margin=dict(t=30, b=30, l=50, r=50)
            )

            st.plotly_chart(fig_thc, use_container_width=True, key="chart_visao_confirmatorio_thc")
        else:
            st.warning("Nenhum dado de confirmatório THC encontrado")

    # Terceira linha - Gráfico de RENACH
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Status RENACH")

        total_renach = renach_data["no_renach"] + renach_data["fora_renach"]

        if total_renach > 0:
            # Calcular percentuais
            pct_no_renach = (renach_data["no_renach"] / total_renach) * 100
            pct_fora_renach = (renach_data["fora_renach"] / total_renach) * 100

            # Criar dataframe para o gráfico
            df_renach = pd.DataFrame({
                'Status': ['No RENACH', 'Fora do RENACH'],
                'Quantidade': [renach_data["no_renach"], renach_data["fora_renach"]],
                'Percentual': [pct_no_renach, pct_fora_renach]
            })

            # Criar gráfico de pizza
            fig_renach = px.pie(
                df_renach,
                values='Quantidade',
                names='Status',
                color='Status',
                color_discrete_map={'No RENACH': '#00CED1', 'Fora do RENACH': '#1A1A2E'},
                hole=0.4
            )

            fig_renach.update_traces(
                textposition='inside',
                textinfo='value+percent',
                texttemplate='%{value:,.0f}<br>(%{percent:.2%})',
                textfont_size=12
            )

            fig_renach.update_layout(
                showlegend=True,
                legend_title_text='Status RENACH',
                height=400,
                margin=dict(t=30, b=30, l=50, r=50)
            )

            st.plotly_chart(fig_renach, use_container_width=True, key="chart_visao_renach")
        else:
            st.warning("Nenhum dado de RENACH encontrado")

    # Gráfico de Amostras por Finalidade
    with col6:
        st.subheader("Amostras por Finalidade")

        if purpose_data:
            # Ordenar por quantidade (decrescente)
            sorted_purposes = sorted(purpose_data.items(), key=lambda x: x[1], reverse=True)
            finalidades = [p[0] for p in sorted_purposes]
            quantidades = [p[1] for p in sorted_purposes]
            total_finalidade = sum(quantidades)

            # Criar dataframe para o gráfico
            df_purpose = pd.DataFrame({
                'Finalidade': finalidades,
                'Quantidade': quantidades
            })

            # Calcular percentuais
            df_purpose['Percentual'] = (df_purpose['Quantidade'] / total_finalidade * 100).round(2)
            df_purpose['Texto'] = df_purpose.apply(
                lambda row: f"{row['Quantidade']:,} ({row['Percentual']:.1f}%)".replace(",", "."), axis=1
            )

            # Criar gráfico de barras horizontais
            fig_purpose = px.bar(
                df_purpose,
                y='Finalidade',
                x='Quantidade',
                orientation='h',
                text='Texto',
                color='Quantidade',
                color_continuous_scale=['#1A1A2E', '#00CED1']
            )

            fig_purpose.update_traces(
                textposition='outside',
                textfont_size=10
            )

            fig_purpose.update_layout(
                showlegend=False,
                height=400,
                margin=dict(t=30, b=30, l=50, r=80),
                xaxis_title="Quantidade",
                yaxis_title="",
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_purpose, use_container_width=True, key="chart_visao_finalidade")
        else:
            st.warning("Nenhum dado de finalidade encontrado")


def get_monthly_positivity_data(
    laboratory_ids: list = None,
    purpose_type: str = None,
    renach_status: str = None,
    analysis_type: str = "screening",
    state: str = None,
    city: str = None,
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
) -> dict:
    """
    Busca dados de positividade por mês.
    Filtros avançados:
    - laboratory_ids: lista de IDs de laboratórios
    - purpose_type: finalidade da amostra
    - renach_status: "sim" ou "nao"
    - analysis_type: screening, confirmatory, confirmatoryTHC ou "all"
    - state/city: localização do laboratório
    - start_date_filter/end_date_filter: período personalizado
    Usa cache de sessão para evitar recarregamentos.
    """
    # Verificar cache de sessão
    cache_key = generate_cache_key(
        "monthly_positivity", laboratory_ids, purpose_type, renach_status,
        analysis_type, state, city, start_date_filter, end_date_filter
    )
    cached = get_cached_data("monthly_positivity_data", cache_key)
    if cached is not None:
        return cached

    try:
        lots_collection = get_collection("lots")
        results_collection = get_collection("results")

        # Definir período (usar filtro de data ou últimos 30 dias por padrão)
        if start_date_filter and end_date_filter:
            year_start = start_date_filter
            year_end = end_date_filter
        else:
            year_start = DEFAULT_START_DATE
            year_end = DEFAULT_END_DATE

        # Buscar amostras filtradas
        # Retorna (chain_ids para lots._samples, sample_codes para results.samples._sample)
        allowed_chain_ids, allowed_sample_codes = get_filtered_samples_advanced(
            laboratory_ids=laboratory_ids,
            purpose_type=purpose_type,
            renach_status=renach_status,
            state=state,
            city=city,
            start_date=year_start,
            end_date=year_end
        )

        monthly_data = {}
        meses_nomes = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
                       "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]

        # Determinar tipos de análise a buscar
        if analysis_type == "all":
            analysis_types = ["screening", "confirmatory", "confirmatoryTHC"]
        else:
            analysis_types = [analysis_type]

        # Iterar pelos meses dentro do período filtrado
        current_date = year_start.replace(day=1)
        end_limit = year_end

        while current_date <= end_limit:
            month = current_date.month
            year = current_date.year
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year, 12, 31, 23, 59, 59)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)

            # Ajustar para não ultrapassar o período filtrado
            if start_date < year_start:
                start_date = year_start
            if end_date > year_end:
                end_date = year_end

            mes_label = f"{meses_nomes[month - 1]}/{year}"

            positivo_total = 0
            negativo_total = 0

            for a_type in analysis_types:
                # Buscar lotes do mês
                lots = list(lots_collection.find(
                    {
                        "analysisType": a_type,
                        "createdAt": {"$gte": start_date, "$lte": end_date}
                    },
                    {"code": 1, "_samples": 1}
                ))

                if not lots:
                    continue

                # Filtrar lotes se necessário
                lot_codes = []

                for lot in lots:
                    lot_code = lot.get('code')
                    lot_samples = lot.get('_samples', [])  # ObjectIds das chainofcustodies

                    if lot_code:
                        if allowed_chain_ids is not None:
                            # Filtrar: só incluir se o lote tem amostras permitidas
                            matching_samples = [s for s in lot_samples if s in allowed_chain_ids]
                            if matching_samples:
                                lot_codes.append(lot_code)
                        else:
                            lot_codes.append(lot_code)

                if not lot_codes:
                    continue

                # Buscar results
                results = list(results_collection.find(
                    {"_lot": {"$in": lot_codes}},
                    {"_lot": 1, "samples._sample": 1, "samples.positive": 1}
                ))

                for result in results:
                    for sample in result.get('samples', []):
                        sample_code = sample.get('_sample')  # Este é o sample.code (número)

                        # Se tiver filtros, verificar se o sample.code está permitido
                        if allowed_sample_codes is not None:
                            if sample_code not in allowed_sample_codes:
                                continue

                        if sample.get('positive', False):
                            positivo_total += 1
                        else:
                            negativo_total += 1

            total = positivo_total + negativo_total
            taxa = (positivo_total / total * 100) if total > 0 else 0.0

            monthly_data[mes_label] = {
                "positivo": positivo_total,
                "negativo": negativo_total,
                "taxa": taxa
            }

            # Próximo mês
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)

        # Salvar no cache de sessão
        set_cached_data("monthly_positivity_data", cache_key, monthly_data)
        return monthly_data

    except Exception as e:
        st.error(f"Erro ao buscar dados mensais: {e}")
        return {}


def get_metrics_data(
    laboratory_ids: list = None,
    purpose_type: str = None,
    renach_status: str = None,
    analysis_type: str = None,
    state: str = None,
    city: str = None,
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
) -> dict:
    """
    Busca métricas agregadas com filtros avançados:
    - Quantidade Amostras Negativas na Triagem
    - Quantidade Amostras Negativas em Confirmatório
    - Quantidade Amostras Positivas em Confirmatório
    Usa cache de sessão para evitar recarregamentos.
    """
    # Verificar cache de sessão
    cache_key = generate_cache_key(
        "metrics", laboratory_ids, purpose_type, renach_status,
        analysis_type, state, city, start_date_filter, end_date_filter
    )
    cached = get_cached_data("metrics_data", cache_key)
    if cached is not None:
        return cached

    try:
        lots_collection = get_collection("lots")
        results_collection = get_collection("results")

        # Definir período (usar filtro de data ou últimos 30 dias por padrão)
        if start_date_filter and end_date_filter:
            start_date = start_date_filter
            end_date = end_date_filter
        else:
            start_date = DEFAULT_START_DATE
            end_date = DEFAULT_END_DATE

        # Buscar amostras filtradas
        # Retorna (chain_ids para lots._samples, sample_codes para results.samples._sample)
        allowed_chain_ids, allowed_sample_codes = get_filtered_samples_advanced(
            laboratory_ids=laboratory_ids,
            purpose_type=purpose_type,
            renach_status=renach_status,
            state=state,
            city=city,
            start_date=start_date,
            end_date=end_date
        )

        metrics = {
            "negativas_triagem": 0,
            "positivas_triagem": 0,
            "negativas_confirmatorio": 0,
            "positivas_confirmatorio": 0,
            "total_amostras": 0,
            "taxa_geral": 0.0
        }

        # Função auxiliar para contar resultados de triagem
        def count_results_triagem() -> tuple:
            """Conta resultados de triagem (screening) para as amostras filtradas."""
            lots = list(lots_collection.find(
                {
                    "analysisType": "screening",
                    "createdAt": {"$gte": start_date, "$lte": end_date}
                },
                {"code": 1, "_samples": 1}
            ))

            if not lots:
                return 0, 0

            lot_codes = []

            for lot in lots:
                lot_code = lot.get('code')
                lot_samples = lot.get('_samples', [])  # ObjectIds das chainofcustodies

                if lot_code:
                    if allowed_chain_ids is not None:
                        # Filtrar: só incluir se o lote tem amostras permitidas
                        matching_samples = [s for s in lot_samples if s in allowed_chain_ids]
                        if matching_samples:
                            lot_codes.append(lot_code)
                    else:
                        lot_codes.append(lot_code)

            if not lot_codes:
                return 0, 0

            results = list(results_collection.find(
                {"_lot": {"$in": lot_codes}},
                {"_lot": 1, "samples._sample": 1, "samples.positive": 1}
            ))

            positivo = 0
            negativo = 0

            for result in results:
                for sample in result.get('samples', []):
                    sample_code = sample.get('_sample')  # Este é o sample.code (número)

                    # Se tiver filtros, verificar se o sample.code está permitido
                    if allowed_sample_codes is not None:
                        if sample_code not in allowed_sample_codes:
                            continue

                    if sample.get('positive', False):
                        positivo += 1
                    else:
                        negativo += 1

            return positivo, negativo

        # Função auxiliar para contar resultados de confirmatório
        def count_results_confirmatorio() -> tuple:
            """
            Conta resultados de confirmatório para as amostras filtradas.

            LÓGICA CORRETA:
            1. Busca lotes do tipo 'confirmatory' e 'confirmatoryTHC'
            2. Verifica quais das amostras filtradas (allowed_chain_ids) estão nesses lotes
            3. Conta os resultados apenas dessas amostras

            Isso garante que só contamos amostras que realmente foram para confirmatório.
            """
            # Buscar lotes de confirmatório
            lots_conf = list(lots_collection.find(
                {
                    "analysisType": {"$in": ["confirmatory", "confirmatoryTHC"]},
                    "createdAt": {"$gte": start_date, "$lte": end_date}
                },
                {"code": 1, "_samples": 1}
            ))

            if not lots_conf:
                return 0, 0

            # Identificar quais amostras filtradas estão em lotes de confirmatório
            lot_codes = []
            samples_in_confirmatorio = set()  # chain_ids que foram para confirmatório

            for lot in lots_conf:
                lot_code = lot.get('code')
                lot_samples = lot.get('_samples', [])  # ObjectIds das chainofcustodies

                if lot_code:
                    if allowed_chain_ids is not None:
                        # Verificar quais amostras do filtro estão neste lote
                        matching_samples = [s for s in lot_samples if s in allowed_chain_ids]
                        if matching_samples:
                            lot_codes.append(lot_code)
                            samples_in_confirmatorio.update(matching_samples)
                    else:
                        lot_codes.append(lot_code)
                        samples_in_confirmatorio.update(lot_samples)

            if not lot_codes:
                return 0, 0

            # Buscar os sample.codes das amostras que foram para confirmatório
            if samples_in_confirmatorio:
                confirmatorio_chain_to_code = get_chain_to_sample_code_map(samples_in_confirmatorio)
                confirmatorio_sample_codes = set(confirmatorio_chain_to_code.values())
            else:
                confirmatorio_sample_codes = set()

            # Buscar resultados dos lotes de confirmatório
            results = list(results_collection.find(
                {"_lot": {"$in": lot_codes}},
                {"_lot": 1, "samples._sample": 1, "samples.positive": 1}
            ))

            positivo = 0
            negativo = 0

            for result in results:
                for sample in result.get('samples', []):
                    sample_code = sample.get('_sample')  # Este é o sample.code (número)

                    # Verificar se esta amostra é das que queremos contar
                    if allowed_sample_codes is not None:
                        # Usar a interseção: amostras filtradas que foram para confirmatório
                        if sample_code not in confirmatorio_sample_codes:
                            continue

                    if sample.get('positive', False):
                        positivo += 1
                    else:
                        negativo += 1

            return positivo, negativo

        # Se tiver filtro de tipo de análise específico
        if analysis_type and analysis_type != "all":
            if analysis_type == "screening":
                pos, neg = count_results_triagem()
                metrics["negativas_triagem"] = neg
                metrics["positivas_triagem"] = pos
            else:
                pos, neg = count_results_confirmatorio()
                metrics["negativas_confirmatorio"] = neg
                metrics["positivas_confirmatorio"] = pos
        else:
            # Triagem
            pos_triagem, neg_triagem = count_results_triagem()
            metrics["negativas_triagem"] = neg_triagem
            metrics["positivas_triagem"] = pos_triagem

            # Confirmatório (já inclui confirmatory e confirmatoryTHC)
            pos_conf, neg_conf = count_results_confirmatorio()
            metrics["negativas_confirmatorio"] = neg_conf
            metrics["positivas_confirmatorio"] = pos_conf

        # Calcular totais e taxa geral
        # Total de amostras = todas as amostras que passaram pela triagem
        metrics["total_amostras"] = metrics["negativas_triagem"] + metrics["positivas_triagem"]

        # Taxa geral de positividade = positivas na triagem / total de amostras na triagem
        # (consistente com a lógica do gráfico mensal)
        metrics["taxa_geral"] = (metrics["positivas_triagem"] / metrics["total_amostras"] * 100) if metrics["total_amostras"] > 0 else 0.0

        # Salvar no cache de sessão
        set_cached_data("metrics_data", cache_key, metrics)
        return metrics

    except Exception as e:
        st.error(f"Erro ao buscar métricas: {e}")
        return {
            "negativas_triagem": 0, "positivas_triagem": 0,
            "negativas_confirmatorio": 0, "positivas_confirmatorio": 0,
            "total_amostras": 0, "taxa_geral": 0.0
        }


def get_monthly_data_by_lab(
    laboratory_ids: list = None,
    purpose_type: str = None,
    renach_status: str = None,
    analysis_type: str = "screening",
    state: str = None,
    city: str = None,
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
) -> dict:
    """
    Busca dados de positividade por mês para cada laboratório individualmente.
    Retorna um dicionário {lab_id: {lab_name: str, monthly_data: dict, metrics: dict}}
    Usa processamento paralelo para múltiplos laboratórios.
    """
    if not laboratory_ids:
        return {}

    labs_map = get_laboratories_map()

    def fetch_lab_data(lab_id):
        """Função auxiliar para buscar dados de um laboratório"""
        lab_name = labs_map.get(lab_id, 'Desconhecido')

        monthly_data = get_monthly_positivity_data(
            laboratory_ids=[lab_id],
            purpose_type=purpose_type,
            renach_status=renach_status,
            analysis_type=analysis_type,
            state=state,
            city=city,
            start_date_filter=start_date_filter,
            end_date_filter=end_date_filter
        )

        metrics = get_metrics_data(
            laboratory_ids=[lab_id],
            purpose_type=purpose_type,
            renach_status=renach_status,
            analysis_type=analysis_type,
            state=state,
            city=city,
            start_date_filter=start_date_filter,
            end_date_filter=end_date_filter
        )

        return {
            "lab_name": lab_name,
            "monthly_data": monthly_data,
            "metrics": metrics
        }

    # Processar laboratórios em paralelo
    results_by_lab = {}

    with ThreadPoolExecutor(max_workers=min(len(laboratory_ids), 5)) as executor:
        future_to_lab = {executor.submit(fetch_lab_data, lab_id): lab_id for lab_id in laboratory_ids}

        for future in as_completed(future_to_lab):
            lab_id = future_to_lab[future]
            try:
                results_by_lab[lab_id] = future.result()
            except Exception as e:
                results_by_lab[lab_id] = {
                    "lab_name": labs_map.get(lab_id, 'Desconhecido'),
                    "monthly_data": {},
                    "metrics": {"negativas_triagem": 0, "negativas_confirmatorio": 0, "positivas_confirmatorio": 0}
                }

    return results_by_lab


def get_positivity_by_substance(
    laboratory_ids: list = None,
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
) -> dict:
    """
    Busca a distribuição de positividade por substância.
    Retorna dict {substancia: {positivo: int, total: int, percentual: float}}
    """
    cache_key = generate_cache_key("positivity_substance", laboratory_ids, start_date_filter, end_date_filter)
    cached = get_cached_data("positivity_substance", cache_key)
    if cached is not None:
        return cached

    try:
        # Definir período (usar filtro de data ou últimos 30 dias por padrão)
        start_date = start_date_filter or DEFAULT_START_DATE
        end_date = end_date_filter or DEFAULT_END_DATE

        # Buscar amostras filtradas
        allowed_chain_ids, allowed_sample_codes = get_filtered_samples_advanced(
            laboratory_ids=laboratory_ids,
            start_date=start_date,
            end_date=end_date
        )

        # Buscar lotes do período
        lots_collection = get_collection("lots")
        lots = list(lots_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"code": 1, "_samples": 1}
        ))

        if not lots:
            return {}

        # Filtrar lotes
        lot_codes = []
        for lot in lots:
            lot_code = lot.get('code')
            lot_samples = lot.get('_samples', [])
            if lot_code:
                if allowed_chain_ids is not None:
                    if any(s in allowed_chain_ids for s in lot_samples):
                        lot_codes.append(lot_code)
                else:
                    lot_codes.append(lot_code)

        if not lot_codes:
            return {}

        # Buscar results com compounds
        results_collection = get_collection("results")
        results = list(results_collection.find(
            {"_lot": {"$in": lot_codes}},
            {"samples": 1}
        ))

        # Mapeamento de compounds
        compounds_map = get_compounds_map()

        # Contar por substância
        substance_counts = {}
        for result in results:
            for sample in result.get('samples', []):
                sample_code = sample.get('_sample')
                if allowed_sample_codes is not None and sample_code not in allowed_sample_codes:
                    continue

                for compound in sample.get('_compound', []):
                    compound_id = compound.get('_id')
                    is_positive = compound.get('positive', False)

                    if isinstance(compound_id, ObjectId):
                        compound_id = str(compound_id)

                    substance_name = compounds_map.get(compound_id, 'Desconhecido')
                    if substance_name == 'Desconhecido':
                        continue

                    if substance_name not in substance_counts:
                        substance_counts[substance_name] = {"positivo": 0, "total": 0}

                    substance_counts[substance_name]["total"] += 1
                    if is_positive:
                        substance_counts[substance_name]["positivo"] += 1

        # Calcular percentuais
        for substance in substance_counts:
            total = substance_counts[substance]["total"]
            positivo = substance_counts[substance]["positivo"]
            substance_counts[substance]["percentual"] = (positivo / total * 100) if total > 0 else 0

        set_cached_data("positivity_substance", cache_key, substance_counts)
        return substance_counts

    except Exception as e:
        st.error(f"Erro ao buscar dados por substância: {e}")
        return {}


def get_national_average_rate(
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
) -> float:
    """
    Busca a taxa média nacional de positividade (sem filtros de laboratório).
    Usado para comparação com laboratórios individuais.
    """
    cache_key = generate_cache_key("national_rate", start_date_filter, end_date_filter)
    cached = get_cached_data("national_rate", cache_key)
    if cached is not None:
        return cached

    try:
        # Buscar métricas sem filtro de laboratório
        metrics = get_metrics_data(
            laboratory_ids=None,
            purpose_type=None,
            renach_status=None,
            analysis_type="all",
            state=None,
            city=None,
            start_date_filter=start_date_filter,
            end_date_filter=end_date_filter
        )

        rate = metrics.get("taxa_geral", 0.0)
        set_cached_data("national_rate", cache_key, rate)
        return rate

    except Exception:
        return 0.0


def get_positivity_by_state(
    laboratory_ids: list = None,
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
) -> dict:
    """
    Busca a distribuição de positividade por estado.
    Retorna dict {estado: {positivo: int, negativo: int, total: int, taxa: float}}
    """
    cache_key = generate_cache_key("positivity_state", laboratory_ids, start_date_filter, end_date_filter)
    cached = get_cached_data("positivity_state", cache_key)
    if cached is not None:
        return cached

    try:
        start_date = start_date_filter or DEFAULT_START_DATE
        end_date = end_date_filter or DEFAULT_END_DATE

        # Buscar labs por estado
        labs = get_laboratories_with_address()
        labs_by_state = {}
        for lab in labs:
            state = lab.get('state', 'Não informado') or 'Não informado'
            if state not in labs_by_state:
                labs_by_state[state] = []
            labs_by_state[state].append(lab['id'])

        state_data = {}
        for state, lab_ids in labs_by_state.items():
            # Filtrar por laboratory_ids se fornecido
            if laboratory_ids:
                filtered_ids = [lid for lid in lab_ids if lid in laboratory_ids]
                if not filtered_ids:
                    continue
                lab_ids = filtered_ids

            # Buscar dados para este estado
            monthly = get_monthly_positivity_data(
                laboratory_ids=lab_ids,
                analysis_type="all",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            total_positivo = sum(m.get("positivo", 0) for m in monthly.values())
            total_negativo = sum(m.get("negativo", 0) for m in monthly.values())
            total = total_positivo + total_negativo

            if total > 0:
                state_data[state] = {
                    "positivo": total_positivo,
                    "negativo": total_negativo,
                    "total": total,
                    "taxa": (total_positivo / total * 100)
                }

        set_cached_data("positivity_state", cache_key, state_data)
        return state_data

    except Exception as e:
        st.error(f"Erro ao buscar dados por estado: {e}")
        return {}


def get_positivity_by_purpose(
    laboratory_ids: list = None,
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
) -> dict:
    """
    Busca a distribuição de positividade por finalidade da amostra.
    Retorna dict {finalidade: {positivo: int, negativo: int, total: int, taxa: float}}
    """
    cache_key = generate_cache_key("positivity_purpose", laboratory_ids, start_date_filter, end_date_filter)
    cached = get_cached_data("positivity_purpose", cache_key)
    if cached is not None:
        return cached

    try:
        start_date = start_date_filter or DEFAULT_START_DATE
        end_date = end_date_filter or DEFAULT_END_DATE

        # Mapeamento de finalidades
        purpose_names = {
            "clt": "CLT",
            "cltCnh": "CLT + CNH",
            "civilService": "Concurso Público",
            "periodic": "Periódico",
            "categoryChange": "Mudança de Categoria",
            "hiring": "Admissão",
            "renovation": "Renovação",
            "resignation": "Demissão"
        }

        purpose_data = {}
        for purpose_key, purpose_name in purpose_names.items():
            monthly = get_monthly_positivity_data(
                laboratory_ids=laboratory_ids,
                purpose_type=purpose_key,
                analysis_type="all",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            total_positivo = sum(m.get("positivo", 0) for m in monthly.values())
            total_negativo = sum(m.get("negativo", 0) for m in monthly.values())
            total = total_positivo + total_negativo

            if total > 0:
                purpose_data[purpose_name] = {
                    "positivo": total_positivo,
                    "negativo": total_negativo,
                    "total": total,
                    "taxa": (total_positivo / total * 100)
                }

        set_cached_data("positivity_purpose", cache_key, purpose_data)
        return purpose_data

    except Exception as e:
        st.error(f"Erro ao buscar dados por finalidade: {e}")
        return {}


def get_positivity_by_lot_type(
    laboratory_ids: list = None,
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
) -> dict:
    """
    Busca a distribuição de positividade por tipo de lote.
    Retorna dict {tipo_lote: {positivo: int, negativo: int, total: int, taxa: float}}
    """
    cache_key = generate_cache_key("positivity_lot_type", laboratory_ids, start_date_filter, end_date_filter)
    cached = get_cached_data("positivity_lot_type", cache_key)
    if cached is not None:
        return cached

    try:
        start_date = start_date_filter or DEFAULT_START_DATE
        end_date = end_date_filter or DEFAULT_END_DATE

        lot_type_names = {
            "screening": "Triagem",
            "confirmatory": "Confirmatório",
            "confirmatoryTHC": "Confirmatório THC"
        }

        lot_type_data = {}
        for lot_key, lot_name in lot_type_names.items():
            monthly = get_monthly_positivity_data(
                laboratory_ids=laboratory_ids,
                analysis_type=lot_key,
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            total_positivo = sum(m.get("positivo", 0) for m in monthly.values())
            total_negativo = sum(m.get("negativo", 0) for m in monthly.values())
            total = total_positivo + total_negativo

            if total > 0:
                lot_type_data[lot_name] = {
                    "positivo": total_positivo,
                    "negativo": total_negativo,
                    "total": total,
                    "taxa": (total_positivo / total * 100)
                }

        set_cached_data("positivity_lot_type", cache_key, lot_type_data)
        return lot_type_data

    except Exception as e:
        st.error(f"Erro ao buscar dados por tipo de lote: {e}")
        return {}


def get_positivity_by_renach(
    laboratory_ids: list = None,
    start_date_filter: datetime = None,
    end_date_filter: datetime = None
) -> dict:
    """
    Busca a distribuição de positividade por status RENACH.
    Retorna dict {status: {positivo: int, negativo: int, total: int, taxa: float}}
    """
    cache_key = generate_cache_key("positivity_renach", laboratory_ids, start_date_filter, end_date_filter)
    cached = get_cached_data("positivity_renach", cache_key)
    if cached is not None:
        return cached

    try:
        start_date = start_date_filter or DEFAULT_START_DATE
        end_date = end_date_filter or DEFAULT_END_DATE

        renach_data = {}
        for renach_status, renach_name in [("sim", "No RENACH"), ("nao", "Fora do RENACH")]:
            monthly = get_monthly_positivity_data(
                laboratory_ids=laboratory_ids,
                renach_status=renach_status,
                analysis_type="all",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            total_positivo = sum(m.get("positivo", 0) for m in monthly.values())
            total_negativo = sum(m.get("negativo", 0) for m in monthly.values())
            total = total_positivo + total_negativo

            if total > 0:
                renach_data[renach_name] = {
                    "positivo": total_positivo,
                    "negativo": total_negativo,
                    "total": total,
                    "taxa": (total_positivo / total * 100)
                }

        set_cached_data("positivity_renach", cache_key, renach_data)
        return renach_data

    except Exception as e:
        st.error(f"Erro ao buscar dados por RENACH: {e}")
        return {}


def render_taxa_positividade():

    st.title("📊 Taxa de Positividade")

    # Filtros na página
    st.markdown("### 🔍 Filtros")

    # Primeira linha de filtros: CNPJ Laboratório (multiselect) e Período
    col_f1, col_f2 = st.columns([2, 1])

    with col_f1:
        # Buscar laboratórios com CNPJ
        labs = get_laboratories_with_address()
        lab_options = {}
        for lab in labs:
            cnpj = lab.get('cnpj', '')
            name = lab.get('name', 'Desconhecido')
            if cnpj:
                label = f"{name} ({cnpj})"
            else:
                label = name
            lab_options[label] = lab['id']

        selected_labs = st.multiselect(
            "CNPJ Laboratório (seleção múltipla)",
            options=list(lab_options.keys()),
            default=[],
            key="taxa_labs_multi",
            placeholder="Selecione um ou mais laboratórios..."
        )
        lab_ids_filter = [lab_options[lab] for lab in selected_labs] if selected_labs else None

    with col_f2:
        # Mostrar período selecionado (vem do filtro global na sidebar)
        periodo_inicio, periodo_fim = get_selected_period()
        st.markdown("**Período (filtro global)**")
        st.info(f"📅 {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')}")

    # Segunda linha de filtros
    col_f3, col_f4, col_f5, col_f6 = st.columns(4)

    with col_f3:
        finalidades = {
            "Todos": None,
            "Periódico": "periodic",
            "Mudança de Categoria": "categoryChange",
            "Admissão": "hiring",
            "Renovação": "renovation",
            "Demissão": "resignation"
        }
        selected_finalidade = st.selectbox(
            "Finalidade da Amostra",
            options=list(finalidades.keys()),
            index=0,
            key="taxa_finalidade"
        )
        purpose_filter = finalidades[selected_finalidade]

    with col_f4:
        renach_options = {
            "Todos": None,
            "Sim": "sim",
            "Não": "nao"
        }
        selected_renach = st.selectbox(
            "Status Renach",
            options=list(renach_options.keys()),
            index=0,
            key="taxa_renach"
        )
        renach_filter = renach_options[selected_renach]

    with col_f5:
        tipo_lote_options = {
            "Todos": "all",
            "Triagem": "screening",
            "Confirmatório": "confirmatory",
            "Confirmatório THC": "confirmatoryTHC"
        }
        selected_tipo_lote = st.selectbox(
            "Tipo de Lote",
            options=list(tipo_lote_options.keys()),
            index=0,
            key="taxa_tipo_lote"
        )
        analysis_type_filter = tipo_lote_options[selected_tipo_lote]

    with col_f6:
        # Estado do laboratório
        states = get_unique_states()
        state_options = ["Todos"] + states
        selected_state = st.selectbox(
            "Estado (UF)",
            options=state_options,
            index=0,
            key="taxa_estado"
        )
        state_filter = selected_state if selected_state != "Todos" else None

    # Terceira linha: Cidade (dependente do estado)
    col_f7, col_f8, col_f9 = st.columns([1, 1, 2])

    with col_f7:
        cities = get_cities_by_state(state_filter)
        city_options = ["Todas"] + cities
        selected_city = st.selectbox(
            "Cidade",
            options=city_options,
            index=0,
            key="taxa_cidade"
        )
        city_filter = selected_city if selected_city != "Todas" else None


    # Usar período do filtro global
    start_date_filter, end_date_filter = get_selected_period()

    # Parâmetros comuns para as funções
    common_params = {
        "laboratory_ids": lab_ids_filter,
        "purpose_type": purpose_filter,
        "renach_status": renach_filter,
        "analysis_type": analysis_type_filter,
        "state": state_filter,
        "city": city_filter,
        "start_date_filter": start_date_filter,
        "end_date_filter": end_date_filter
    }

    with st.spinner("Carregando dados em paralelo..."):
        # Definir tarefas para execução paralela
        tasks = {
            "monthly_data": (get_monthly_positivity_data, (), common_params),
            "metrics": (get_metrics_data, (), common_params),
        }

        # Se houver laboratórios selecionados, adicionar tarefa de dados por lab
        if lab_ids_filter and len(lab_ids_filter) > 0:
            tasks["data_by_lab"] = (get_monthly_data_by_lab, (), common_params)

        # Executar todas as tarefas em paralelo
        results = run_parallel_tasks(tasks, max_workers=3)

        monthly_data = results.get("monthly_data", {})
        metrics = results.get("metrics", {
            "negativas_triagem": 0, "positivas_triagem": 0,
            "negativas_confirmatorio": 0, "positivas_confirmatorio": 0,
            "total_amostras": 0, "taxa_geral": 0.0
        })
        data_by_lab = results.get("data_by_lab", {})

        # Buscar taxa média nacional para comparação
        taxa_nacional = get_national_average_rate(start_date_filter, end_date_filter)

    # Função para formatar números com ponto como separador de milhares
    def format_number(n: int) -> str:
        return f"{n:,}".replace(",", ".")

    # Calcular diferença em relação à média nacional
    taxa_lab = metrics.get("taxa_geral", 0.0)
    diferenca_nacional = taxa_lab - taxa_nacional if taxa_nacional > 0 else 0.0
    diferenca_texto = f"+{diferenca_nacional:.2f}%" if diferenca_nacional >= 0 else f"{diferenca_nacional:.2f}%"
    diferenca_cor = "#FF6B6B" if diferenca_nacional > 0 else "#4CAF50"  # Vermelho se acima, verde se abaixo

    # Cards - Primeira linha (principais)
    st.markdown("### Resumo para o(s) CNPJ(s) selecionado(s)")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%); padding: 15px; border-radius: 8px; text-align: center;">
                <h2 style="color: #FFD700; margin: 0; font-size: 2rem;">{taxa_lab:.2f}%</h2>
                <p style="color: #E8E8E8; margin: 8px 0 0 0; font-size: 0.8rem;">Taxa Geral de Positividade</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%); padding: 15px; border-radius: 8px; text-align: center;">
                <h2 style="color: #00CED1; margin: 0; font-size: 2rem;">{format_number(metrics.get('total_amostras', 0))}</h2>
                <p style="color: #E8E8E8; margin: 8px 0 0 0; font-size: 0.8rem;">Total de Amostras</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m3:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%); padding: 15px; border-radius: 8px; text-align: center;">
                <h2 style="color: #00CED1; margin: 0; font-size: 2rem;">{taxa_nacional:.2f}%</h2>
                <p style="color: #E8E8E8; margin: 8px 0 0 0; font-size: 0.8rem;">Taxa Média Nacional</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m4:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%); padding: 15px; border-radius: 8px; text-align: center;">
                <h2 style="color: {diferenca_cor}; margin: 0; font-size: 2rem;">{diferenca_texto}</h2>
                <p style="color: #E8E8E8; margin: 8px 0 0 0; font-size: 0.8rem;">Diferença vs Nacional</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Cards - Segunda linha (detalhamento triagem e confirmatório)
    col_t1, col_t2, col_c1, col_c2 = st.columns(4)

    with col_t1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0D4F4F 0%, #1A3A3A 100%); padding: 12px; border-radius: 8px; text-align: center;">
                <h3 style="color: #4CAF50; margin: 0; font-size: 1.5rem;">{format_number(metrics.get('negativas_triagem', 0))}</h3>
                <p style="color: #E8E8E8; margin: 5px 0 0 0; font-size: 0.75rem;">Negativas (Triagem)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_t2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #4F0D0D 0%, #3A1A1A 100%); padding: 12px; border-radius: 8px; text-align: center;">
                <h3 style="color: #FF6B6B; margin: 0; font-size: 1.5rem;">{format_number(metrics.get('positivas_triagem', 0))}</h3>
                <p style="color: #E8E8E8; margin: 5px 0 0 0; font-size: 0.75rem;">Positivas (Triagem)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_c1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0D4F4F 0%, #1A3A3A 100%); padding: 12px; border-radius: 8px; text-align: center;">
                <h3 style="color: #4CAF50; margin: 0; font-size: 1.5rem;">{format_number(metrics.get('negativas_confirmatorio', 0))}</h3>
                <p style="color: #E8E8E8; margin: 5px 0 0 0; font-size: 0.75rem;">Negativas (Confirmatório)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_c2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #4F0D0D 0%, #3A1A1A 100%); padding: 12px; border-radius: 8px; text-align: center;">
                <h3 style="color: #FF6B6B; margin: 0; font-size: 1.5rem;">{format_number(metrics.get('positivas_confirmatorio', 0))}</h3>
                <p style="color: #E8E8E8; margin: 5px 0 0 0; font-size: 0.75rem;">Positivas (Confirmatório)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Se há laboratórios selecionados, mostrar gráfico de linhas por laboratório
    if data_by_lab and len(data_by_lab) > 0:
        st.subheader("Taxa de Positividade por Laboratório")

        # Preparar dados para gráfico de linhas
        meses_nomes = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
                       "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]

        chart_data = []
        for lab_data in data_by_lab.values():
            lab_name = lab_data["lab_name"]
            monthly = lab_data["monthly_data"]
            for mes in meses_nomes:
                if mes in monthly:
                    chart_data.append({
                        "Mês": mes,
                        "Taxa": monthly[mes]["taxa"],
                        "Laboratório": lab_name
                    })

        if chart_data:
            df_lines = pd.DataFrame(chart_data)

            # Ordenar meses corretamente
            df_lines["Mês"] = pd.Categorical(df_lines["Mês"], categories=meses_nomes, ordered=True)
            df_lines = df_lines.sort_values("Mês")

            fig_lines = px.line(
                df_lines,
                x="Mês",
                y="Taxa",
                color="Laboratório",
                markers=True,
                title="Comparativo de Taxa de Positividade por Laboratório"
            )

            fig_lines.update_layout(
                xaxis_title="",
                yaxis_title="Taxa (%)",
                yaxis_ticksuffix="%",
                height=500,
                margin=dict(t=50, b=50, l=50, r=50),
                xaxis_tickangle=-45,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig_lines, use_container_width=True, key="chart_taxa_labs_linhas")

        # Tabela com totais por laboratório
        st.subheader("Métricas por Laboratório")

        table_data = []
        for lab_data in data_by_lab.values():
            lab_metrics = lab_data["metrics"]
            monthly = lab_data["monthly_data"]

            # Calcular totais e taxa média
            total_positivo = sum(m.get("positivo", 0) for m in monthly.values())
            total_negativo = sum(m.get("negativo", 0) for m in monthly.values())
            total_amostras = total_positivo + total_negativo
            taxa_media = (total_positivo / total_amostras * 100) if total_amostras > 0 else 0

            table_data.append({
                "Laboratório": lab_data["lab_name"],
                "Neg. Triagem": format_number(lab_metrics["negativas_triagem"]),
                "Neg. Confirmatório": format_number(lab_metrics["negativas_confirmatorio"]),
                "Pos. Confirmatório": format_number(lab_metrics["positivas_confirmatorio"]),
                "Total Amostras": format_number(total_amostras),
                "Taxa Média (%)": f"{taxa_media:.2f}%"
            })

        if table_data:
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Gráfico de barras (total ou quando nenhum lab selecionado)
    if monthly_data:
        if not data_by_lab:
            st.subheader("Total de Amostras por Mês")

        meses = list(monthly_data.keys())
        totais = [monthly_data[m]["positivo"] + monthly_data[m]["negativo"] for m in meses]
        taxas = [monthly_data[m]["taxa"] for m in meses]

        df_chart = pd.DataFrame({
            "Mês": meses,
            "Total": totais,
            "Taxa": taxas
        })

        # Texto mostrando total e taxa
        df_chart["Texto"] = df_chart.apply(
            lambda row: f"{row['Total']:,} ({row['Taxa']:.1f}%)".replace(",", "."), axis=1
        )

        title = "Total de Amostras por Mês" if data_by_lab else "Total de Amostras por Mês"

        fig = px.bar(
            df_chart,
            x="Mês",
            y="Total",
            title=title,
            text="Texto",
            color="Taxa",
            color_continuous_scale=['#00CED1', '#1A1A2E']
        )

        fig.update_traces(
            textposition='outside',
            textfont_size=10
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Total de Amostras",
            height=500,
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis_tickangle=-45,
            showlegend=False,
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True, key="chart_taxa_mensal")
    else:
        st.warning("Nenhum dado encontrado")

    st.markdown("---")

    # ============================================
    # SEÇÃO DE ANÁLISES DETALHADAS
    # ============================================
    st.subheader("Análises Detalhadas")

    # Buscar dados para os gráficos analíticos em paralelo
    with st.spinner("Carregando análises detalhadas em paralelo..."):
        detailed_tasks = {
            "substance": (get_positivity_by_substance, (lab_ids_filter, start_date_filter, end_date_filter)),
            "state": (get_positivity_by_state, (lab_ids_filter, start_date_filter, end_date_filter)),
            "purpose": (get_positivity_by_purpose, (lab_ids_filter, start_date_filter, end_date_filter)),
            "lot_type": (get_positivity_by_lot_type, (lab_ids_filter, start_date_filter, end_date_filter)),
            "renach": (get_positivity_by_renach, (lab_ids_filter, start_date_filter, end_date_filter)),
        }

        detailed_results = run_parallel_tasks(detailed_tasks, max_workers=5)

        substance_data = detailed_results.get("substance", {})
        state_data = detailed_results.get("state", {})
        purpose_data = detailed_results.get("purpose", {})
        lot_type_data = detailed_results.get("lot_type", {})
        renach_status_data = detailed_results.get("renach", {})

    # Primeira linha: Substâncias - gráfico de barras horizontal
    st.markdown("#### Positividade por Substância")
    if substance_data:
        # Filtrar apenas substâncias com positivos
        positive_substances = {k: v for k, v in substance_data.items() if v["positivo"] > 0}

        if positive_substances:
            df_substance = pd.DataFrame([
                {
                    "Substância": k,
                    "Positivos": v["positivo"],
                    "Total": v["total"],
                    "Taxa (%)": v["percentual"]
                }
                for k, v in positive_substances.items()
            ])
            df_substance = df_substance.sort_values("Positivos", ascending=True)

            # Calcular percentual de cada substância no total de positivos
            total_positivos = df_substance["Positivos"].sum()
            df_substance["% do Total"] = (df_substance["Positivos"] / total_positivos * 100).round(2)

            col_graf_sub, col_tab_sub = st.columns([2, 1])

            with col_graf_sub:
                # Gráfico de barras horizontal - quantidade de positivos por substância
                fig_bar_sub = px.bar(
                    df_substance,
                    x='Positivos',
                    y='Substância',
                    orientation='h',
                    title='Quantidade de Amostras Positivas por Substância',
                    text=df_substance['Positivos'].apply(lambda x: f"{x:,}".replace(",", ".")),
                    color='Positivos',
                    color_continuous_scale=['#00CED1', '#1A1A2E']
                )
                fig_bar_sub.update_traces(textposition='outside', textfont_size=10)
                fig_bar_sub.update_layout(
                    height=max(400, len(df_substance) * 25),
                    xaxis_title="Quantidade de Positivos",
                    yaxis_title="",
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_bar_sub, use_container_width=True, key="chart_taxa_substancias")

            with col_tab_sub:
                # Tabela resumo
                df_table_sub = df_substance.sort_values("Positivos", ascending=False).copy()
                df_table_sub["Positivos"] = df_table_sub["Positivos"].apply(lambda x: f"{x:,}".replace(",", "."))
                df_table_sub["Total"] = df_table_sub["Total"].apply(lambda x: f"{x:,}".replace(",", "."))
                df_table_sub["Taxa (%)"] = df_table_sub["Taxa (%)"].apply(lambda x: f"{x:.2f}%")
                df_table_sub["% do Total"] = df_table_sub["% do Total"].apply(lambda x: f"{x:.1f}%")

                st.markdown("**Resumo por Substância**")
                st.dataframe(
                    df_table_sub[["Substância", "Positivos", "Taxa (%)", "% do Total"]],
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, len(df_table_sub) * 35 + 40)
                )
        else:
            st.info("Nenhuma amostra positiva encontrada para o período selecionado")
    else:
        st.info("Carregando dados de substâncias...")

    st.markdown("---")

    # Segunda linha: Estado e Finalidade
    col_geo, col_purpose = st.columns(2)

    with col_geo:
        st.markdown("#### Positividade por Estado")
        if state_data:
            df_state = pd.DataFrame([
                {"Estado": k, "Total": v["total"], "Positivos": v["positivo"], "Taxa": v["taxa"]}
                for k, v in state_data.items()
            ])
            df_state = df_state.sort_values("Total", ascending=True)

            # Texto mostrando total e taxa
            df_state["Texto"] = df_state.apply(
                lambda row: f"{row['Total']:,} ({row['Taxa']:.1f}%)".replace(",", "."), axis=1
            )

            fig_state = px.bar(
                df_state,
                x='Total',
                y='Estado',
                orientation='h',
                title='Total de Amostras por Estado',
                text='Texto',
                color='Taxa',
                color_continuous_scale=['#00CED1', '#1A1A2E']
            )
            fig_state.update_traces(textposition='outside', textfont_size=10)
            fig_state.update_layout(
                height=max(350, len(df_state) * 40),
                xaxis_title="Total de Amostras",
                yaxis_title="",
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_state, use_container_width=True, key="chart_taxa_estado")
        else:
            st.info("Nenhum dado de estado encontrado")

    with col_purpose:
        st.markdown("#### Positividade por Finalidade")
        if purpose_data:
            df_purpose = pd.DataFrame([
                {"Finalidade": k, "Total": v["total"], "Positivos": v["positivo"], "Taxa": v["taxa"]}
                for k, v in purpose_data.items()
            ])
            df_purpose = df_purpose.sort_values("Total", ascending=False)

            # Texto mostrando total e taxa
            df_purpose["Texto"] = df_purpose.apply(
                lambda row: f"{row['Total']:,} ({row['Taxa']:.1f}%)".replace(",", "."), axis=1
            )

            fig_purpose = px.bar(
                df_purpose,
                x='Finalidade',
                y='Total',
                title='Total de Amostras por Finalidade',
                text='Texto',
                color='Taxa',
                color_continuous_scale=['#00CED1', '#1A1A2E']
            )
            fig_purpose.update_traces(textposition='outside', textfont_size=10)
            fig_purpose.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="Total de Amostras",
                xaxis_tickangle=-45,
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_purpose, use_container_width=True, key="chart_taxa_finalidade")
        else:
            st.info("Nenhum dado de finalidade encontrado")

    st.markdown("---")

    # Terceira linha: Tipo de Lote e Status RENACH
    col_lot, col_renach = st.columns(2)

    with col_lot:
        st.markdown("#### Positividade por Tipo de Lote")
        if lot_type_data:
            df_lot = pd.DataFrame([
                {"Tipo de Lote": k, "Total": v["total"], "Positivos": v["positivo"],
                 "Negativos": v["negativo"], "Taxa": v["taxa"]}
                for k, v in lot_type_data.items()
            ])

            # Texto mostrando total e taxa
            df_lot["Texto"] = df_lot.apply(
                lambda row: f"{row['Total']:,} ({row['Taxa']:.1f}%)".replace(",", "."), axis=1
            )

            # Gráfico de barras com total
            fig_lot = px.bar(
                df_lot,
                x='Tipo de Lote',
                y='Total',
                title='Total de Amostras por Tipo de Lote',
                text='Texto',
                color='Taxa',
                color_continuous_scale=['#00CED1', '#1A1A2E']
            )
            fig_lot.update_traces(textposition='outside', textfont_size=11)
            fig_lot.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="Total de Amostras",
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_lot, use_container_width=True, key="chart_taxa_tipo_lote")

            # Tabela com detalhes
            df_lot_display = df_lot.copy()
            df_lot_display['Taxa'] = df_lot_display['Taxa'].apply(lambda x: f"{x:.2f}%")
            df_lot_display['Total'] = df_lot_display['Total'].apply(lambda x: f"{x:,}".replace(",", "."))
            df_lot_display['Positivos'] = df_lot_display['Positivos'].apply(lambda x: f"{x:,}".replace(",", "."))
            df_lot_display['Negativos'] = df_lot_display['Negativos'].apply(lambda x: f"{x:,}".replace(",", "."))
            st.dataframe(df_lot_display[['Tipo de Lote', 'Total', 'Positivos', 'Negativos', 'Taxa']], use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum dado de tipo de lote encontrado")

    with col_renach:
        st.markdown("#### Positividade por Status RENACH")
        if renach_status_data:
            df_renach = pd.DataFrame([
                {"Status RENACH": k, "Total": v["total"], "Positivos": v["positivo"],
                 "Negativos": v["negativo"], "Taxa": v["taxa"]}
                for k, v in renach_status_data.items()
            ])

            # Texto mostrando total e taxa
            df_renach["Texto"] = df_renach.apply(
                lambda row: f"{row['Total']:,} ({row['Taxa']:.1f}%)".replace(",", "."), axis=1
            )

            # Gráfico de barras com total
            fig_renach = px.bar(
                df_renach,
                x='Status RENACH',
                y='Total',
                title='Total de Amostras por Status RENACH',
                text='Texto',
                color='Status RENACH',
                color_discrete_map={'No RENACH': '#00CED1', 'Fora do RENACH': '#1A1A2E'}
            )
            fig_renach.update_traces(textposition='outside', textfont_size=11)
            fig_renach.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="Total de Amostras",
                showlegend=False
            )
            st.plotly_chart(fig_renach, use_container_width=True, key="chart_taxa_renach_status")

            # Tabela com detalhes
            df_renach_display = df_renach.copy()
            df_renach_display['Taxa'] = df_renach_display['Taxa'].apply(lambda x: f"{x:.2f}%")
            df_renach_display['Total'] = df_renach_display['Total'].apply(lambda x: f"{x:,}".replace(",", "."))
            df_renach_display['Positivos'] = df_renach_display['Positivos'].apply(lambda x: f"{x:,}".replace(",", "."))
            df_renach_display['Negativos'] = df_renach_display['Negativos'].apply(lambda x: f"{x:,}".replace(",", "."))
            st.dataframe(df_renach_display[['Status RENACH', 'Total', 'Positivos', 'Negativos', 'Taxa']], use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum dado de status RENACH encontrado")


def render_substancias():
    """
    Página 2 - Substâncias: ranking, frequência e positividade
    """
    st.title("🧪 Substâncias")

    # Mostrar período selecionado
    periodo_inicio, periodo_fim = get_selected_period()
    st.info(f"📅 **Período:** {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')} *(altere na sidebar)*")

    st.markdown("---")

    # Filtros
    st.markdown("### 🔍 Filtros")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        laboratories_map = get_laboratories_map()
        lab_options = {"Todos": None}
        lab_options.update({name: lab_id for lab_id, name in laboratories_map.items()})
        selected_lab_name = st.selectbox(
            "Laboratório",
            options=list(lab_options.keys()),
            index=0,
            key="subst_lab"
        )
        selected_lab_id = lab_options[selected_lab_name]

    with col_f2:
        meses = {
            "Todos": None, "Janeiro": 1, "Fevereiro": 2, "Março": 3,
            "Abril": 4, "Maio": 5, "Junho": 6, "Julho": 7,
            "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
        }
        selected_mes_name = st.selectbox("Mês", options=list(meses.keys()), index=0, key="subst_mes")
        selected_month = meses[selected_mes_name]

    with col_f3:
        analysis_options = {
            "Todos": "all",
            "Triagem": "screening",
            "Confirmatório": "confirmatory",
            "Confirmatório THC": "confirmatoryTHC"
        }
        selected_analysis_name = st.selectbox(
            "Tipo de Análise",
            options=list(analysis_options.keys()),
            index=0,
            key="subst_analysis"
        )
        selected_analysis = analysis_options[selected_analysis_name]

    st.markdown("---")

    with st.spinner("Carregando dados de substâncias..."):
        substance_stats = get_substance_statistics(selected_lab_id, selected_month, selected_analysis)

    if not substance_stats:
        st.warning("Nenhum dado encontrado para os filtros selecionados")
        return

    # Preparar dados para visualização
    df_stats = pd.DataFrame([
        {
            "Substância": name,
            "Total Análises": data["total"],
            "Positivos": data["positivos"],
            "Negativos": data["negativos"],
            "Taxa Positividade (%)": data["taxa"]
        }
        for name, data in substance_stats.items()
    ])

    # Ordenar por total de positivos (ranking)
    df_stats = df_stats.sort_values("Positivos", ascending=False)

    # Cards de resumo
    col1, col2, col3, col4 = st.columns(4)

    total_analises = df_stats["Total Análises"].sum()
    total_positivos = df_stats["Positivos"].sum()
    total_negativos = df_stats["Negativos"].sum()
    taxa_geral = (total_positivos / total_analises * 100) if total_analises > 0 else 0

    with col1:
        st.metric("Total de Análises", f"{total_analises:,}".replace(",", "."))
    with col2:
        st.metric("Total Positivos", f"{total_positivos:,}".replace(",", "."))
    with col3:
        st.metric("Total Negativos", f"{total_negativos:,}".replace(",", "."))
    with col4:
        st.metric("Taxa Geral", f"{taxa_geral:.2f}%")

    st.markdown("---")

    # Gráficos lado a lado
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("🏆 Ranking de Positividade")

        # Top 15 substâncias com mais positivos
        df_top = df_stats.head(15).copy()

        if not df_top.empty:
            df_top["Texto"] = df_top.apply(
                lambda row: f"{row['Positivos']:,} ({row['Taxa Positividade (%)']:.1f}%)".replace(",", "."),
                axis=1
            )

            fig_ranking = px.bar(
                df_top,
                y="Substância",
                x="Positivos",
                orientation="h",
                text="Texto",
                color="Taxa Positividade (%)",
                color_continuous_scale=["#1A1A2E", "#FF6B6B"]
            )

            fig_ranking.update_traces(textposition="outside", textfont_size=9)
            fig_ranking.update_layout(
                height=500,
                margin=dict(t=30, b=30, l=150, r=80),
                xaxis_title="Quantidade de Positivos",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_ranking, use_container_width=True, key="chart_ranking_substancias")

    with col_g2:
        st.subheader("📊 Taxa de Positividade por Substância")

        # Top 15 por taxa de positividade (com mínimo de análises)
        df_taxa = df_stats[df_stats["Total Análises"] >= 10].copy()
        df_taxa = df_taxa.sort_values("Taxa Positividade (%)", ascending=False).head(15)

        if not df_taxa.empty:
            df_taxa["Texto"] = df_taxa.apply(
                lambda row: f"{row['Taxa Positividade (%)']:.1f}%",
                axis=1
            )

            fig_taxa = px.bar(
                df_taxa,
                y="Substância",
                x="Taxa Positividade (%)",
                orientation="h",
                text="Texto",
                color="Taxa Positividade (%)",
                color_continuous_scale=["#00CED1", "#FF6B6B"]
            )

            fig_taxa.update_traces(textposition="outside", textfont_size=10)
            fig_taxa.update_layout(
                height=500,
                margin=dict(t=30, b=30, l=150, r=50),
                xaxis_title="Taxa de Positividade (%)",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_taxa, use_container_width=True, key="chart_taxa_substancias")
        else:
            st.info("Nenhuma substância com 10+ análises")

    st.markdown("---")

    # Gráfico de frequência (distribuição)
    st.subheader("📈 Frequência de Análises por Substância")

    df_freq = df_stats.sort_values("Total Análises", ascending=False).head(20)

    if not df_freq.empty:
        df_freq["Texto"] = df_freq.apply(
            lambda row: f"{row['Total Análises']:,}".replace(",", "."),
            axis=1
        )

        fig_freq = px.bar(
            df_freq,
            x="Substância",
            y="Total Análises",
            text="Texto",
            color="Taxa Positividade (%)",
            color_continuous_scale=["#00CED1", "#1A1A2E"]
        )

        fig_freq.update_traces(textposition="outside", textfont_size=9)
        fig_freq.update_layout(
            height=450,
            margin=dict(t=30, b=100, l=50, r=50),
            xaxis_title="",
            yaxis_title="Total de Análises",
            xaxis_tickangle=-45,
            coloraxis_showscale=True,
            coloraxis_colorbar_title="Taxa (%)"
        )

        st.plotly_chart(fig_freq, use_container_width=True, key="chart_freq_substancias")

    st.markdown("---")

    # Tabela completa
    st.subheader("📋 Tabela Completa")

    st.dataframe(
        df_stats.style.format({
            "Total Análises": "{:,.0f}",
            "Positivos": "{:,.0f}",
            "Negativos": "{:,.0f}",
            "Taxa Positividade (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # Download
    csv = df_stats.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "⬇️ Download CSV",
        csv,
        "substancias_estatisticas.csv",
        "text/csv",
        use_container_width=False
    )


def get_substance_statistics(laboratory_id: str = None, month: int = None, analysis_type: str = "all") -> dict:
    """
    Busca estatísticas de positividade por substância.
    Retorna dict {substância: {total, positivos, negativos, taxa}}
    """
    cache_key = generate_cache_key("substance_stats", laboratory_id, month, analysis_type)
    cached = get_cached_data("substance_stats", cache_key)
    if cached is not None:
        return cached

    try:
        lots_collection = get_collection("lots")
        results_collection = get_collection("results")

        # Período (usar últimos 30 dias por padrão)
        if month:
            # Se mês específico foi passado, usar o ano atual
            year = datetime.now().year
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year, 12, 31, 23, 59, 59)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        else:
            start_date = DEFAULT_START_DATE
            end_date = DEFAULT_END_DATE

        # Tipos de análise
        if analysis_type == "all":
            analysis_types = ["screening", "confirmatory", "confirmatoryTHC"]
        else:
            analysis_types = [analysis_type]

        # Filtrar amostras se necessário
        allowed_chain_ids, allowed_sample_codes = get_filtered_samples(laboratory_id, None)

        # Buscar lotes
        lots = list(lots_collection.find(
            {
                "analysisType": {"$in": analysis_types},
                "createdAt": {"$gte": start_date, "$lte": end_date}
            },
            {"code": 1, "_samples": 1}
        ))

        if not lots:
            return {}

        # Filtrar lotes
        lot_codes = []
        for lot in lots:
            lot_code = lot.get('code')
            lot_samples = lot.get('_samples', [])
            if lot_code:
                if allowed_chain_ids is not None:
                    if any(s in allowed_chain_ids for s in lot_samples):
                        lot_codes.append(lot_code)
                else:
                    lot_codes.append(lot_code)

        if not lot_codes:
            return {}

        # Buscar results
        results = list(results_collection.find(
            {"_lot": {"$in": lot_codes}},
            {"samples": 1}
        ))

        # Mapeamento de compounds
        compounds_map = get_compounds_map()

        # Contabilizar por substância
        substance_stats = {}

        for result in results:
            for sample in result.get('samples', []):
                sample_code = sample.get('_sample')

                # Filtrar amostra
                if allowed_sample_codes is not None and sample_code not in allowed_sample_codes:
                    continue

                compounds = sample.get('compounds', [])
                for compound in compounds:
                    compound_id = str(compound.get('_compound', ''))
                    is_positive = compound.get('positive', False)

                    # Nome da substância
                    substance_name = compounds_map.get(compound_id, compound_id)

                    if substance_name not in substance_stats:
                        substance_stats[substance_name] = {
                            "total": 0,
                            "positivos": 0,
                            "negativos": 0,
                            "taxa": 0.0
                        }

                    substance_stats[substance_name]["total"] += 1
                    if is_positive:
                        substance_stats[substance_name]["positivos"] += 1
                    else:
                        substance_stats[substance_name]["negativos"] += 1

        # Calcular taxas
        for name, data in substance_stats.items():
            if data["total"] > 0:
                data["taxa"] = round(data["positivos"] / data["total"] * 100, 2)

        set_cached_data("substance_stats", cache_key, substance_stats)
        return substance_stats

    except Exception as e:
        st.error(f"Erro ao buscar estatísticas de substâncias: {e}")
        return {}


def render_mapa():
    """
    Página 3 - Mapa Geográfico: regiões e cidades
    """
    st.title("🗺️ Mapa Geográfico")

    # Mostrar período selecionado
    periodo_inicio, periodo_fim = get_selected_period()
    st.info(f"📅 **Período:** {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')} *(altere na sidebar)*")

    st.markdown("---")

    # Filtros
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        meses = {
            "Todos": None, "Janeiro": 1, "Fevereiro": 2, "Março": 3,
            "Abril": 4, "Maio": 5, "Junho": 6, "Julho": 7,
            "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
        }
        selected_mes = st.selectbox("Mês", options=list(meses.keys()), index=0, key="mapa_mes")
        selected_month = meses[selected_mes]

    with col_f2:
        analysis_options = {"Todos": "all", "Triagem": "screening", "Confirmatório": "confirmatory"}
        selected_analysis = st.selectbox("Tipo de Análise", options=list(analysis_options.keys()), index=0, key="mapa_analysis")
        analysis_type = analysis_options[selected_analysis]

    st.markdown("---")

    with st.spinner("Carregando dados geográficos..."):
        geo_data = get_geographic_data(selected_month, analysis_type)

    if not geo_data:
        st.warning("Nenhum dado geográfico encontrado")
        return

    # Dados por estado
    df_estado = pd.DataFrame([
        {
            "Estado": estado,
            "Total": data["total"],
            "Positivos": data["positivos"],
            "Taxa (%)": data["taxa"]
        }
        for estado, data in geo_data.get("por_estado", {}).items()
    ]).sort_values("Total", ascending=False)

    # Dados por cidade
    df_cidade = pd.DataFrame([
        {
            "Cidade": cidade,
            "Estado": data["estado"],
            "Total": data["total"],
            "Positivos": data["positivos"],
            "Taxa (%)": data["taxa"]
        }
        for cidade, data in geo_data.get("por_cidade", {}).items()
    ]).sort_values("Total", ascending=False)

    # Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estados Ativos", len(df_estado))
    with col2:
        st.metric("Cidades Ativas", len(df_cidade))
    with col3:
        total_amostras = df_estado["Total"].sum() if not df_estado.empty else 0
        st.metric("Total Amostras", f"{total_amostras:,}".replace(",", "."))

    st.markdown("---")

    # Gráficos
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("📊 Amostras por Estado")

        if not df_estado.empty:
            df_estado["Texto"] = df_estado.apply(
                lambda row: f"{row['Total']:,} ({row['Taxa (%)']:.1f}%)".replace(",", "."), axis=1
            )

            fig_estado = px.bar(
                df_estado.head(15),
                y="Estado",
                x="Total",
                orientation="h",
                text="Texto",
                color="Taxa (%)",
                color_continuous_scale=["#00CED1", "#FF6B6B"]
            )

            fig_estado.update_traces(textposition="outside", textfont_size=9)
            fig_estado.update_layout(
                height=500,
                margin=dict(t=30, b=30, l=50, r=80),
                xaxis_title="Total de Amostras",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_estado, use_container_width=True, key="chart_mapa_estado")

    with col_g2:
        st.subheader("🏙️ Top 15 Cidades")

        if not df_cidade.empty:
            df_cidade["Texto"] = df_cidade.apply(
                lambda row: f"{row['Total']:,}".replace(",", "."), axis=1
            )

            fig_cidade = px.bar(
                df_cidade.head(15),
                y="Cidade",
                x="Total",
                orientation="h",
                text="Texto",
                color="Taxa (%)",
                color_continuous_scale=["#00CED1", "#FF6B6B"]
            )

            fig_cidade.update_traces(textposition="outside", textfont_size=9)
            fig_cidade.update_layout(
                height=500,
                margin=dict(t=30, b=30, l=120, r=50),
                xaxis_title="Total de Amostras",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_cidade, use_container_width=True, key="chart_mapa_cidade")

    st.markdown("---")

    # Tabelas
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.subheader("📋 Dados por Estado")
        st.dataframe(df_estado, use_container_width=True, hide_index=True, height=300)

    with col_t2:
        st.subheader("📋 Dados por Cidade")
        st.dataframe(df_cidade, use_container_width=True, hide_index=True, height=300)


def get_geographic_data(month: int = None, analysis_type: str = "all") -> dict:
    """
    Busca dados geográficos por estado e cidade.
    """
    cache_key = generate_cache_key("geographic", month, analysis_type)
    cached = get_cached_data("geographic_data", cache_key)
    if cached is not None:
        return cached

    try:
        # Buscar laboratórios com endereço
        labs = get_laboratories_with_address()

        # Criar mapeamento lab_id -> estado/cidade
        lab_location = {}
        for lab in labs:
            lab_location[lab['id']] = {
                "estado": lab.get('state', 'Não informado') or 'Não informado',
                "cidade": lab.get('city', 'Não informada') or 'Não informada'
            }

        # Período (usar últimos 30 dias por padrão)
        if month:
            # Se mês específico foi passado, usar o ano atual
            year = datetime.now().year
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year, 12, 31, 23, 59, 59)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        else:
            start_date = DEFAULT_START_DATE
            end_date = DEFAULT_END_DATE

        # Tipos de análise
        if analysis_type == "all":
            analysis_types = ["screening", "confirmatory", "confirmatoryTHC"]
        else:
            analysis_types = [analysis_type]

        lots_collection = get_collection("lots")
        results_collection = get_collection("results")
        gatherings_collection = get_collection("gatherings")

        # Buscar gatherings para mapear chainOfCustody -> laboratory
        gatherings = list(gatherings_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"_chainOfCustody": 1, "_laboratory": 1}
        ))

        chain_to_lab = {}
        for g in gatherings:
            chain_id = g.get('_chainOfCustody')
            lab_id = str(g.get('_laboratory', ''))
            if chain_id and lab_id:
                chain_to_lab[chain_id] = lab_id

        # Buscar lotes
        lots = list(lots_collection.find(
            {
                "analysisType": {"$in": analysis_types},
                "createdAt": {"$gte": start_date, "$lte": end_date}
            },
            {"code": 1, "_samples": 1}
        ))

        # Mapear lot_code -> chain_ids
        lot_to_chains = {}
        lot_codes = []
        for lot in lots:
            code = lot.get('code')
            samples = lot.get('_samples', [])
            if code:
                lot_codes.append(code)
                lot_to_chains[code] = samples

        if not lot_codes:
            return {"por_estado": {}, "por_cidade": {}}

        # Buscar results
        results = list(results_collection.find(
            {"_lot": {"$in": lot_codes}},
            {"_lot": 1, "samples._sample": 1, "samples.positive": 1}
        ))

        # Contabilizar por estado e cidade
        por_estado = {}
        por_cidade = {}

        for result in results:
            lot_code = result.get('_lot')
            chain_ids = lot_to_chains.get(lot_code, [])

            for sample in result.get('samples', []):
                is_positive = sample.get('positive', False)

                # Encontrar o laboratório dessa amostra
                for chain_id in chain_ids:
                    lab_id = chain_to_lab.get(chain_id)
                    if lab_id and lab_id in lab_location:
                        loc = lab_location[lab_id]
                        estado = loc["estado"]
                        cidade = loc["cidade"]

                        # Por estado
                        if estado not in por_estado:
                            por_estado[estado] = {"total": 0, "positivos": 0, "taxa": 0.0}
                        por_estado[estado]["total"] += 1
                        if is_positive:
                            por_estado[estado]["positivos"] += 1

                        # Por cidade
                        if cidade not in por_cidade:
                            por_cidade[cidade] = {"estado": estado, "total": 0, "positivos": 0, "taxa": 0.0}
                        por_cidade[cidade]["total"] += 1
                        if is_positive:
                            por_cidade[cidade]["positivos"] += 1

                        break  # Só conta uma vez por amostra

        # Calcular taxas
        for estado, data in por_estado.items():
            if data["total"] > 0:
                data["taxa"] = round(data["positivos"] / data["total"] * 100, 2)

        for cidade, data in por_cidade.items():
            if data["total"] > 0:
                data["taxa"] = round(data["positivos"] / data["total"] * 100, 2)

        result_data = {"por_estado": por_estado, "por_cidade": por_cidade}
        set_cached_data("geographic_data", cache_key, result_data)
        return result_data

    except Exception as e:
        st.error(f"Erro ao buscar dados geográficos: {e}")
        return {"por_estado": {}, "por_cidade": {}}


def render_temporal():
    """
    Página 4 - Análise Temporal: linha, barras e tendência
    """
    st.title("📈 Análise Temporal")

    # Mostrar período selecionado
    periodo_inicio, periodo_fim = get_selected_period()
    st.info(f"📅 **Período:** {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')} *(altere na sidebar)*")

    st.markdown("---")

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        laboratories_map = get_laboratories_map()
        lab_options = {"Todos": None}
        lab_options.update({name: lab_id for lab_id, name in laboratories_map.items()})
        selected_lab_name = st.selectbox("Laboratório", options=list(lab_options.keys()), index=0, key="temp_lab")
        selected_lab_id = [lab_options[selected_lab_name]] if lab_options[selected_lab_name] else None

    with col_f2:
        analysis_options = {"Triagem": "screening", "Confirmatório": "confirmatory", "Todos": "all"}
        selected_analysis = st.selectbox("Tipo de Análise", options=list(analysis_options.keys()), index=0, key="temp_analysis")
        analysis_type = analysis_options[selected_analysis]

    with col_f3:
        view_options = {"Mensal": "monthly", "Semanal": "weekly"}
        selected_view = st.selectbox("Visualização", options=list(view_options.keys()), index=0, key="temp_view")
        view_type = view_options[selected_view]

    st.markdown("---")

    with st.spinner("Carregando dados temporais..."):
        if view_type == "monthly":
            temporal_data = get_monthly_positivity_data(
                laboratory_ids=selected_lab_id,
                analysis_type=analysis_type
            )
        else:
            temporal_data = get_weekly_data(selected_lab_id, analysis_type)

    if not temporal_data:
        st.warning("Nenhum dado encontrado")
        return

    # Preparar dados
    df_temp = pd.DataFrame([
        {
            "Período": periodo,
            "Total": data["positivo"] + data["negativo"],
            "Positivos": data["positivo"],
            "Negativos": data["negativo"],
            "Taxa (%)": data["taxa"]
        }
        for periodo, data in temporal_data.items()
    ])

    # Cards
    col1, col2, col3, col4 = st.columns(4)

    total_geral = df_temp["Total"].sum()
    total_positivos = df_temp["Positivos"].sum()
    taxa_media = df_temp["Taxa (%)"].mean()
    maior_taxa = df_temp["Taxa (%)"].max()

    with col1:
        st.metric("Total Amostras", f"{total_geral:,}".replace(",", "."))
    with col2:
        st.metric("Total Positivos", f"{total_positivos:,}".replace(",", "."))
    with col3:
        st.metric("Taxa Média", f"{taxa_media:.2f}%")
    with col4:
        st.metric("Maior Taxa", f"{maior_taxa:.2f}%")

    st.markdown("---")

    # Gráfico de linha - Taxa de Positividade
    st.subheader("📉 Evolução da Taxa de Positividade")

    fig_linha = px.line(
        df_temp,
        x="Período",
        y="Taxa (%)",
        markers=True,
        line_shape="spline"
    )

    fig_linha.update_traces(
        line=dict(color="#FF6B6B", width=3),
        marker=dict(size=8, color="#FF6B6B")
    )

    # Adicionar linha de tendência
    if len(df_temp) > 1:
        z = np.polyfit(range(len(df_temp)), df_temp["Taxa (%)"], 1)
        p = np.poly1d(z)
        df_temp["Tendência"] = p(range(len(df_temp)))

        fig_linha.add_scatter(
            x=df_temp["Período"],
            y=df_temp["Tendência"],
            mode="lines",
            name="Tendência",
            line=dict(color="#00CED1", width=2, dash="dash")
        )

    fig_linha.update_layout(
        height=400,
        margin=dict(t=30, b=50, l=50, r=50),
        xaxis_title="",
        yaxis_title="Taxa de Positividade (%)",
        xaxis_tickangle=-45,
        showlegend=True
    )

    st.plotly_chart(fig_linha, use_container_width=True, key="chart_temporal_linha")

    st.markdown("---")

    # Gráfico de barras - Total de Amostras
    st.subheader("📊 Total de Amostras por Período")

    df_temp["Texto"] = df_temp.apply(
        lambda row: f"{row['Total']:,} ({row['Taxa (%)']:.1f}%)".replace(",", "."), axis=1
    )

    fig_barras = px.bar(
        df_temp,
        x="Período",
        y="Total",
        text="Texto",
        color="Taxa (%)",
        color_continuous_scale=["#00CED1", "#FF6B6B"]
    )

    fig_barras.update_traces(textposition="outside", textfont_size=9)
    fig_barras.update_layout(
        height=400,
        margin=dict(t=30, b=50, l=50, r=50),
        xaxis_title="",
        yaxis_title="Total de Amostras",
        xaxis_tickangle=-45,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_barras, use_container_width=True, key="chart_temporal_barras")

    st.markdown("---")

    # Gráfico empilhado - Positivos vs Negativos
    st.subheader("📈 Positivos vs Negativos")

    fig_stack = go.Figure()

    fig_stack.add_trace(go.Bar(
        name='Negativos',
        x=df_temp["Período"],
        y=df_temp["Negativos"],
        marker_color='#00CED1'
    ))

    fig_stack.add_trace(go.Bar(
        name='Positivos',
        x=df_temp["Período"],
        y=df_temp["Positivos"],
        marker_color='#FF6B6B'
    ))

    fig_stack.update_layout(
        barmode='stack',
        height=400,
        margin=dict(t=30, b=50, l=50, r=50),
        xaxis_title="",
        yaxis_title="Quantidade",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_stack, use_container_width=True, key="chart_temporal_stack")


def get_weekly_data(laboratory_ids: list = None, analysis_type: str = "screening") -> dict:
    """
    Busca dados semanais de positividade.
    """
    cache_key = generate_cache_key("weekly_data", laboratory_ids, analysis_type)
    cached = get_cached_data("weekly_data", cache_key)
    if cached is not None:
        return cached

    try:
        lots_collection = get_collection("lots")
        results_collection = get_collection("results")

        # Usar período padrão (últimos 30 dias)
        start_date = DEFAULT_START_DATE
        end_date = DEFAULT_END_DATE

        # Tipos de análise
        if analysis_type == "all":
            analysis_types = ["screening", "confirmatory", "confirmatoryTHC"]
        else:
            analysis_types = [analysis_type]

        # Filtrar amostras
        allowed_chain_ids, allowed_sample_codes = get_filtered_samples_advanced(
            laboratory_ids=laboratory_ids,
            start_date=start_date,
            end_date=end_date
        )

        weekly_data = {}

        # Iterar por semanas
        current = start_date
        week_num = 1

        while current <= end_date:
            week_start = current
            week_end = min(current + timedelta(days=6, hours=23, minutes=59, seconds=59), end_date)

            lots = list(lots_collection.find(
                {
                    "analysisType": {"$in": analysis_types},
                    "createdAt": {"$gte": week_start, "$lte": week_end}
                },
                {"code": 1, "_samples": 1}
            ))

            if lots:
                lot_codes = []
                for lot in lots:
                    lot_code = lot.get('code')
                    lot_samples = lot.get('_samples', [])
                    if lot_code:
                        if allowed_chain_ids is not None:
                            if any(s in allowed_chain_ids for s in lot_samples):
                                lot_codes.append(lot_code)
                        else:
                            lot_codes.append(lot_code)

                positivo = 0
                negativo = 0

                if lot_codes:
                    results = list(results_collection.find(
                        {"_lot": {"$in": lot_codes}},
                        {"samples._sample": 1, "samples.positive": 1}
                    ))

                    for result in results:
                        for sample in result.get('samples', []):
                            sample_code = sample.get('_sample')
                            if allowed_sample_codes is not None and sample_code not in allowed_sample_codes:
                                continue
                            if sample.get('positive', False):
                                positivo += 1
                            else:
                                negativo += 1

                total = positivo + negativo
                taxa = (positivo / total * 100) if total > 0 else 0

                week_label = f"Sem {week_num}"
                weekly_data[week_label] = {
                    "positivo": positivo,
                    "negativo": negativo,
                    "taxa": round(taxa, 2)
                }

            current += timedelta(days=7)
            week_num += 1

        set_cached_data("weekly_data", cache_key, weekly_data)
        return weekly_data

    except Exception as e:
        st.error(f"Erro ao buscar dados semanais: {e}")
        return {}


def render_tabela_detalhada():
    """
    Página 5 - Tabela Detalhada com Exportação
    """
    st.title("📋 Tabela Detalhada")

    # Mostrar período selecionado
    periodo_inicio, periodo_fim = get_selected_period()
    st.info(f"📅 **Período:** {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')} *(altere na sidebar)*")

    st.markdown("---")

    # Filtros
    st.markdown("### 🔍 Filtros")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        filtro_lote = st.text_input("Lote", placeholder="Digite o código do lote...", key="det_lote")

    with col2:
        filtro_amostra = st.text_input("Código da Amostra", placeholder="Digite o código...", key="det_amostra")

    with col3:
        filtro_positivas = st.selectbox("Resultado", ["Todos", "Positivo", "Negativo"], key="det_resultado")

    with col4:
        compounds_map = get_compounds_map()
        substancias_opcoes = ['Todas'] + list(compounds_map.values())
        filtro_substancia = st.selectbox("Substância Positiva", substancias_opcoes, key="det_substancia")

    st.markdown("---")

    with st.spinner("Carregando dados..."):
        df = get_substance_data()

    if df.empty:
        st.warning("⚠️ Nenhum dado encontrado")
        return

    # Colunas de substâncias (excluindo colunas de metadados)
    substance_cols = [col for col in df.columns if col not in ['Data', 'Lote', 'Tipo de Lote', 'Amostra']]

    # Aplicar filtros
    df_filtrado = df.copy()

    if filtro_lote:
        df_filtrado = df_filtrado[df_filtrado['Lote'].str.contains(filtro_lote, case=False, na=False)]

    if filtro_amostra:
        df_filtrado = df_filtrado[df_filtrado['Amostra'].str.contains(filtro_amostra, case=False, na=False)]

    if filtro_positivas == 'Positivo':
        df_filtrado = df_filtrado[df_filtrado[substance_cols].apply(lambda row: (row == 'Positivo').any(), axis=1)]
    elif filtro_positivas == 'Negativo':
        df_filtrado = df_filtrado[df_filtrado[substance_cols].apply(lambda row: (row == 'Negativo').all(), axis=1)]

    if filtro_substancia != 'Todas':
        df_filtrado = df_filtrado[df_filtrado[filtro_substancia] == 'Positivo']

    # Estatísticas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Registros", f"{len(df_filtrado):,}".replace(",", "."))

    with col2:
        total_lotes = df_filtrado['Lote'].nunique()
        st.metric("Lotes Únicos", f"{total_lotes:,}".replace(",", "."))

    with col3:
        amostras_positivas = df_filtrado[df_filtrado[substance_cols].apply(lambda row: (row == 'Positivo').any(), axis=1)].shape[0]
        st.metric("Amostras Positivas", f"{amostras_positivas:,}".replace(",", "."))

    with col4:
        taxa = (amostras_positivas / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
        st.metric("Taxa de Positividade", f"{taxa:.2f}%")

    st.markdown("---")

    # Função de estilização para a tabela
    def style_table(df):
        """
        Aplica estilização à tabela:
        - Linhas com pelo menos uma substância positiva: fundo vermelho, texto preto
        - Células de substâncias positivas: texto branco, negrito
        """
        # Colunas de substâncias
        sub_cols = [col for col in df.columns if col not in ['Data', 'Lote', 'Tipo de Lote', 'Amostra']]

        # Criar DataFrame de estilos
        styles = pd.DataFrame('', index=df.index, columns=df.columns)

        for idx in df.index:
            row = df.loc[idx]
            # Verificar se a linha tem alguma substância positiva
            has_positive = any(row[col] == 'Positivo' for col in sub_cols if col in row.index)

            if has_positive:
                # Aplicar fundo vermelho claro em toda a linha
                for col in df.columns:
                    styles.loc[idx, col] = 'background-color: #ffcccc; color: black;'

                # Aplicar estilo especial nas células de substâncias positivas
                for col in sub_cols:
                    if col in row.index and row[col] == 'Positivo':
                        styles.loc[idx, col] = 'background-color: #dc3545; color: white; font-weight: bold;'

        return styles

    # Tabela com estilo
    if not df_filtrado.empty:
        styled_df = df_filtrado.style.apply(lambda _: style_table(df_filtrado), axis=None)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)
    else:
        st.dataframe(df_filtrado, use_container_width=True, hide_index=True, height=500)

    # Exportação
    st.markdown("### ⬇️ Exportar Dados")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        csv = df_filtrado.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📄 Download CSV",
            csv,
            "amostras_detalhadas.csv",
            "text/csv",
            use_container_width=True
        )

    with col_exp2:
        # Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_filtrado.to_excel(writer, index=False, sheet_name='Amostras')
        excel_data = output.getvalue()

        st.download_button(
            "📊 Download Excel",
            excel_data,
            "amostras_detalhadas.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col_exp3:
        json_data = df_filtrado.to_json(orient='records', force_ascii=False, indent=2)
        st.download_button(
            "📋 Download JSON",
            json_data,
            "amostras_detalhadas.json",
            "application/json",
            use_container_width=True
        )

    st.caption(f"Total: {len(df_filtrado):,} registros".replace(",", "."))


def render_auditoria():
    """
    Página 6 - Auditoria e Anomalias
    """
    st.title("🔍 Auditoria e Anomalias")

    # Mostrar período selecionado
    periodo_inicio, periodo_fim = get_selected_period()
    st.info(f"📅 **Período:** {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')} *(altere na sidebar)*")

    st.markdown("---")

    with st.spinner("Analisando dados..."):
        anomalias = detect_anomalies()

    if not anomalias:
        st.success("✅ Nenhuma anomalia detectada!")
        return

    # Resumo
    total_anomalias = sum(len(v) for v in anomalias.values())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de Anomalias", total_anomalias)

    with col2:
        tipos = len([k for k, v in anomalias.items() if v])
        st.metric("Tipos de Anomalias", tipos)

    with col3:
        severidade = "Alta" if total_anomalias > 50 else "Média" if total_anomalias > 10 else "Baixa"
        st.metric("Severidade Geral", severidade)

    st.markdown("---")

    # Detalhamento por tipo
    if anomalias.get("taxas_extremas"):
        with st.expander(f"⚠️ Taxas de Positividade Extremas ({len(anomalias['taxas_extremas'])} encontradas)", expanded=True):
            st.markdown("Laboratórios com taxas fora do padrão (muito alta ou muito baixa)")
            df_taxas = pd.DataFrame(anomalias["taxas_extremas"])
            st.dataframe(df_taxas, use_container_width=True, hide_index=True)

    if anomalias.get("volumes_atipicos"):
        with st.expander(f"📊 Volumes Atípicos ({len(anomalias['volumes_atipicos'])} encontrados)", expanded=True):
            st.markdown("Períodos com volume de amostras muito diferente da média")
            df_volumes = pd.DataFrame(anomalias["volumes_atipicos"])
            st.dataframe(df_volumes, use_container_width=True, hide_index=True)

    if anomalias.get("substancias_raras"):
        with st.expander(f"🧪 Substâncias Raras ({len(anomalias['substancias_raras'])} encontradas)", expanded=True):
            st.markdown("Substâncias com positividade muito alta ou muito baixa")
            df_subst = pd.DataFrame(anomalias["substancias_raras"])
            st.dataframe(df_subst, use_container_width=True, hide_index=True)

    if anomalias.get("dados_faltantes"):
        with st.expander(f"❌ Dados Faltantes ({len(anomalias['dados_faltantes'])} encontrados)", expanded=True):
            st.markdown("Registros com informações incompletas")
            df_faltantes = pd.DataFrame(anomalias["dados_faltantes"])
            st.dataframe(df_faltantes, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Exportar relatório
    st.markdown("### 📥 Exportar Relatório de Auditoria")

    report_data = {
        "data_analise": datetime.now().isoformat(),
        "total_anomalias": total_anomalias,
        "anomalias": anomalias
    }

    json_report = json.dumps(report_data, ensure_ascii=False, indent=2, default=str)

    st.download_button(
        "📋 Download Relatório JSON",
        json_report,
        "relatorio_auditoria.json",
        "application/json",
        use_container_width=False
    )


def detect_anomalies() -> dict:
    """
    Detecta anomalias nos dados.
    """
    cache_key = "anomalies_detection"
    cached = get_cached_data("anomalies", cache_key)
    if cached is not None:
        return cached

    anomalias = {
        "taxas_extremas": [],
        "volumes_atipicos": [],
        "substancias_raras": [],
        "dados_faltantes": []
    }

    try:
        # 1. Verificar taxas extremas por laboratório
        labs_map = get_laboratories_map()

        for lab_id, lab_name in labs_map.items():
            metrics = get_metrics_data(laboratory_ids=[lab_id])

            total = metrics.get("total_amostras", 0)
            taxa = metrics.get("taxa_geral", 0)

            if total > 100:  # Mínimo de amostras para considerar
                if taxa > 25:  # Taxa muito alta
                    anomalias["taxas_extremas"].append({
                        "Laboratório": lab_name,
                        "Total Amostras": total,
                        "Taxa (%)": f"{taxa:.2f}%",
                        "Tipo": "Taxa muito alta"
                    })
                elif taxa == 0 and total > 500:  # Nenhum positivo com muitas amostras
                    anomalias["taxas_extremas"].append({
                        "Laboratório": lab_name,
                        "Total Amostras": total,
                        "Taxa (%)": "0%",
                        "Tipo": "Nenhum positivo"
                    })

        # 2. Verificar volumes atípicos por mês
        monthly_data = get_monthly_positivity_data()

        if monthly_data:
            volumes = [data["positivo"] + data["negativo"] for data in monthly_data.values()]
            if volumes:
                media = sum(volumes) / len(volumes)
                desvio = (sum((v - media) ** 2 for v in volumes) / len(volumes)) ** 0.5

                for mes, data in monthly_data.items():
                    volume = data["positivo"] + data["negativo"]
                    if desvio > 0:
                        z_score = (volume - media) / desvio
                        if abs(z_score) > 2:  # Mais de 2 desvios padrão
                            anomalias["volumes_atipicos"].append({
                                "Mês": mes,
                                "Volume": volume,
                                "Média": int(media),
                                "Desvio": f"{z_score:.2f}σ",
                                "Tipo": "Volume alto" if z_score > 0 else "Volume baixo"
                            })

        # 3. Verificar substâncias com taxas extremas
        substance_stats = get_substance_statistics()

        for subst, data in substance_stats.items():
            if data["total"] >= 50:  # Mínimo de análises
                if data["taxa"] > 50:  # Taxa muito alta para substância
                    anomalias["substancias_raras"].append({
                        "Substância": subst,
                        "Total Análises": data["total"],
                        "Positivos": data["positivos"],
                        "Taxa (%)": f"{data['taxa']:.2f}%",
                        "Tipo": "Taxa muito alta"
                    })

        set_cached_data("anomalies", cache_key, anomalias)
        return anomalias

    except Exception as e:
        st.error(f"Erro ao detectar anomalias: {e}")
        return anomalias


# ============================================
# PÁGINA 7 - REDE (COMPARAÇÃO ENTRE FILIAIS)
# ============================================

def get_networks_by_cnpj() -> dict:
    """
    Identifica redes de laboratórios pelos primeiros 8 dígitos do CNPJ.
    Retorna dict {cnpj_base: [{id, name, cnpj, city, state}]}
    """
    labs = get_laboratories_with_address()

    networks = {}
    for lab in labs:
        cnpj = lab.get('cnpj', '')
        # Limpar CNPJ (remover pontuação)
        cnpj_limpo = ''.join(filter(str.isdigit, str(cnpj)))

        if len(cnpj_limpo) >= 8:
            # Primeiros 8 dígitos identificam a matriz/rede
            cnpj_base = cnpj_limpo[:8]

            if cnpj_base not in networks:
                networks[cnpj_base] = []

            networks[cnpj_base].append({
                "id": lab.get('id'),
                "name": lab.get('name', 'Desconhecido'),
                "cnpj": cnpj,
                "city": lab.get('city', ''),
                "state": lab.get('state', '')
            })

    # Filtrar apenas redes com mais de 1 filial
    networks_filtradas = {k: v for k, v in networks.items() if len(v) > 1}

    return networks_filtradas


def get_network_comparison_data(network_labs: list) -> list:
    """
    Busca dados de comparação entre filiais de uma rede.
    Retorna lista com métricas de cada filial.
    """
    comparison_data = []

    for lab in network_labs:
        lab_id = lab.get('id')
        lab_name = lab.get('name', 'Desconhecido')
        lab_city = lab.get('city', '')
        lab_state = lab.get('state', '')

        # Buscar métricas do laboratório
        metrics = get_metrics_data(laboratory_ids=[lab_id])

        comparison_data.append({
            "id": lab_id,
            "nome": lab_name,
            "cidade": lab_city,
            "estado": lab_state,
            "total_amostras": metrics.get("total_amostras", 0),
            "positivas_triagem": metrics.get("positivas_triagem", 0),
            "negativas_triagem": metrics.get("negativas_triagem", 0),
            "positivas_confirmatorio": metrics.get("positivas_confirmatorio", 0),
            "negativas_confirmatorio": metrics.get("negativas_confirmatorio", 0),
            "taxa_positividade": metrics.get("taxa_geral", 0)
        })

    return comparison_data


def render_rede():
    """
    Página 7 - Comparação entre Filiais da Rede
    """
    st.title("🏢 Rede - Comparação entre Filiais")

    # Mostrar período selecionado
    periodo_inicio, periodo_fim = get_selected_period()
    st.info(f"📅 **Período:** {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')} *(altere na sidebar)*")

    # Buscar redes identificadas por CNPJ
    with st.spinner("Identificando redes por CNPJ..."):
        networks = get_networks_by_cnpj()

    if not networks:
        st.warning("Nenhuma rede identificada. Redes são identificadas quando há mais de um laboratório com os mesmos 8 primeiros dígitos do CNPJ.")
        return

    st.markdown("---")

    # Criar opções de seleção de rede
    network_options = {}
    for cnpj_base, labs in networks.items():
        # Usar o nome do primeiro lab como referência da rede
        primeiro_lab = labs[0]['name']
        qtd_filiais = len(labs)
        network_options[f"{primeiro_lab} ({qtd_filiais} filiais) - CNPJ base: {cnpj_base}"] = cnpj_base

    # Seletor de rede
    rede_selecionada = st.selectbox(
        "Selecione a Rede:",
        options=list(network_options.keys()),
        key="select_rede"
    )

    if not rede_selecionada:
        return

    cnpj_base = network_options[rede_selecionada]
    labs_da_rede = networks[cnpj_base]

    st.markdown("---")

    # Buscar dados de comparação
    with st.spinner("Carregando dados das filiais..."):
        comparison_data = get_network_comparison_data(labs_da_rede)

    if not comparison_data:
        st.warning("Nenhum dado encontrado para as filiais desta rede no período selecionado.")
        return

    # Criar DataFrame para exibição
    df_comparison = pd.DataFrame(comparison_data)

    # Criar labels para os gráficos
    df_comparison['label'] = df_comparison.apply(
        lambda x: f"{x['cidade']}/{x['estado']}" if x['cidade'] and x['estado'] else x['nome'][:20],
        axis=1
    )

    # Ordenar por volume de amostras (maior primeiro)
    df_chart = df_comparison.sort_values('total_amostras', ascending=False)
    taxa_media_rede = df_comparison['taxa_positividade'].mean()

    # Gráfico combinado: Barras (Amostras) + Marcadores (Taxa)
    st.subheader("Comparativo: Volume de Amostras x Taxa de Positividade")

    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Barras - Volume de Amostras
    fig.add_trace(
        go.Bar(
            x=df_chart['label'],
            y=df_chart['total_amostras'],
            name='Amostras',
            marker_color='#1A1A2E',
            text=df_chart['total_amostras'].apply(lambda x: f'{x:,}'.replace(',', '.')),
            textposition='outside',
            textfont=dict(size=11)
        ),
        secondary_y=False
    )

    # Cores dos marcadores baseadas na média
    cores_marcador = []
    for taxa in df_chart['taxa_positividade']:
        if taxa > taxa_media_rede * 1.5:
            cores_marcador.append('#DC3545')  # Vermelho
        elif taxa > taxa_media_rede:
            cores_marcador.append('#FFA500')  # Laranja
        else:
            cores_marcador.append('#28A745')  # Verde

    # Linha + Marcadores - Taxa de Positividade
    fig.add_trace(
        go.Scatter(
            x=df_chart['label'],
            y=df_chart['taxa_positividade'],
            name='Taxa (%)',
            mode='lines+markers+text',
            marker=dict(size=14, color=cores_marcador, line=dict(width=2, color='white')),
            line=dict(color='#666', width=2, dash='dot'),
            text=df_chart['taxa_positividade'].apply(lambda x: f'{x:.1f}%'),
            textposition='top center',
            textfont=dict(size=11, color='#333')
        ),
        secondary_y=True
    )

    # Linha da média da taxa
    fig.add_hline(
        y=taxa_media_rede,
        line_dash="dash",
        line_color="#999",
        annotation_text=f"Média: {taxa_media_rede:.1f}%",
        annotation_position="right",
        secondary_y=True
    )

    fig.update_layout(
        height=450,
        margin=dict(t=30, b=80, l=60, r=60),
        plot_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(
            tickangle=-45,
            gridcolor='#eee'
        )
    )

    fig.update_yaxes(
        title_text="Volume de Amostras",
        secondary_y=False,
        gridcolor='#eee'
    )

    fig.update_yaxes(
        title_text="Taxa de Positividade (%)",
        secondary_y=True,
        range=[0, max(df_chart['taxa_positividade'].max() * 1.5, 10)]
    )

    st.plotly_chart(fig, use_container_width=True, key="chart_rede_combinado")

    # Legenda de cores
    st.markdown("""
    <div style="display: flex; gap: 20px; justify-content: center; margin: 10px 0; font-size: 13px;">
        <span><strong>Taxa:</strong></span>
        <span style="color: #28A745;">● Abaixo da média</span>
        <span style="color: #FFA500;">● Acima da média</span>
        <span style="color: #DC3545;">● Muito acima (+50%)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabela detalhada
    st.subheader("Tabela Comparativa Detalhada")

    # Preparar DataFrame para exibição
    df_display = df_comparison[[
        'nome', 'cidade', 'estado', 'total_amostras',
        'positivas_triagem', 'negativas_triagem', 'taxa_positividade'
    ]].copy()

    df_display.columns = [
        'Filial', 'Cidade', 'Estado', 'Total Amostras',
        'Positivas Triagem', 'Negativas Triagem', 'Taxa Positividade (%)'
    ]

    # Formatar taxa
    df_display['Taxa Positividade (%)'] = df_display['Taxa Positividade (%)'].apply(lambda x: f"{x:.2f}%")

    # Ordenar por total de amostras
    df_display = df_display.sort_values('Total Amostras', ascending=False)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )

    # Exportar dados
    st.markdown("---")
    st.subheader("📥 Exportar Dados da Rede")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        csv_buffer = io.StringIO()
        df_display.to_csv(csv_buffer, index=False, sep=';', encoding='utf-8')
        st.download_button(
            "📊 Download CSV",
            csv_buffer.getvalue(),
            f"rede_{cnpj_base}_comparacao.csv",
            "text/csv",
            use_container_width=True
        )

    with col_exp2:
        json_data = df_comparison.to_json(orient='records', force_ascii=False)
        st.download_button(
            "📋 Download JSON",
            json_data,
            f"rede_{cnpj_base}_comparacao.json",
            "application/json",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
