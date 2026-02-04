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
import time
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


def loading_with_progress(tasks: list, message: str = "Carregando dados...", silent: bool = False):
    """
    Executa uma lista de tarefas mostrando progress bar usando placeholder.

    Args:
        tasks: lista de tuplas [(nome, funcao, args, kwargs), ...]
               ou [(nome, funcao), ...] para funções sem argumentos
        message: mensagem exibida durante o carregamento
        silent: se True, não mostra progress bar (carregamento silencioso)

    Returns:
        dict com {nome: resultado}
    """
    results = {}
    total = len(tasks)

    # Usar placeholder para evitar blur/dimming (só se não for silencioso)
    placeholder = st.empty() if not silent else None

    for i, task in enumerate(tasks):
        if len(task) == 2:
            name, func = task
            args, kwargs = (), {}
        elif len(task) == 3:
            name, func, args = task
            kwargs = {}
        else:
            name, func, args, kwargs = task

        # Atualizar progress bar dentro do placeholder (só se não for silencioso)
        if placeholder:
            progress = (i) / total
            placeholder.progress(progress, text=f"{message} ({name})")

        try:
            if args and kwargs:
                results[name] = func(*args, **kwargs)
            elif args:
                results[name] = func(*args)
            elif kwargs:
                results[name] = func(**kwargs)
            else:
                results[name] = func()
        except Exception as e:
            results[name] = None
            if not silent:
                st.error(f"Erro ao carregar {name}: {e}")

    # Completar progress bar (só se não for silencioso)
    if placeholder:
        placeholder.progress(1.0, text="Concluído!")
        time.sleep(0.3)  # Pequeno delay para mostrar "Concluído!"
        placeholder.empty()  # Remove o placeholder

    return results


def loading_single(func, message: str = "Carregando...", *args, **kwargs):
    """
    Executa uma única função mostrando progress bar simples usando placeholder.

    Args:
        func: função a executar
        message: mensagem exibida durante o carregamento
        *args, **kwargs: argumentos para a função

    Returns:
        resultado da função
    """
    # Usar placeholder para evitar blur/dimming
    placeholder = st.empty()

    # Simular progresso inicial
    placeholder.progress(0.3, text=message)

    try:
        result = func(*args, **kwargs)
        placeholder.progress(1.0, text="Concluído!")
        time.sleep(0.2)
        placeholder.empty()
        return result
    except Exception as e:
        placeholder.empty()
        raise e


def show_loading_spinner(page_name: str):
    """
    Mostra spinner central de carregamento com overlay que esconde o conteúdo.
    """
    st.markdown(f"""
    <div class="loading-overlay" id="loading-overlay">
        <div class="loading-spinner"></div>
        <div class="loading-text">Carregando {page_name}...</div>
    </div>
    """, unsafe_allow_html=True)


def hide_loading_spinner():
    """
    Injeta script para remover o overlay de carregamento.
    """
    st.markdown("""
    <script>
        var overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(function() {
                overlay.style.display = 'none';
            }, 300);
        }
    </script>
    <style>
        #loading-overlay {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)


def render_page_with_loading(page_name: str, render_func: callable):
    """
    Wrapper que mostra spinner central antes de renderizar a página.
    Esconde o conteúdo durante carregamento e mostra após finalizar.
    """
    # Placeholder para o loader
    loader_placeholder = st.empty()

    # Mostrar spinner imediatamente
    with loader_placeholder:
        show_loading_spinner(page_name)

    # Renderizar conteúdo
    render_func()

    # Remover loader e mostrar conteúdo
    loader_placeholder.empty()


def render_page_with_unified_loading(page_name: str, render_func: callable):
    """
    Carrega dados base + renderiza página em um único spinner.
    Evita múltiplos indicadores de carregamento aparecendo sequencialmente.
    """
    # Verificar se precisa carregar dados base
    needs_base_data = not st.session_state.get('base_data_loaded', False)

    # Se precisa carregar dados, mostra spinner único para tudo
    if needs_base_data:
        # Placeholder para o loader
        loader_placeholder = st.empty()

        # Mostrar spinner imediatamente
        with loader_placeholder:
            show_loading_spinner(page_name)

        # Carregar dados base silenciosamente (sem progress bar próprio)
        load_base_data(silent=True)

        # Renderizar conteúdo
        render_func()

        # Remover loader
        loader_placeholder.empty()
    else:
        # Dados já carregados, só renderiza a página
        render_page_with_loading(page_name, render_func)


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


def load_base_data(silent: bool = False):
    """
    Carrega dados básicos que são usados em múltiplas páginas.
    Chamado uma vez no início da sessão.
    Pré-carrega TODOS os dados do período padrão para acelerar consultas subsequentes.

    Args:
        silent: se True, não mostra progress bar (usado quando já há spinner global)
    """
    init_session_cache()

    if st.session_state.base_data_loaded:
        return

    # Garantir índices (só executa se necessário)
    ensure_indexes()

    # Pré-carregar dados (silencioso se já houver spinner global)
    preload_all_data(silent=silent)

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
                    "compounds": sample.get('_compound', []) or sample.get('compounds', [])
                })

        return results_dict
    except Exception as e:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def get_all_gatherings(start_date: datetime = None, end_date: datetime = None):
    """
    Pré-carrega mapeamento de chainOfCustody -> laboratory do período selecionado.
    Retorna dict com chain_to_lab, chain_to_purpose, chain_to_subtype e chain_to_date
    """
    try:
        gatherings_collection = get_collection("gatherings")
        if start_date is None or end_date is None:
            start_date, end_date = get_default_period()

        gatherings = list(gatherings_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"_chainOfCustody": 1, "_laboratory": 1, "purpose.type": 1, "purpose.subType": 1, "createdAt": 1}
        ))

        chain_to_lab = {}
        chain_to_purpose = {}
        chain_to_subtype = {}
        chain_to_date = {}

        for g in gatherings:
            chain_id = g.get('_chainOfCustody')
            lab_id = g.get('_laboratory')
            purpose = g.get('purpose', {})
            purpose_type = purpose.get('type') if purpose else None
            purpose_subtype = purpose.get('subType') if purpose else None
            created_at = g.get('createdAt')

            if chain_id:
                if lab_id:
                    chain_to_lab[chain_id] = str(lab_id) if isinstance(lab_id, ObjectId) else lab_id
                if purpose_type:
                    chain_to_purpose[chain_id] = purpose_type
                if purpose_subtype:
                    chain_to_subtype[chain_id] = purpose_subtype
                if created_at:
                    chain_to_date[chain_id] = created_at

        return {
            "chain_to_lab": chain_to_lab,
            "chain_to_purpose": chain_to_purpose,
            "chain_to_subtype": chain_to_subtype,
            "chain_to_date": chain_to_date
        }
    except Exception as e:
        return {"chain_to_lab": {}, "chain_to_purpose": {}, "chain_to_subtype": {}, "chain_to_date": {}}


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


def preload_all_data(silent: bool = False):
    """
    Pré-carrega todos os dados necessários usando paralelização.
    Chamado uma vez no início para popular o cache.
    Usa o período selecionado pelo usuário.

    Args:
        silent: se True, não mostra progress bar (usado quando já há spinner global)
    """
    # Obter período selecionado
    start_date, end_date = get_selected_period()

    results = {}

    # Funções que podem rodar em paralelo (não dependem umas das outras)
    parallel_tasks = {
        "Lotes": lambda: get_all_lots(start_date, end_date),
        "Coletas": lambda: get_all_gatherings(start_date, end_date),
        "Mapeamento de amostras": lambda: get_chain_to_sample_map(start_date, end_date),
        "Dados RENACH": lambda: get_renach_data_cached(start_date, end_date),
        "Substâncias": get_compounds_map,
        "Laboratórios": get_laboratories_map,
        "Endereços": get_laboratories_with_address,
    }

    total_tasks = len(parallel_tasks) + 1  # +1 para Resultados (depende de Lotes)
    completed = 0

    # Criar placeholder para progress (se não silent)
    progress_placeholder = None
    if not silent:
        progress_placeholder = st.empty()
        progress_placeholder.progress(0, text="Carregando dados iniciais...")

    # Executar tarefas em paralelo
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(func): name for name, func in parallel_tasks.items()}

        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = None
                if not silent:
                    st.error(f"Erro ao carregar {name}: {e}")

            completed += 1
            if progress_placeholder:
                progress_placeholder.progress(
                    completed / total_tasks,
                    text=f"Carregando {name}..."
                )

    # Carregar Resultados DEPOIS de Lotes (depende de lotes para filtrar)
    if progress_placeholder:
        progress_placeholder.progress(
            completed / total_tasks,
            text="Carregando Resultados..."
        )
    try:
        results["Resultados"] = get_all_results(start_date, end_date)
    except Exception as e:
        results["Resultados"] = None
        if not silent:
            st.error(f"Erro ao carregar Resultados: {e}")

    completed += 1
    if progress_placeholder:
        progress_placeholder.progress(1.0, text="Dados carregados!")
        time.sleep(0.3)
        progress_placeholder.empty()

    return results


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


def get_compound_type_map() -> dict:
    """
    Retorna um mapeamento de compound ObjectID para tipo de substância.
    Tipos: Ilícito, Medicamento, Psicoativo legal
    """
    return {
        # Ilícito
        "5cdf0e844040d61c3c496407": "Ilícito",  # 6-MAM (Heroína)
        "5cdf0e844040d61c3c496408": "Ilícito",  # MDMA (Ecstasy)
        "5cdf0e844040d61c3c496409": "Ilícito",  # MDA
        "5cdf0e844040d61c3c49640a": "Ilícito",  # Anfetamina
        "5cdf0e844040d61c3c49640b": "Ilícito",  # Metanfetamina
        "5cdf0e844040d61c3c49640c": "Ilícito",  # THC
        "5cdf0e844040d61c3c49640d": "Ilícito",  # Cocaína
        "5cdf0e844040d61c3c49640e": "Ilícito",  # Benzoilecgonina
        "5cdf0e844040d61c3c49640f": "Ilícito",  # Cocaetileno
        "5cdf0e844040d61c3c496415": "Ilícito",  # Cetamina
        # Medicamento
        "5cdf0e844040d61c3c496410": "Medicamento",  # Codeína
        "5cdf0e844040d61c3c496411": "Medicamento",  # Morfina
        "5cdf0e844040d61c3c496412": "Medicamento",  # Tramadol
        "5cdf0e844040d61c3c496413": "Medicamento",  # Fentanil
        "5cdf0e844040d61c3c496414": "Medicamento",  # Metadona
        "5cdf0e844040d61c3c496416": "Medicamento",  # Zolpidem
    }


def get_compounds_by_type(compound_type: str) -> list:
    """
    Retorna a lista de ObjectIDs de compounds de um determinado tipo.
    compound_type: 'Ilícito', 'Medicamento', 'Psicoativo legal', ou 'Todos'
    """
    type_map = get_compound_type_map()
    if compound_type == 'Todos':
        return list(type_map.keys())
    return [cid for cid, ctype in type_map.items() if ctype == compound_type]


def get_compound_names_by_type(compound_type: str) -> list:
    """
    Retorna a lista de nomes de compounds de um determinado tipo.
    """
    compounds_map = get_compounds_map()
    compound_ids = get_compounds_by_type(compound_type)
    return [compounds_map.get(cid, 'Desconhecido') for cid in compound_ids if cid in compounds_map]


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


@st.cache_data(ttl=3600)
def get_laboratories_by_cnpj() -> dict:
    """
    Retorna dicionário de laboratórios indexado por CNPJ.
    Formato: {cnpj: {id, name, city, state}}
    Laboratórios sem CNPJ são ignorados.
    """
    labs = get_laboratories_with_address()
    labs_by_cnpj = {}
    for lab in labs:
        cnpj = lab.get('cnpj', '').strip()
        if cnpj:
            labs_by_cnpj[cnpj] = {
                "id": lab.get('id'),
                "name": lab.get('name', 'Desconhecido'),
                "city": lab.get('city', ''),
                "state": lab.get('state', '')
            }
    return labs_by_cnpj


# Mapeamento de finalidades (purpose.type no banco -> nome exibido)
PURPOSE_MAP = {
    "clt": "CLT",
    "cltCnh": "CLT + CNH",
    "civilService": "Concurso Público",
    "againstProof": "Contra Prova",
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

    /* Desabilitar efeito de dimming durante carregamento */
    div[data-testid="stAppViewBlockContainer"] {
        opacity: 1 !important;
    }
    .stSpinner {
        background: transparent !important;
    }
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    /* Manter conteúdo visível durante rerun */
    .stApp > div {
        opacity: 1 !important;
    }

    /* Spinner central de carregamento */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(14, 17, 23, 0.95);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-top: 4px solid #0066CC;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    .loading-text {
        margin-top: 1.5rem;
        color: #FAFAFA;
        font-size: 1.1rem;
        font-weight: 500;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    /* Esconder conteúdo durante carregamento */
    .content-hidden {
        visibility: hidden;
        opacity: 0;
        height: 0;
        overflow: hidden;
    }
    .content-visible {
        visibility: visible;
        opacity: 1;
        transition: opacity 0.3s ease-in;
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

    # Sidebar - Navegação
    with st.sidebar:
        st.markdown("---")

        if "pagina_atual" not in st.session_state:
            st.session_state.pagina_atual = "🏠 Visão Geral"

        paginas = [
            "🏠 Visão Geral",
            "🗺️ Mapa Geográfico",
            "📈 Análise Temporal",
            "📋 Tabela Detalhada",
            "🔍 Auditoria",
            "🏢 Rede",
            "👤 Perfil Demográfico"
        ]

        for pag in paginas:
            tipo_botao = "primary" if st.session_state.pagina_atual == pag else "secondary"
            if st.button(pag, key=f"nav_{pag}", use_container_width=True, type=tipo_botao):
                st.session_state.pagina_atual = pag
                st.rerun()

        pagina = st.session_state.pagina_atual

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

    # Mapeamento de páginas
    paginas_render = {
        "🏠 Visão Geral": ("Visão Geral", render_visao_geral),
        "👤 Perfil Demográfico": ("Perfil Demográfico", render_perfil_demografico),
        "🗺️ Mapa Geográfico": ("Mapa Geográfico", render_mapa),
        "📈 Análise Temporal": ("Análise Temporal", render_temporal),
        "📋 Tabela Detalhada": ("Tabela Detalhada", render_tabela_detalhada),
        "🔍 Auditoria": ("Auditoria", render_auditoria),
        "🏢 Rede": ("Rede", render_rede),
    }

    # Renderizar página com carregamento unificado (dados + página em um único spinner)
    if pagina in paginas_render:
        page_name, render_func = paginas_render[pagina]
        render_page_with_unified_loading(page_name, render_func)


# ============================================
# PÁGINAS
# ============================================

def get_substance_data() -> pd.DataFrame:
    """
    Busca dados de substâncias - amostras do período selecionado
    Usa aggregation pipeline do MongoDB para fazer $lookup e unwind
    Retorna uma linha por amostra, com uma coluna para cada substância
    """
    # Obter período selecionado
    start_date, end_date = get_selected_period()

    # Verificar cache de sessão (incluindo período na chave)
    cache_key = generate_cache_key("substance_data", start_date, end_date)
    cached = get_cached_data("substance_data", cache_key)
    if cached is not None:
        return cached

    try:
        # 1. Buscar lotes do período para criar mapeamento de tipo de análise
        lots_collection = get_collection("lots")
        lots_period = list(lots_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"code": 1, "analysisType": 1, "createdAt": 1, "_samples": 1}
        ))

        if not lots_period:
            return pd.DataFrame()

        # Criar mapeamentos de lote
        lot_codes = []
        lot_type_map = {}
        lot_date_map = {}
        lot_samples_map = {}  # Mapeamento lote -> lista de sample_ids (chainsOfCustody._id)
        analysis_type_names = {
            "screening": "Triagem",
            "confirmatory": "Confirmatório",
            "confirmatoryTHC": "Confirmatório THC"
        }

        all_sample_ids = []  # Todos os _samples de todos os lotes
        for lot in lots_period:
            code = lot.get('code')
            if code:
                lot_codes.append(code)
                analysis_type = lot.get('analysisType', '')
                lot_type_map[code] = analysis_type_names.get(analysis_type, analysis_type or 'N/A')
                created_at = lot.get('createdAt')
                if created_at:
                    # Converter UTC para UTC-3 (horário de Brasília)
                    created_at_brt = created_at - timedelta(hours=3)
                    lot_date_map[code] = created_at_brt.strftime('%d/%m/%Y')
                else:
                    lot_date_map[code] = 'N/A'
                # Guardar _samples (são os IDs das chainsOfCustody)
                samples = lot.get('_samples', [])
                if samples:
                    lot_samples_map[code] = samples
                    all_sample_ids.extend(samples)

        # 1.1 Buscar gatherings para mapear chainOfCustody -> purpose.type e purpose.subType
        # Caminho: lots._samples = chainsOfCustody._id -> gatherings._chainOfCustody
        client = get_mongo_client()
        db = client["ctox"]
        gatherings_collection = db["gatherings"]

        # Buscar todos os gatherings que referenciam as chainsOfCustody dos lotes
        unique_sample_ids = list(set(all_sample_ids))
        gatherings = list(gatherings_collection.find(
            {"_chainOfCustody": {"$in": unique_sample_ids}},
            {"_chainOfCustody": 1, "purpose.type": 1, "purpose.subType": 1}
        ))

        # Mapeamento chainOfCustody._id -> type e subType
        chain_to_type = {}
        chain_to_subtype = {}
        for g in gatherings:
            chain_id = g.get("_chainOfCustody")
            purpose = g.get("purpose", {})
            ptype = purpose.get("type", "") if purpose else ""
            subtype = purpose.get("subType", "") if purpose else ""
            if chain_id:
                chain_to_type[chain_id] = ptype
                chain_to_subtype[chain_id] = subtype

        # Mapeamentos para tradução
        tipos_map = {
            "cnh": "CNH",
            "clt": "CLT",
            "cltCnh": "CLT + CNH"
        }

        subtipos_map = {
            "periodic": "Periódico",
            "hiring": "Admissional",
            "resignation": "Demissional",
            "firstLicense": "Primeira Habilitação",
            "firstCnh": "Primeira Habilitação",
            "renovation": "Renovação",
            "categoryChange": "Mudança de Categoria",
            "functionChange": "Mudança de Função",
            "return": "Retorno ao Trabalho"
        }

        # Mapeamento lote -> tipo e subtipo (traduzidos)
        # Como um lote pode ter várias amostras, pegamos o tipo/subtipo da primeira amostra que tiver dados
        lot_tipo_map = {}
        lot_subtipo_map = {}
        for lot_code, sample_ids in lot_samples_map.items():
            ptype = ""
            subtype = ""
            # Procurar o primeiro sample_id que tenha tipo/subtipo definido
            for sample_id in sample_ids:
                if sample_id in chain_to_type and chain_to_type[sample_id]:
                    ptype = chain_to_type[sample_id]
                    subtype = chain_to_subtype.get(sample_id, "")
                    break
            lot_tipo_map[lot_code] = tipos_map.get(ptype, ptype) if ptype else "N/A"
            lot_subtipo_map[lot_code] = subtipos_map.get(subtype, subtype) if subtype else "N/A"

        # 2. Aggregation pipeline no MongoDB
        client = get_mongo_client()
        db = client["ctox"]
        results_collection = db["results"]

        pipeline = [
            # Filtrar pelos lotes do período
            {"$match": {"_lot": {"$in": lot_codes}}},

            # Desaninhar as amostras - cada sample vira um documento separado
            {"$unwind": "$samples"},

            # Desaninhar os compounds de cada amostra
            {"$unwind": "$samples._compound"},

            # Lookup para pegar o nome do compound da collection compounds
            {
                "$lookup": {
                    "from": "compounds",
                    "localField": "samples._compound._id",
                    "foreignField": "_id",
                    "as": "compoundInfo"
                }
            },

            # Pegar só o primeiro resultado do lookup
            {"$unwind": "$compoundInfo"},

            # Projetar os campos que interessam
            {
                "$project": {
                    "_lot": 1,
                    "_sample": "$samples._sample",
                    "samplePositive": "$samples.positive",
                    "compoundName": "$compoundInfo.name",
                    "compoundPositive": "$samples._compound.positive",
                    "concentration": "$samples._compound.concentration"
                }
            }
        ]

        # Executar aggregation
        results = list(results_collection.aggregate(pipeline, allowDiskUse=True))

        if not results:
            return pd.DataFrame()

        # 3. Converter para DataFrame no formato "long"
        df_long = pd.DataFrame(results)

        # 4. Pivotar para ter cada compound como coluna
        # Primeiro, criar tabela base com metadados únicos por amostra
        df_meta = df_long[['_lot', '_sample', 'samplePositive']].drop_duplicates()

        # Pivotar os compounds
        # Usar 'max' como aggfunc para que se houver qualquer True, o resultado seja True
        # (True > False na comparação, então max(True, False) = True)
        df_pivot = df_long.pivot_table(
            index='_sample',
            columns='compoundName',
            values='compoundPositive',
            aggfunc='max'  # Se houver qualquer positivo, considera positivo
        ).reset_index()

        # Converter True/False para "Positivo"/"Negativo"
        compound_cols = [col for col in df_pivot.columns if col != '_sample']
        for col in compound_cols:
            df_pivot[col] = df_pivot[col].apply(lambda x: 'Positivo' if x == True else 'Negativo')

        # 5. Juntar metadados com dados pivotados
        df_final = df_meta.merge(df_pivot, on='_sample', how='left')

        # 6. Adicionar colunas de Data, Tipo de Lote, Tipo Finalidade e Subfinalidade
        df_final['Data'] = df_final['_lot'].map(lot_date_map).fillna('N/A')
        df_final['Tipo de Lote'] = df_final['_lot'].map(lot_type_map).fillna('N/A')
        df_final['Tipo Exame'] = df_final['_lot'].map(lot_tipo_map).fillna('N/A')
        df_final['Subfinalidade'] = df_final['_lot'].map(lot_subtipo_map).fillna('N/A')

        # 7. Renomear e reorganizar colunas
        df_final = df_final.rename(columns={'_lot': 'Lote', '_sample': 'Amostra'})

        # Reordenar: Data, Lote, Tipo de Lote, Tipo Exame, Subfinalidade, Amostra, depois as substâncias
        first_cols = ['Data', 'Lote', 'Tipo de Lote', 'Tipo Exame', 'Subfinalidade', 'Amostra']
        other_cols = [col for col in df_final.columns if col not in first_cols and col != 'samplePositive']
        df_final = df_final[first_cols + sorted(other_cols)]

        # Salvar no cache de sessão
        set_cached_data("substance_data", cache_key, df_final)

        return df_final

    except Exception as e:
        st.error(f"Erro ao buscar dados de substâncias: {e}")
        return pd.DataFrame()


def get_substance_data_full(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Busca dados de substâncias para um período específico (sem usar sessão).
    Usado para exportações com datas customizadas.
    """
    try:
        # 1. Buscar lotes do período para criar mapeamento de tipo de análise
        lots_collection = get_collection("lots")
        lots_period = list(lots_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"code": 1, "analysisType": 1, "createdAt": 1, "_samples": 1}
        ))

        if not lots_period:
            return pd.DataFrame()

        # Criar mapeamentos de lote
        lot_codes = []
        lot_type_map = {}
        lot_date_map = {}
        lot_samples_map = {}
        analysis_type_names = {
            "screening": "Triagem",
            "confirmatory": "Confirmatório",
            "confirmatoryTHC": "Confirmatório THC"
        }

        all_sample_ids = []
        for lot in lots_period:
            code = lot.get('code')
            if code:
                lot_codes.append(code)
                analysis_type = lot.get('analysisType', '')
                lot_type_map[code] = analysis_type_names.get(analysis_type, analysis_type or 'N/A')
                created_at = lot.get('createdAt')
                if created_at:
                    created_at_brt = created_at - timedelta(hours=3)
                    lot_date_map[code] = created_at_brt.strftime('%d/%m/%Y')
                else:
                    lot_date_map[code] = 'N/A'
                samples = lot.get('_samples', [])
                if samples:
                    lot_samples_map[code] = samples
                    all_sample_ids.extend(samples)

        # 1.1 Buscar gatherings para mapear chainOfCustody -> purpose.type e purpose.subType
        client = get_mongo_client()
        db = client["ctox"]
        gatherings_collection = db["gatherings"]

        unique_sample_ids = list(set(all_sample_ids))
        gatherings = list(gatherings_collection.find(
            {"_chainOfCustody": {"$in": unique_sample_ids}},
            {"_chainOfCustody": 1, "purpose.type": 1, "purpose.subType": 1}
        ))

        chain_to_type = {}
        chain_to_subtype = {}
        for g in gatherings:
            chain_id = g.get("_chainOfCustody")
            purpose = g.get("purpose", {})
            ptype = purpose.get("type", "") if purpose else ""
            subtype = purpose.get("subType", "") if purpose else ""
            if chain_id:
                chain_to_type[chain_id] = ptype
                chain_to_subtype[chain_id] = subtype

        tipos_map = {
            "cnh": "CNH",
            "clt": "CLT",
            "cltCnh": "CLT + CNH"
        }

        subtipos_map = {
            "periodic": "Periódico",
            "hiring": "Admissional",
            "resignation": "Demissional",
            "firstLicense": "Primeira Habilitação",
            "firstCnh": "Primeira Habilitação",
            "renovation": "Renovação",
            "categoryChange": "Mudança de Categoria",
            "functionChange": "Mudança de Função",
            "return": "Retorno ao Trabalho"
        }

        lot_tipo_map = {}
        lot_subtipo_map = {}
        for lot_code, sample_ids in lot_samples_map.items():
            ptype = ""
            subtype = ""
            for sample_id in sample_ids:
                if sample_id in chain_to_type and chain_to_type[sample_id]:
                    ptype = chain_to_type[sample_id]
                    subtype = chain_to_subtype.get(sample_id, "")
                    break
            lot_tipo_map[lot_code] = tipos_map.get(ptype, ptype) if ptype else "N/A"
            lot_subtipo_map[lot_code] = subtipos_map.get(subtype, subtype) if subtype else "N/A"

        # 2. Aggregation pipeline no MongoDB
        results_collection = db["results"]

        pipeline = [
            {"$match": {"_lot": {"$in": lot_codes}}},
            {"$unwind": "$samples"},
            {"$unwind": "$samples._compound"},
            {
                "$lookup": {
                    "from": "compounds",
                    "localField": "samples._compound._id",
                    "foreignField": "_id",
                    "as": "compoundInfo"
                }
            },
            {"$unwind": "$compoundInfo"},
            {
                "$project": {
                    "_lot": 1,
                    "_sample": "$samples._sample",
                    "samplePositive": "$samples.positive",
                    "compoundName": "$compoundInfo.name",
                    "compoundPositive": "$samples._compound.positive",
                    "concentration": "$samples._compound.concentration"
                }
            }
        ]

        results = list(results_collection.aggregate(pipeline, allowDiskUse=True))

        if not results:
            return pd.DataFrame()

        df_long = pd.DataFrame(results)

        df_meta = df_long[['_lot', '_sample', 'samplePositive']].drop_duplicates()

        df_pivot = df_long.pivot_table(
            index='_sample',
            columns='compoundName',
            values='compoundPositive',
            aggfunc='max'
        ).reset_index()

        compound_cols = [col for col in df_pivot.columns if col != '_sample']
        for col in compound_cols:
            df_pivot[col] = df_pivot[col].apply(lambda x: 'Positivo' if x == True else 'Negativo')

        df_final = df_meta.merge(df_pivot, on='_sample', how='left')

        df_final['Data'] = df_final['_lot'].map(lot_date_map).fillna('N/A')
        df_final['Tipo de Lote'] = df_final['_lot'].map(lot_type_map).fillna('N/A')
        df_final['Tipo Exame'] = df_final['_lot'].map(lot_tipo_map).fillna('N/A')
        df_final['Subfinalidade'] = df_final['_lot'].map(lot_subtipo_map).fillna('N/A')

        df_final = df_final.rename(columns={'_lot': 'Lote', '_sample': 'Amostra'})

        first_cols = ['Data', 'Lote', 'Tipo de Lote', 'Tipo Exame', 'Subfinalidade', 'Amostra']
        other_cols = [col for col in df_final.columns if col not in first_cols and col != 'samplePositive']
        df_final = df_final[first_cols + sorted(other_cols)]

        return df_final

    except Exception as e:
        st.error(f"Erro ao buscar dados de substâncias (período completo): {e}")
        return pd.DataFrame()


def get_substance_data_paginated(page: int = 1, page_size: int = 100, filters: dict = None) -> tuple:
    """
    Busca dados de substâncias com paginação server-side.

    Args:
        page: Número da página (começa em 1)
        page_size: Quantidade de registros por página
        filters: Dicionário com filtros opcionais {
            'tipo_lote': str,
            'tipo_exame': str,
            'subfinalidade': str,
            'resultado': str  # 'Positivo' ou 'Negativo'
        }

    Returns:
        Tuple (df: DataFrame, total_count: int, total_pages: int)
    """
    start_date, end_date = get_selected_period()

    try:
        # 1. Buscar lotes do período para criar mapeamento de tipo de análise
        lots_collection = get_collection("lots")
        lots_period = list(lots_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"code": 1, "analysisType": 1, "createdAt": 1, "_samples": 1}
        ))

        if not lots_period:
            return pd.DataFrame(), 0, 0

        # Criar mapeamentos de lote
        lot_codes = []
        lot_type_map = {}
        lot_date_map = {}
        lot_samples_map = {}
        analysis_type_names = {
            "screening": "Triagem",
            "confirmatory": "Confirmatório",
            "confirmatoryTHC": "Confirmatório THC"
        }

        all_sample_ids = []
        for lot in lots_period:
            code = lot.get('code')
            if code:
                lot_codes.append(code)
                analysis_type = lot.get('analysisType', '')
                lot_type_map[code] = analysis_type_names.get(analysis_type, analysis_type or 'N/A')
                created_at = lot.get('createdAt')
                if created_at:
                    created_at_brt = created_at - timedelta(hours=3)
                    lot_date_map[code] = created_at_brt.strftime('%d/%m/%Y')
                else:
                    lot_date_map[code] = 'N/A'
                samples = lot.get('_samples', [])
                if samples:
                    lot_samples_map[code] = samples
                    all_sample_ids.extend(samples)

        # 1.1 Buscar gatherings para mapear chainOfCustody -> purpose.type e purpose.subType
        client = get_mongo_client()
        db = client["ctox"]
        gatherings_collection = db["gatherings"]

        unique_sample_ids = list(set(all_sample_ids))
        gatherings = list(gatherings_collection.find(
            {"_chainOfCustody": {"$in": unique_sample_ids}},
            {"_chainOfCustody": 1, "purpose.type": 1, "purpose.subType": 1}
        ))

        chain_to_type = {}
        chain_to_subtype = {}
        for g in gatherings:
            chain_id = g.get("_chainOfCustody")
            purpose = g.get("purpose", {})
            ptype = purpose.get("type", "") if purpose else ""
            subtype = purpose.get("subType", "") if purpose else ""
            if chain_id:
                chain_to_type[chain_id] = ptype
                chain_to_subtype[chain_id] = subtype

        # Mapeamentos para tradução
        tipos_map = {
            "cnh": "CNH",
            "clt": "CLT",
            "cltCnh": "CLT + CNH"
        }

        subtipos_map = {
            "periodic": "Periódico",
            "hiring": "Admissional",
            "resignation": "Demissional",
            "firstLicense": "Primeira Habilitação",
            "firstCnh": "Primeira Habilitação",
            "renovation": "Renovação",
            "categoryChange": "Mudança de Categoria",
            "functionChange": "Mudança de Função",
            "return": "Retorno ao Trabalho",
            "againstProof": "Contra Prova"
        }

        lot_tipo_map = {}
        lot_subtipo_map = {}
        for lot_code, sample_ids in lot_samples_map.items():
            ptype = ""
            subtype = ""
            for sample_id in sample_ids:
                if sample_id in chain_to_type and chain_to_type[sample_id]:
                    ptype = chain_to_type[sample_id]
                    subtype = chain_to_subtype.get(sample_id, "")
                    break
            lot_tipo_map[lot_code] = tipos_map.get(ptype, ptype) if ptype else "N/A"
            lot_subtipo_map[lot_code] = subtipos_map.get(subtype, subtype) if subtype else "N/A"

        # Aplicar filtros de lote se necessário
        filtered_lot_codes = lot_codes
        if filters:
            if filters.get('tipo_lote') and filters['tipo_lote'] != 'Todos':
                filtered_lot_codes = [c for c in filtered_lot_codes if lot_type_map.get(c) == filters['tipo_lote']]
            if filters.get('tipo_exame') and filters['tipo_exame'] != 'Todos':
                filtered_lot_codes = [c for c in filtered_lot_codes if lot_tipo_map.get(c) == filters['tipo_exame']]
            if filters.get('subfinalidade') and filters['subfinalidade'] != 'Todos':
                filtered_lot_codes = [c for c in filtered_lot_codes if lot_subtipo_map.get(c) == filters['subfinalidade']]

        if not filtered_lot_codes:
            return pd.DataFrame(), 0, 0

        # 2. Pipeline com contagem total
        results_collection = db["results"]

        # Pipeline base para contagem
        match_stage = {"$match": {"_lot": {"$in": filtered_lot_codes}}}

        # Contar total de amostras únicas
        count_pipeline = [
            match_stage,
            {"$unwind": "$samples"},
            {"$group": {"_id": "$samples._sample"}},
            {"$count": "total"}
        ]

        count_result = list(results_collection.aggregate(count_pipeline, allowDiskUse=True))
        total_count = count_result[0]['total'] if count_result else 0
        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0

        if total_count == 0:
            return pd.DataFrame(), 0, 0

        # Pipeline com paginação
        skip = (page - 1) * page_size

        pipeline = [
            match_stage,
            {"$unwind": "$samples"},
            {"$group": {
                "_id": "$samples._sample",
                "_lot": {"$first": "$_lot"},
                "samplePositive": {"$first": "$samples.positive"},
                "compounds": {"$first": "$samples._compound"}
            }},
            {"$sort": {"_lot": 1, "_id": 1}},
            {"$skip": skip},
            {"$limit": page_size},
            {"$unwind": "$compounds"},
            {
                "$lookup": {
                    "from": "compounds",
                    "localField": "compounds._id",
                    "foreignField": "_id",
                    "as": "compoundInfo"
                }
            },
            {"$unwind": "$compoundInfo"},
            {
                "$project": {
                    "_lot": 1,
                    "_sample": "$_id",
                    "samplePositive": 1,
                    "compoundName": "$compoundInfo.name",
                    "compoundPositive": "$compounds.positive",
                    "concentration": "$compounds.concentration"
                }
            }
        ]

        results = list(results_collection.aggregate(pipeline, allowDiskUse=True))

        if not results:
            return pd.DataFrame(), total_count, total_pages

        # 3. Converter para DataFrame no formato "long"
        df_long = pd.DataFrame(results)

        # 4. Pivotar para ter cada compound como coluna
        df_meta = df_long[['_lot', '_sample', 'samplePositive']].drop_duplicates()

        df_pivot = df_long.pivot_table(
            index='_sample',
            columns='compoundName',
            values='compoundPositive',
            aggfunc='max'
        ).reset_index()

        # Converter True/False para "Positivo"/"Negativo"
        compound_cols = [col for col in df_pivot.columns if col != '_sample']
        for col in compound_cols:
            df_pivot[col] = df_pivot[col].apply(lambda x: 'Positivo' if x == True else 'Negativo')

        # 5. Juntar metadados com dados pivotados
        df_final = df_meta.merge(df_pivot, on='_sample', how='left')

        # 6. Adicionar colunas de Data, Tipo de Lote, Tipo Exame e Subfinalidade
        df_final['Data'] = df_final['_lot'].map(lot_date_map).fillna('N/A')
        df_final['Tipo de Lote'] = df_final['_lot'].map(lot_type_map).fillna('N/A')
        df_final['Tipo Exame'] = df_final['_lot'].map(lot_tipo_map).fillna('N/A')
        df_final['Subfinalidade'] = df_final['_lot'].map(lot_subtipo_map).fillna('N/A')

        # 7. Renomear e reorganizar colunas
        df_final = df_final.rename(columns={'_lot': 'Lote', '_sample': 'Amostra'})

        first_cols = ['Data', 'Lote', 'Tipo de Lote', 'Tipo Exame', 'Subfinalidade', 'Amostra']
        other_cols = [col for col in df_final.columns if col not in first_cols and col != 'samplePositive']
        df_final = df_final[first_cols + sorted(other_cols)]

        # Aplicar filtro de resultado se necessário
        if filters and filters.get('resultado') and filters['resultado'] != 'Todos':
            # Verificar se alguma substância é positiva
            substance_cols = [c for c in df_final.columns if c not in first_cols]
            if filters['resultado'] == 'Positivo':
                mask = df_final[substance_cols].apply(lambda row: any(v == 'Positivo' for v in row), axis=1)
            else:
                mask = df_final[substance_cols].apply(lambda row: all(v == 'Negativo' for v in row), axis=1)
            df_final = df_final[mask]

        return df_final, total_count, total_pages

    except Exception as e:
        st.error(f"Erro ao buscar dados paginados: {e}")
        return pd.DataFrame(), 0, 0


def get_sample_concentration_data(sample_code: str) -> dict:
    """
    Busca os dados de concentração de uma amostra específica.
    Retorna dict {compound_name: {"concentration": float, "positive": bool}}

    Estrutura do MongoDB (collection results):
    - samples: array de amostras
    - samples._sample: código da amostra (string ou número)
    - samples._compound: array de compostos
    - samples._compound._id: ObjectId referenciando collection compounds
    - samples._compound.positive: bool indicando se o composto é positivo
    - samples._compound.concentration: float com a concentração
    """
    try:
        client = get_mongo_client()
        db = client["ctox"]
        results_collection = db["results"]

        # O _sample pode ser string ou número no MongoDB, tentar ambos
        sample_code_clean = str(sample_code).strip()
        sample_match_values = [sample_code_clean]
        try:
            # Tentar como número inteiro
            sample_match_values.append(int(sample_code_clean))
        except:
            pass

        # Buscar o resultado que contém essa amostra
        pipeline = [
            {"$unwind": "$samples"},
            {"$match": {"samples._sample": {"$in": sample_match_values}}},
            {"$unwind": "$samples._compound"},
            {
                "$lookup": {
                    "from": "compounds",
                    "localField": "samples._compound._id",
                    "foreignField": "_id",
                    "as": "compoundInfo"
                }
            },
            {"$unwind": {"path": "$compoundInfo", "preserveNullAndEmptyArrays": True}},
            {
                "$project": {
                    "compoundName": "$compoundInfo.name",
                    "concentration": "$samples._compound.concentration",
                    "positive": "$samples._compound.positive",
                    "lsqOverflow": "$samples._compound.lsqOverflow"
                }
            }
        ]

        results = list(results_collection.aggregate(pipeline))

        if not results:
            return {}

        concentration_data = {}
        for r in results:
            compound_name = r.get("compoundName", "")
            if compound_name:
                concentration_data[compound_name] = {
                    "concentration": r.get("concentration", 0),
                    "positive": r.get("positive", False),
                    "lsqOverflow": r.get("lsqOverflow", False)
                }

        return concentration_data

    except Exception as e:
        st.error(f"Erro ao buscar concentrações da amostra: {e}")
        return {}


def get_average_concentrations() -> dict:
    """
    Busca a média de concentrações por substância de todas as amostras POSITIVAS do período.
    Retorna dict {compound_name: {"avg_concentration": float, "count": int}}
    """
    start_date, end_date = get_selected_period()

    cache_key = generate_cache_key("avg_concentrations", start_date, end_date)
    cached = get_cached_data("avg_concentrations", cache_key)
    if cached is not None:
        return cached

    try:
        client = get_mongo_client()
        db = client["ctox"]
        results_collection = db["results"]
        lots_collection = db["lots"]

        # Buscar lotes do período
        lots_period = list(lots_collection.find(
            {"createdAt": {"$gte": start_date, "$lte": end_date}},
            {"code": 1}
        ))
        lot_codes = [lot.get("code") for lot in lots_period if lot.get("code")]

        if not lot_codes:
            return {}

        # Pipeline para calcular média de concentração por substância (apenas positivos)
        pipeline = [
            {"$match": {"_lot": {"$in": lot_codes}}},
            {"$unwind": "$samples"},
            {"$unwind": "$samples._compound"},
            {"$match": {"samples._compound.positive": True}},  # Apenas positivos
            {
                "$lookup": {
                    "from": "compounds",
                    "localField": "samples._compound._id",
                    "foreignField": "_id",
                    "as": "compoundInfo"
                }
            },
            {"$unwind": "$compoundInfo"},
            {
                "$group": {
                    "_id": "$compoundInfo.name",
                    "avg_concentration": {"$avg": "$samples._compound.concentration"},
                    "max_concentration": {"$max": "$samples._compound.concentration"},
                    "min_concentration": {"$min": "$samples._compound.concentration"},
                    "count": {"$sum": 1}
                }
            }
        ]

        results = list(results_collection.aggregate(pipeline, allowDiskUse=True))

        avg_data = {}
        for r in results:
            compound_name = r.get("_id", "")
            if compound_name:
                avg_data[compound_name] = {
                    "avg_concentration": r.get("avg_concentration", 0),
                    "max_concentration": r.get("max_concentration", 0),
                    "min_concentration": r.get("min_concentration", 0),
                    "count": r.get("count", 0)
                }

        set_cached_data("avg_concentrations", cache_key, avg_data)
        return avg_data

    except Exception as e:
        st.error(f"Erro ao buscar médias de concentração: {e}")
        return {}


def get_sample_laboratory(sample_code: str) -> dict:
    """
    Busca o laboratório vinculado a uma amostra.
    Caminho: sample_code -> chainofcustodies (sample.code) -> gatherings (_chainOfCustody) -> laboratory
    Retorna dict com lab_id, lab_name, lab_city, lab_state
    """
    try:
        client = get_mongo_client()
        db = client["ctox"]

        # O sample.code pode ser string ou número
        sample_code_clean = str(sample_code).strip()
        sample_match_values = [sample_code_clean]
        try:
            sample_match_values.append(int(sample_code_clean))
        except:
            pass

        # 1. Buscar o chainOfCustody pelo sample.code
        chains_collection = db["chainofcustodies"]
        chain = chains_collection.find_one(
            {"sample.code": {"$in": sample_match_values}},
            {"_id": 1}
        )

        if not chain:
            return {}

        chain_id = chain.get("_id")
        if not chain_id:
            return {}

        # 2. Buscar o gathering para pegar o _laboratory
        gatherings_collection = db["gatherings"]
        gathering = gatherings_collection.find_one(
            {"_chainOfCustody": chain_id},
            {"_laboratory": 1}
        )

        if not gathering:
            return {}

        lab_id = gathering.get("_laboratory")
        if not lab_id:
            return {}

        # 3. Buscar informações do laboratório
        labs_map = get_laboratories_map()
        lab_id_str = str(lab_id)

        if lab_id_str in labs_map:
            lab_name = labs_map[lab_id_str]
        else:
            lab_name = "Laboratório"

        # Buscar mais detalhes do laboratório
        labs_with_address = get_laboratories_with_address()
        lab_details = next((lab for lab in labs_with_address if lab.get("id") == lab_id_str), {})

        return {
            "lab_id": lab_id_str,
            "lab_name": lab_name,
            "lab_city": lab_details.get("city", ""),
            "lab_state": lab_details.get("state", "")
        }

    except Exception as e:
        return {}


def get_average_concentrations_by_lab(laboratory_id: str) -> dict:
    """
    Busca a média de concentrações por substância de amostras POSITIVAS de um laboratório específico.
    Retorna dict {compound_name: {"avg_concentration": float, "count": int}}
    """
    if not laboratory_id:
        return {}

    start_date, end_date = get_selected_period()

    cache_key = generate_cache_key("avg_concentrations_lab", laboratory_id, start_date, end_date)
    cached = get_cached_data("avg_concentrations_lab", cache_key)
    if cached is not None:
        return cached

    try:
        client = get_mongo_client()
        db = client["ctox"]
        results_collection = db["results"]
        lots_collection = db["lots"]
        gatherings_collection = db["gatherings"]
        chains_collection = db["chainofcustodies"]

        # Converter laboratory_id para ObjectId
        try:
            lab_oid = ObjectId(laboratory_id)
        except Exception:
            return {}

        # 1. Buscar gatherings do laboratório para pegar os chainOfCustody IDs
        lab_gatherings = list(gatherings_collection.find(
            {
                "_laboratory": lab_oid,
                "createdAt": {"$gte": start_date, "$lte": end_date}
            },
            {"_chainOfCustody": 1}
        ))
        chain_ids = [g.get("_chainOfCustody") for g in lab_gatherings if g.get("_chainOfCustody")]

        if not chain_ids:
            # Tentar buscar também por string do laboratory_id
            lab_gatherings = list(gatherings_collection.find(
                {
                    "_laboratory": laboratory_id,
                    "createdAt": {"$gte": start_date, "$lte": end_date}
                },
                {"_chainOfCustody": 1}
            ))
            chain_ids = [g.get("_chainOfCustody") for g in lab_gatherings if g.get("_chainOfCustody")]

        if not chain_ids:
            return {}

        # 2. Buscar os sample codes das chains do laboratório
        chains = list(chains_collection.find(
            {"_id": {"$in": chain_ids}},
            {"sample.code": 1}
        ))

        # Criar lista de sample_codes com ambos os tipos (string e int) para match correto
        sample_codes = []
        for c in chains:
            code = c.get("sample", {}).get("code")
            if code is not None:
                sample_codes.append(code)
                # Adicionar também a versão string/int para garantir o match
                if isinstance(code, int):
                    sample_codes.append(str(code))
                elif isinstance(code, str):
                    try:
                        sample_codes.append(int(code))
                    except:
                        pass

        if not sample_codes:
            return {}

        # 3. Buscar lotes do período que contêm essas chains
        lots_period = list(lots_collection.find(
            {
                "createdAt": {"$gte": start_date, "$lte": end_date},
                "_samples": {"$in": chain_ids}
            },
            {"code": 1}
        ))
        lot_codes = [lot.get("code") for lot in lots_period if lot.get("code")]

        if not lot_codes:
            return {}

        # 4. Pipeline para calcular média de concentração por substância (apenas positivos do lab)
        pipeline = [
            {"$match": {"_lot": {"$in": lot_codes}}},
            {"$unwind": "$samples"},
            {"$match": {"samples._sample": {"$in": sample_codes}}},  # Filtrar apenas amostras do lab
            {"$unwind": "$samples._compound"},
            {"$match": {"samples._compound.positive": True}},
            {
                "$lookup": {
                    "from": "compounds",
                    "localField": "samples._compound._id",
                    "foreignField": "_id",
                    "as": "compoundInfo"
                }
            },
            {"$unwind": "$compoundInfo"},
            {
                "$group": {
                    "_id": "$compoundInfo.name",
                    "avg_concentration": {"$avg": "$samples._compound.concentration"},
                    "max_concentration": {"$max": "$samples._compound.concentration"},
                    "min_concentration": {"$min": "$samples._compound.concentration"},
                    "count": {"$sum": 1}
                }
            }
        ]

        results = list(results_collection.aggregate(pipeline, allowDiskUse=True))

        avg_data = {}
        for r in results:
            compound_name = r.get("_id", "")
            if compound_name:
                avg_data[compound_name] = {
                    "avg_concentration": r.get("avg_concentration", 0),
                    "max_concentration": r.get("max_concentration", 0),
                    "min_concentration": r.get("min_concentration", 0),
                    "count": r.get("count", 0)
                }

        set_cached_data("avg_concentrations_lab", cache_key, avg_data)
        return avg_data

    except Exception as e:
        return {}


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


def get_positivity_by_laboratory(start_date: datetime = None, end_date: datetime = None) -> dict:
    """
    Retorna dados agregados de positividade por laboratório.
    Usado pela página de Auditoria para detectar anomalias.

    Retorna: dict {lab_id: {"total": int, "positivos": int, "taxa": float}}

    Taxa = positivas confirmatório / total triagem * 100
    """
    cache_key = f"positivity_by_lab_{start_date}_{end_date}"
    cached = get_cached_data("positivity_by_lab", cache_key)
    if cached is not None:
        return cached

    try:
        if start_date is None or end_date is None:
            start_date, end_date = get_selected_period()

        # Obter mapeamentos
        gatherings_data = get_all_gatherings(start_date, end_date)
        chain_to_lab = gatherings_data.get("chain_to_lab", {})
        chain_to_code = get_chain_to_sample_map(start_date, end_date)

        # Mapear sample_code para lab_id
        sample_to_lab = {}
        for chain_id, lab_id in chain_to_lab.items():
            sample_code = chain_to_code.get(chain_id)
            if sample_code:
                sample_to_lab[sample_code] = lab_id

        # Obter lotes e resultados
        lots_dict = get_all_lots(start_date, end_date)
        results_dict = get_all_results(start_date, end_date)

        # Dicionário para agregar por laboratório
        lab_data = {}

        # Contar triagem (screening) por laboratório
        for lot_code, lot_data in lots_dict.items():
            if lot_data.get("analysisType") != "screening":
                continue

            lot_samples = lot_data.get("_samples", set())
            lot_results = results_dict.get(lot_code, [])

            for sample in lot_results:
                sample_code = sample.get("_sample")
                lab_id = sample_to_lab.get(sample_code)

                if lab_id:
                    if lab_id not in lab_data:
                        lab_data[lab_id] = {"total": 0, "positivos": 0}
                    lab_data[lab_id]["total"] += 1

        # Contar positivos do confirmatório por laboratório
        for lot_code, lot_data in lots_dict.items():
            if lot_data.get("analysisType") != "confirmatory":
                continue

            lot_results = results_dict.get(lot_code, [])

            for sample in lot_results:
                sample_code = sample.get("_sample")
                lab_id = sample_to_lab.get(sample_code)

                if lab_id and sample.get("positive", False):
                    if lab_id not in lab_data:
                        lab_data[lab_id] = {"total": 0, "positivos": 0}
                    lab_data[lab_id]["positivos"] += 1

        # Calcular taxa para cada laboratório
        for lab_id, data in lab_data.items():
            total = data["total"]
            positivos = data["positivos"]
            data["taxa"] = (positivos / total * 100) if total > 0 else 0

        set_cached_data("positivity_by_lab", cache_key, lab_data)
        return lab_data

    except Exception as e:
        return {}


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
    Busca contagem de amostras por finalidade - USA DADOS PRÉ-CARREGADOS.
    Conta apenas amostras que têm resultados processados (consistente com outras métricas).
    """
    cache_key = generate_cache_key("samples_purpose", laboratory_id, month)
    cached = get_cached_data("samples_purpose_data", cache_key)
    if cached is not None:
        return cached

    try:
        # Usar dados pré-carregados com período selecionado
        start_date, end_date = get_selected_period()
        gatherings_data = get_all_gatherings(start_date, end_date)
        lots_dict = get_all_lots(start_date, end_date)
        results_dict = get_all_results(start_date, end_date)

        chain_to_lab = gatherings_data.get("chain_to_lab", {})
        chain_to_purpose = gatherings_data.get("chain_to_purpose", {})

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
            "againstProof": "Contra Prova",
            "periodic": "Periódico",
            "categoryChange": "Mudança de Categoria",
            "hiring": "Admissão",
            "renovation": "Renovação",
            "resignation": "Demissão"
        }

        # Coletar chain_ids que têm resultados (apenas triagem para não duplicar)
        chains_com_resultados = set()
        for lot_code, lot_data in lots_dict.items():
            # Apenas triagem para não contar a mesma amostra múltiplas vezes
            if lot_data.get("analysisType") != "screening":
                continue

            # Filtrar por mês
            if month and lot_data.get("month") != month:
                continue

            # Pegar as chains deste lote que têm resultados
            lot_samples = lot_data.get("_samples", set())
            lot_results = results_dict.get(lot_code, [])

            if lot_results:
                chains_com_resultados.update(lot_samples)

        purpose_counts = {}

        for chain_id in chains_com_resultados:
            purpose_type = chain_to_purpose.get(chain_id, "")

            # Filtrar por laboratório
            if laboratory_id:
                lab_id = chain_to_lab.get(chain_id)
                if lab_id != laboratory_id:
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


def get_samples_by_subtype(laboratory_id: str = None, month: int = None, purpose_type: str = None) -> dict:
    """
    Busca contagem de amostras por subfinalidade (purpose.subType) - USA DADOS PRÉ-CARREGADOS.
    Conta apenas amostras que têm resultados processados (consistente com outras métricas).
    """
    cache_key = generate_cache_key("samples_subtype", laboratory_id, month, purpose_type)
    cached = get_cached_data("samples_subtype_data", cache_key)
    if cached is not None:
        return cached

    try:
        # Usar dados pré-carregados com período selecionado
        start_date, end_date = get_selected_period()
        gatherings_data = get_all_gatherings(start_date, end_date)
        lots_dict = get_all_lots(start_date, end_date)
        results_dict = get_all_results(start_date, end_date)

        chain_to_lab = gatherings_data.get("chain_to_lab", {})
        chain_to_purpose = gatherings_data.get("chain_to_purpose", {})
        chain_to_subtype = gatherings_data.get("chain_to_subtype", {})

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

        # Mapear tipos de subfinalidade para nomes legíveis
        subtype_names = {
            "periodic": "Periódico",
            "categoryChange": "Mudança de Categoria",
            "hiring": "Admissão",
            "renovation": "Renovação",
            "resignation": "Demissão",
            "returnToWork": "Retorno ao Trabalho",
            "return": "Retorno ao Trabalho",
            "functionChange": "Mudança de Função",
            "firstLicense": "Primeira Habilitação",
            "random": "Aleatório",
            "postAccident": "Pós-Acidente",
            "reasonableCause": "Causa Razoável",
            "followUp": "Acompanhamento",
            "preEmployment": "Pré-Admissão"
        }

        # Coletar chain_ids que têm resultados (apenas triagem para não duplicar)
        chains_com_resultados = set()
        for lot_code, lot_data in lots_dict.items():
            # Apenas triagem para não contar a mesma amostra múltiplas vezes
            if lot_data.get("analysisType") != "screening":
                continue

            # Filtrar por mês
            if month and lot_data.get("month") != month:
                continue

            # Pegar as chains deste lote que têm resultados
            lot_samples = lot_data.get("_samples", set())
            lot_results = results_dict.get(lot_code, [])

            if lot_results:
                chains_com_resultados.update(lot_samples)

        subtype_counts = {}

        for chain_id in chains_com_resultados:
            # Verificar se tem subtype
            subtype = chain_to_subtype.get(chain_id)
            if not subtype:
                subtype = ""

            # Filtrar por laboratório
            if laboratory_id:
                lab_id = chain_to_lab.get(chain_id)
                if lab_id != laboratory_id:
                    continue

            # Filtrar por finalidade (purpose.type)
            if purpose_type:
                chain_purpose = chain_to_purpose.get(chain_id)
                if chain_purpose != purpose_type:
                    continue

            subtype_name = subtype_names.get(subtype, subtype or 'Não informado')

            if subtype_name not in subtype_counts:
                subtype_counts[subtype_name] = 0
            subtype_counts[subtype_name] += 1

        set_cached_data("samples_subtype_data", cache_key, subtype_counts)
        return subtype_counts

    except Exception as e:
        st.error(f"Erro ao buscar dados por subfinalidade: {e}")
        return {}


def render_visao_geral():
    st.title("🏠 Visão Geral")

    # Filtros
    st.markdown("### 🔍 Filtros")

    col_filtro1, col_filtro2, col_filtro3 = st.columns(3)

    # Filtro de Finalidade
    with col_filtro1:
        finalidades = {
            "Todas": None,
            "CLT": "clt",
            "CLT + CNH": "cltCnh",
            "Concurso Público": "civilService",
            "Contra Prova": "againstProof",
        }
        selected_finalidade_name = st.selectbox(
            "Finalidade da Amostra",
            options=list(finalidades.keys()),
            index=0,
            key="visao_geral_finalidade"
        )
        selected_purpose = finalidades[selected_finalidade_name]

    # Filtro de Subfinalidade
    with col_filtro2:
        # Subfinalidades dependem da finalidade selecionada
        subfinalidades_por_finalidade = {
            "Todas": {
                "Todas": None,
                "Periódico": "periodic",
                "Admissão": "hiring",
                "Demissão": "resignation",
                "Mudança de Categoria": "categoryChange",
                "Renovação": "renovation",
                "Retorno ao Trabalho": "returnToWork",
                "Aleatório": "random",
                "Pós-Acidente": "postAccident",
                "Causa Razoável": "reasonableCause",
                "Acompanhamento": "followUp",
                "Pré-Admissão": "preEmployment"
            },
            "CLT": {
                "Todas": None,
                "Periódico": "periodic",
                "Admissão": "hiring",
                "Demissão": "resignation",
                "Mudança de Categoria": "categoryChange",
                "Retorno ao Trabalho": "returnToWork",
                "Aleatório": "random",
                "Pós-Acidente": "postAccident",
                "Causa Razoável": "reasonableCause",
                "Acompanhamento": "followUp"
            },
            "CLT + CNH": {
                "Todas": None,
                "Periódico": "periodic",
                "Admissão": "hiring",
                "Demissão": "resignation",
                "Mudança de Categoria": "categoryChange",
                "Retorno ao Trabalho": "returnToWork",
                "Aleatório": "random",
                "Pós-Acidente": "postAccident",
                "Causa Razoável": "reasonableCause",
                "Acompanhamento": "followUp"
            },
            "Concurso Público": {
                "Todas": None,
                "Pré-Admissão": "preEmployment"
            },
            "Contra Prova": {
                "Todas": None
            }
        }
        subfinalidades_disponiveis = subfinalidades_por_finalidade.get(selected_finalidade_name, {"Todas": None})
        selected_subfinalidade_name = st.selectbox(
            "Subfinalidade",
            options=list(subfinalidades_disponiveis.keys()),
            index=0,
            key="visao_geral_subfinalidade"
        )
        selected_subtype = subfinalidades_disponiveis[selected_subfinalidade_name]

    # Filtro de Laboratório por CNPJ - MÚLTIPLA SELEÇÃO
    with col_filtro3:
        labs_by_cnpj = get_laboratories_by_cnpj()
        cnpj_options = sorted(labs_by_cnpj.keys())

        selected_cnpjs = st.multiselect(
            "CNPJ Laboratórios (PCL)",
            options=cnpj_options,
            default=[],
            placeholder="Todos os laboratórios",
            key="visao_geral_cnpjs"
        )

        # Converter CNPJs selecionados para lista de lab_ids
        if selected_cnpjs:
            selected_lab_ids = [labs_by_cnpj[cnpj]["id"] for cnpj in selected_cnpjs if cnpj in labs_by_cnpj]
            # Mostrar labs selecionados
            lab_names = [labs_by_cnpj[cnpj]["name"] for cnpj in selected_cnpjs if cnpj in labs_by_cnpj]
            if len(lab_names) <= 3:
                st.caption(f"🏢 {', '.join(lab_names)}")
            else:
                st.caption(f"🏢 {len(lab_names)} laboratórios selecionados")
        else:
            selected_lab_ids = None

    st.markdown("---")

    # Verificar se há múltiplos CNPJs selecionados
    multiplos_cnpjs = selected_cnpjs and len(selected_cnpjs) > 1

    if multiplos_cnpjs:
        # ========== MODO COMPARAÇÃO: MÚLTIPLOS CNPJs ==========
        st.markdown("### 📊 Comparação entre Laboratórios")

        # Buscar taxa média nacional para comparação
        periodo_inicio, periodo_fim = get_selected_period()
        taxa_nacional = get_national_average_rate(periodo_inicio, periodo_fim)

        # Carregar dados para cada CNPJ
        dados_por_cnpj = {}
        for cnpj in selected_cnpjs:
            lab_id = labs_by_cnpj[cnpj]["id"]
            lab_name = labs_by_cnpj[cnpj]["name"]

            triagem = get_triagem_data(lab_id, None, selected_purpose)
            confirmatorio = get_confirmatorio_data(lab_id, None, selected_purpose)
            confirmatorio_thc = get_confirmatorio_thc_data(lab_id, None, selected_purpose)

            total_tri = triagem["positivo"] + triagem["negativo"]
            total_conf = confirmatorio["positivo"] + confirmatorio["negativo"]
            total_conf_thc = confirmatorio_thc["positivo"] + confirmatorio_thc["negativo"]
            total = total_tri + total_conf + total_conf_thc

            # Confirmatório combinado
            total_conf_geral = total_conf + total_conf_thc
            pos_conf_geral = confirmatorio["positivo"] + confirmatorio_thc["positivo"]
            neg_conf_geral = confirmatorio["negativo"] + confirmatorio_thc["negativo"]

            # Taxa geral: positivas confirmatórias / total triagem
            taxa_geral = (pos_conf_geral / total_tri * 100) if total_tri > 0 else 0

            # Diferença vs média nacional
            dif_nacional = taxa_geral - taxa_nacional if taxa_nacional > 0 else 0

            dados_por_cnpj[cnpj] = {
                "nome": lab_name,
                "triagem": triagem,
                "confirmatorio": confirmatorio,
                "confirmatorio_thc": confirmatorio_thc,
                "total_triagem": total_tri,
                "total_confirmatorio": total_conf,
                "total_confirmatorio_thc": total_conf_thc,
                "total_confirmatorio_geral": total_conf_geral,
                "pos_confirmatorio_geral": pos_conf_geral,
                "neg_confirmatorio_geral": neg_conf_geral,
                "total": total,
                "taxa_geral": taxa_geral,
                "dif_nacional": dif_nacional
            }

        # Criar DataFrame para comparação
        df_comparacao = pd.DataFrame([
            {
                "CNPJ": cnpj,
                "Laboratório": dados["nome"][:25],
                "Total Amostras": dados["total_triagem"],
                "Taxa Geral (%)": dados["taxa_geral"],
                "vs Nacional": dados["dif_nacional"],
                "Neg. Triagem": dados["triagem"]["negativo"],
                "Pos. Triagem": dados["triagem"]["positivo"],
                "Neg. Confirm.": dados["neg_confirmatorio_geral"],
                "Pos. Confirm.": dados["pos_confirmatorio_geral"]
            }
            for cnpj, dados in dados_por_cnpj.items()
        ])

        # KPIs totais agregados
        total_amostras_geral = sum(d["total_triagem"] for d in dados_por_cnpj.values())
        total_pos_conf_geral = sum(d["pos_confirmatorio_geral"] for d in dados_por_cnpj.values())
        taxa_geral_agregada = (total_pos_conf_geral / total_amostras_geral * 100) if total_amostras_geral > 0 else 0
        dif_nacional_agregada = taxa_geral_agregada - taxa_nacional if taxa_nacional > 0 else 0

        # Totais triagem e confirmatório
        total_neg_triagem = sum(d["triagem"]["negativo"] for d in dados_por_cnpj.values())
        total_pos_triagem = sum(d["triagem"]["positivo"] for d in dados_por_cnpj.values())
        total_neg_conf = sum(d["neg_confirmatorio_geral"] for d in dados_por_cnpj.values())
        total_pos_conf = sum(d["pos_confirmatorio_geral"] for d in dados_por_cnpj.values())
        total_conf_geral = total_neg_conf + total_pos_conf

        # ========== KPIs - Primeira linha ==========
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

        with col_kpi1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 20px; border-radius: 10px; text-align: center;
                        border: 1px solid #FFD700;">
                <p style="color: #888; margin: 0; font-size: 14px;">Taxa Geral de Positividade</p>
                <h2 style="color: {}; margin: 5px 0;">{:.2f}%</h2>
                <p style="color: #888; margin: 0; font-size: 11px;">(Positivas Confirm. / Amostras)</p>
            </div>
            """.format("#FF6B6B" if taxa_geral_agregada > 5 else "#FFD700" if taxa_geral_agregada > 2 else "#00CED1", taxa_geral_agregada), unsafe_allow_html=True)

        with col_kpi2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 20px; border-radius: 10px; text-align: center;
                        border: 1px solid #00CED1;">
                <p style="color: #888; margin: 0; font-size: 14px;">Total de Amostras</p>
                <h2 style="color: #00CED1; margin: 5px 0;">{:,}</h2>
                <p style="color: #888; margin: 0; font-size: 11px;">&nbsp;</p>
            </div>
            """.format(total_amostras_geral).replace(",", "."), unsafe_allow_html=True)

        with col_kpi3:
            diferenca_texto = f"+{dif_nacional_agregada:.2f}%" if dif_nacional_agregada >= 0 else f"{dif_nacional_agregada:.2f}%"
            diferenca_cor = "#FF6B6B" if dif_nacional_agregada > 0 else "#4CAF50"
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 20px; border-radius: 10px; text-align: center;
                        border: 1px solid {};">
                <p style="color: #888; margin: 0; font-size: 14px;">vs Média Nacional</p>
                <h2 style="color: {}; margin: 5px 0;">{}</h2>
                <p style="color: #888; margin: 0; font-size: 11px;">Nacional: {:.2f}%</p>
            </div>
            """.format(diferenca_cor, diferenca_cor, diferenca_texto, taxa_nacional), unsafe_allow_html=True)

        # ========== KPIs - Segunda linha: Triagem e Confirmatório ==========
        st.markdown("")
        col_kpi4, col_kpi5 = st.columns(2)

        with col_kpi4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 15px; border-radius: 10px; text-align: center;
                        border: 1px solid #00CED1;">
                <p style="color: #888; margin: 0; font-size: 14px;">🔬 Triagem (Agregado)</p>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Negativas</p>
                        <h3 style="color: #00CED1; margin: 5px 0;">{:,}</h3>
                    </div>
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Positivas</p>
                        <h3 style="color: #FF6B6B; margin: 5px 0;">{:,}</h3>
                    </div>
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Total</p>
                        <h3 style="color: #E8E8E8; margin: 5px 0;">{:,}</h3>
                    </div>
                </div>
            </div>
            """.format(total_neg_triagem, total_pos_triagem, total_amostras_geral).replace(",", "."), unsafe_allow_html=True)

        with col_kpi5:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 15px; border-radius: 10px; text-align: center;
                        border: 1px solid #9370DB;">
                <p style="color: #888; margin: 0; font-size: 14px;">🧪 Confirmatório (Agregado)</p>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Negativas</p>
                        <h3 style="color: #00CED1; margin: 5px 0;">{:,}</h3>
                    </div>
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Positivas</p>
                        <h3 style="color: #FF6B6B; margin: 5px 0;">{:,}</h3>
                    </div>
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Total</p>
                        <h3 style="color: #E8E8E8; margin: 5px 0;">{:,}</h3>
                    </div>
                </div>
            </div>
            """.format(total_neg_conf, total_pos_conf, total_conf_geral).replace(",", "."), unsafe_allow_html=True)

        st.markdown("---")

        # ========== Gráfico de Taxa Geral por Laboratório ==========
        st.markdown("### 📊 Taxa Geral de Positividade por Laboratório")

        df_taxa = df_comparacao.sort_values("Taxa Geral (%)", ascending=True)

        # Cores baseadas na diferença vs nacional
        cores_barras = ['#FF6B6B' if v > 0 else '#4CAF50' for v in df_taxa["vs Nacional"]]

        fig_taxa = go.Figure()
        fig_taxa.add_trace(go.Bar(
            y=df_taxa["Laboratório"],
            x=df_taxa["Taxa Geral (%)"],
            orientation='h',
            marker_color=cores_barras,
            text=df_taxa.apply(lambda r: f"{r['Taxa Geral (%)']:.2f}% ({'+' if r['vs Nacional'] >= 0 else ''}{r['vs Nacional']:.2f}%)", axis=1),
            textposition='outside',
            textfont=dict(size=11)
        ))

        # Linha da média nacional
        fig_taxa.add_vline(x=taxa_nacional, line_dash="dash", line_color="#FFD700", line_width=2,
                          annotation_text=f"Nacional: {taxa_nacional:.2f}%", annotation_position="top")

        max_taxa = max(df_taxa["Taxa Geral (%)"].max(), taxa_nacional) if len(df_taxa) > 0 else taxa_nacional
        fig_taxa.update_layout(
            height=max(300, len(df_taxa) * 50),
            margin=dict(t=40, b=40, l=180, r=120),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Taxa (%)", range=[0, max_taxa * 1.4], gridcolor='rgba(128,128,128,0.3)'),
            yaxis=dict(title="", tickfont=dict(size=11))
        )

        st.plotly_chart(fig_taxa, use_container_width=True, key="chart_visao_taxa_multi")

        st.markdown("---")

        # ========== Gráfico de Volume por Laboratório ==========
        st.markdown("### 📈 Volume de Amostras por Laboratório")

        df_vol = df_comparacao.sort_values("Total Amostras", ascending=True)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            y=df_vol["Laboratório"],
            x=df_vol["Total Amostras"],
            orientation='h',
            marker_color='#00CED1',
            text=df_vol["Total Amostras"].apply(lambda x: f"{x:,}".replace(",", ".")),
            textposition='outside',
            textfont=dict(size=12)
        ))

        max_vol = df_vol["Total Amostras"].max() if len(df_vol) > 0 else 1
        fig_vol.update_layout(
            height=max(300, len(df_vol) * 50),
            margin=dict(t=20, b=40, l=180, r=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="", range=[0, max_vol * 1.25], showticklabels=False, showgrid=False),
            yaxis=dict(title="", tickfont=dict(size=11))
        )

        st.plotly_chart(fig_vol, use_container_width=True, key="chart_visao_vol_multi")

        st.markdown("---")

        # ========== Tabela Comparativa Completa ==========
        st.markdown("### 📋 Tabela Comparativa Detalhada")
        df_display = df_comparacao.copy()
        df_display["Taxa Geral (%)"] = df_display["Taxa Geral (%)"].apply(lambda x: f"{x:.2f}%")
        df_display["vs Nacional"] = df_display["vs Nacional"].apply(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
        df_display["Neg. Triagem"] = df_display["Neg. Triagem"].apply(lambda x: f"{x:,}".replace(",", "."))
        df_display["Pos. Triagem"] = df_display["Pos. Triagem"].apply(lambda x: f"{x:,}".replace(",", "."))
        df_display["Neg. Confirm."] = df_display["Neg. Confirm."].apply(lambda x: f"{x:,}".replace(",", "."))
        df_display["Pos. Confirm."] = df_display["Pos. Confirm."].apply(lambda x: f"{x:,}".replace(",", "."))
        df_display["Total Amostras"] = df_display["Total Amostras"].apply(lambda x: f"{x:,}".replace(",", "."))

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ========== GRÁFICO DE SUBFINALIDADE (AGREGADO) ==========
        st.markdown("### 📌 Amostras por Subfinalidade (Agregado)")

        # Agregar subfinalidades de todos os CNPJs selecionados
        subtype_data_agregado = {}
        for cnpj in selected_cnpjs:
            lab_id = labs_by_cnpj[cnpj]["id"]
            subtype_data_lab = get_samples_by_subtype(lab_id, None, selected_purpose)
            for sub_name, sub_count in subtype_data_lab.items():
                if sub_name not in subtype_data_agregado:
                    subtype_data_agregado[sub_name] = 0
                subtype_data_agregado[sub_name] += sub_count

        if subtype_data_agregado:
            sorted_subtypes = sorted(subtype_data_agregado.items(), key=lambda x: x[1], reverse=True)
            subfinalidades_lista = [s[0] for s in sorted_subtypes]
            quantidades_sub = [s[1] for s in sorted_subtypes]
            total_subfinalidade = sum(quantidades_sub)

            df_subtype = pd.DataFrame({
                'Subfinalidade': subfinalidades_lista,
                'Quantidade': quantidades_sub
            })

            df_subtype['Percentual'] = (df_subtype['Quantidade'] / total_subfinalidade * 100).round(2)
            df_subtype['Texto'] = df_subtype.apply(
                lambda row: f"{row['Quantidade']:,} ({row['Percentual']:.1f}%)".replace(",", "."), axis=1
            )

            max_qtd_sub = df_subtype['Quantidade'].max()

            cores_subfinalidades = {
                'Periódico': '#00CED1',
                'Admissão': '#4169E1',
                'Demissão': '#FF6B6B',
                'Mudança de Categoria': '#9370DB',
                'Renovação': '#FFD700',
                'Retorno ao Trabalho': '#32CD32',
                'Aleatório': '#FF8C00',
                'Pós-Acidente': '#DC143C',
                'Causa Razoável': '#8B4513',
                'Acompanhamento': '#20B2AA',
                'Pré-Admissão': '#6495ED',
                'Não informado': '#888888'
            }

            df_subtype['Cor'] = df_subtype['Subfinalidade'].map(
                lambda x: cores_subfinalidades.get(x, '#00CED1')
            )

            fig_subtype = go.Figure()

            fig_subtype.add_trace(go.Bar(
                y=df_subtype['Subfinalidade'],
                x=df_subtype['Quantidade'],
                orientation='h',
                text=df_subtype['Texto'],
                textposition='outside',
                textfont=dict(size=10),
                marker_color=df_subtype['Cor']
            ))

            fig_subtype.update_layout(
                showlegend=False,
                height=max(300, len(df_subtype) * 40),
                margin=dict(t=20, b=30, l=120, r=120),
                xaxis_title="",
                yaxis_title="",
                xaxis=dict(range=[0, max_qtd_sub * 1.3], showticklabels=False, showgrid=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_subtype, use_container_width=True, key="chart_visao_subfinalidade_multi")


        else:
            st.warning("Nenhum dado de subfinalidade encontrado")

    else:
        # ========== MODO NORMAL: UM OU NENHUM CNPJ ==========
        # Carregar dados com progress bar
        first_lab_id = selected_lab_ids[0] if selected_lab_ids and len(selected_lab_ids) == 1 else None

        tasks = [
            ("Triagem", get_triagem_data, (first_lab_id, None, selected_purpose)),
            ("Confirmatório", get_confirmatorio_data, (first_lab_id, None, selected_purpose)),
            ("Confirmatório THC", get_confirmatorio_thc_data, (first_lab_id, None, selected_purpose)),
            ("RENACH", get_renach_data, (first_lab_id, None, selected_purpose)),
            ("Finalidades", get_samples_by_purpose, (first_lab_id, None)),
            ("Subfinalidades", get_samples_by_subtype, (first_lab_id, None, selected_purpose)),
        ]
        data = loading_with_progress(tasks, "Carregando visão geral...")
        triagem_data = data["Triagem"]
        confirmatorio_data = data["Confirmatório"]
        confirmatorio_thc_data = data["Confirmatório THC"]
        renach_data = data["RENACH"]
        purpose_data = data["Finalidades"]
        subtype_data = data["Subfinalidades"]

        # Calcular métricas
        total_triagem = triagem_data["positivo"] + triagem_data["negativo"]
        total_confirmatorio = confirmatorio_data["positivo"] + confirmatorio_data["negativo"]
        total_confirmatorio_thc = confirmatorio_thc_data["positivo"] + confirmatorio_thc_data["negativo"]

        # Total confirmatório combinado (confirmatory + confirmatoryTHC)
        total_confirmatorio_geral = total_confirmatorio + total_confirmatorio_thc
        positivas_confirmatorio_geral = confirmatorio_data["positivo"] + confirmatorio_thc_data["positivo"]
        negativas_confirmatorio_geral = confirmatorio_data["negativo"] + confirmatorio_thc_data["negativo"]

        total_amostras = total_triagem + total_confirmatorio + total_confirmatorio_thc

        # Taxa geral de positividade: positivas confirmatórias / total de amostras (triagem)
        # Fórmula: (positivas confirmatorio + positivas confirmatorio THC) / total amostras triagem * 100
        taxa_geral_confirmatorio = (positivas_confirmatorio_geral / total_triagem * 100) if total_triagem > 0 else 0

        # Buscar taxa média nacional para comparação
        periodo_inicio, periodo_fim = get_selected_period()
        taxa_nacional = get_national_average_rate(periodo_inicio, periodo_fim)
        diferenca_nacional = taxa_geral_confirmatorio - taxa_nacional if taxa_nacional > 0 else 0.0

        # ========== KPIs NO TOPO ==========
        st.markdown("### 📊 Indicadores Principais")

        # Primeira linha: Taxa Geral, Total Amostras, Diferença Nacional
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

        with col_kpi1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 20px; border-radius: 10px; text-align: center;
                        border: 1px solid #FFD700;">
                <p style="color: #888; margin: 0; font-size: 14px;">Taxa Geral de Positividade</p>
                <h2 style="color: {}; margin: 5px 0;">{:.2f}%</h2>
                <p style="color: #888; margin: 0; font-size: 11px;">(Positivas Confirm. / Amostras)</p>
            </div>
            """.format("#FF6B6B" if taxa_geral_confirmatorio > 5 else "#FFD700" if taxa_geral_confirmatorio > 2 else "#00CED1", taxa_geral_confirmatorio), unsafe_allow_html=True)

        with col_kpi2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 20px; border-radius: 10px; text-align: center;
                        border: 1px solid #00CED1;">
                <p style="color: #888; margin: 0; font-size: 14px;">Total de Amostras</p>
                <h2 style="color: #00CED1; margin: 5px 0;">{:,}</h2>
                <p style="color: #888; margin: 0; font-size: 11px;">&nbsp;</p>
            </div>
            """.format(total_amostras).replace(",", "."), unsafe_allow_html=True)

        with col_kpi3:
            diferenca_texto = f"+{diferenca_nacional:.2f}%" if diferenca_nacional >= 0 else f"{diferenca_nacional:.2f}%"
            diferenca_cor = "#FF6B6B" if diferenca_nacional > 0 else "#4CAF50"
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 20px; border-radius: 10px; text-align: center;
                        border: 1px solid {};">
                <p style="color: #888; margin: 0; font-size: 14px;">vs Média Nacional</p>
                <h2 style="color: {}; margin: 5px 0;">{}</h2>
                <p style="color: #888; margin: 0; font-size: 11px;">Nacional: {:.2f}%</p>
            </div>
            """.format(diferenca_cor, diferenca_cor, diferenca_texto, taxa_nacional), unsafe_allow_html=True)

        # Segunda linha: Detalhes Triagem e Confirmatório
        st.markdown("")
        col_kpi4, col_kpi5 = st.columns(2)

        with col_kpi4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 15px; border-radius: 10px; text-align: center;
                        border: 1px solid #00CED1;">
                <p style="color: #888; margin: 0; font-size: 14px;">🔬 Triagem</p>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Negativas</p>
                        <h3 style="color: #00CED1; margin: 5px 0;">{:,}</h3>
                    </div>
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Positivas</p>
                        <h3 style="color: #FF6B6B; margin: 5px 0;">{:,}</h3>
                    </div>
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Total</p>
                        <h3 style="color: #E8E8E8; margin: 5px 0;">{:,}</h3>
                    </div>
                </div>
            </div>
            """.format(triagem_data["negativo"], triagem_data["positivo"], total_triagem).replace(",", "."), unsafe_allow_html=True)

        with col_kpi5:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 15px; border-radius: 10px; text-align: center;
                        border: 1px solid #9370DB;">
                <p style="color: #888; margin: 0; font-size: 14px;">🧪 Confirmatório (Geral + THC)</p>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Negativas</p>
                        <h3 style="color: #00CED1; margin: 5px 0;">{:,}</h3>
                    </div>
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Positivas</p>
                        <h3 style="color: #FF6B6B; margin: 5px 0;">{:,}</h3>
                    </div>
                    <div>
                        <p style="color: #888; margin: 0; font-size: 12px;">Total</p>
                        <h3 style="color: #E8E8E8; margin: 5px 0;">{:,}</h3>
                    </div>
                </div>
            </div>
            """.format(negativas_confirmatorio_geral, positivas_confirmatorio_geral, total_confirmatorio_geral).replace(",", "."), unsafe_allow_html=True)

        st.markdown("---")

        # ========== GRÁFICO DE BARRAS EMPILHADAS: Comparação por Tipo de Análise ==========
        st.markdown("### 📈 Comparação por Tipo de Análise")

        # Preparar dados para barras empilhadas
        df_analises = pd.DataFrame({
            'Tipo': ['Triagem', 'Confirmatório', 'Confirm. THC'],
            'Negativos': [triagem_data["negativo"], confirmatorio_data["negativo"], confirmatorio_thc_data["negativo"]],
            'Positivos': [triagem_data["positivo"], confirmatorio_data["positivo"], confirmatorio_thc_data["positivo"]],
            'Total': [total_triagem, total_confirmatorio, total_confirmatorio_thc]
        })

        # Calcular taxas
        df_analises['Taxa (%)'] = df_analises.apply(
            lambda row: round(row['Positivos'] / row['Total'] * 100, 2) if row['Total'] > 0 else 0, axis=1
        )

        if df_analises['Total'].sum() > 0:
            # Criar gráfico de barras agrupadas
            fig_barras = go.Figure()

            # Barras de Negativos
            fig_barras.add_trace(go.Bar(
                name='Negativos',
                x=df_analises['Tipo'],
                y=df_analises['Negativos'],
                marker_color='#00CED1',
                text=df_analises['Negativos'].apply(lambda x: f"{x:,}".replace(",", ".")),
                textposition='auto',
                textfont=dict(size=12)
            ))

            # Barras de Positivos
            fig_barras.add_trace(go.Bar(
                name='Positivos',
                x=df_analises['Tipo'],
                y=df_analises['Positivos'],
                marker_color='#FF6B6B',
                text=df_analises['Positivos'].apply(lambda x: f"{x:,}".replace(",", ".")),
                textposition='auto',
                textfont=dict(size=12)
            ))

            fig_barras.update_layout(
                barmode='group',
                height=400,
                margin=dict(t=30, b=50, l=50, r=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis_title="",
                yaxis_title="Quantidade de Amostras"
            )

            st.plotly_chart(fig_barras, use_container_width=True, key="chart_visao_barras")
        else:
            st.warning("Nenhum dado encontrado para o período selecionado")

        st.markdown("---")

        # ========== RENACH E FINALIDADE LADO A LADO ==========
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("### 📋 Status RENACH")

            total_renach = renach_data["no_renach"] + renach_data["fora_renach"]

            if total_renach > 0:
                pct_no_renach = (renach_data["no_renach"] / total_renach) * 100
                pct_fora_renach = (renach_data["fora_renach"] / total_renach) * 100

                df_renach = pd.DataFrame({
                    'Status': ['No RENACH', 'Fora do RENACH'],
                    'Quantidade': [renach_data["no_renach"], renach_data["fora_renach"]],
                    'Percentual': [pct_no_renach, pct_fora_renach]
                })

                fig_renach = px.pie(
                    df_renach,
                    values='Quantidade',
                    names='Status',
                    color='Status',
                    color_discrete_map={'No RENACH': '#00CED1', 'Fora do RENACH': '#FF6B6B'},
                    hole=0.4
                )

                fig_renach.update_traces(
                    textposition='inside',
                    textinfo='value+percent',
                    texttemplate='%{value:,.0f}<br>(%{percent:.1%})',
                    textfont_size=12
                )

                fig_renach.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    height=350,
                    margin=dict(t=20, b=50, l=20, r=20)
                )

                st.plotly_chart(fig_renach, use_container_width=True, key="chart_visao_renach")
            else:
                st.warning("Nenhum dado de RENACH encontrado")

        with col6:
            st.markdown("### 🎯 Amostras por Finalidade")

            if purpose_data:
                sorted_purposes = sorted(purpose_data.items(), key=lambda x: x[1], reverse=True)
                finalidades_lista = [p[0] for p in sorted_purposes]
                quantidades = [p[1] for p in sorted_purposes]
                total_finalidade = sum(quantidades)

                df_purpose = pd.DataFrame({
                    'Finalidade': finalidades_lista,
                    'Quantidade': quantidades
                })

                df_purpose['Percentual'] = (df_purpose['Quantidade'] / total_finalidade * 100).round(2)
                df_purpose['Texto'] = df_purpose.apply(
                    lambda row: f"{row['Quantidade']:,} ({row['Percentual']:.1f}%)".replace(",", "."), axis=1
                )

                max_qtd = df_purpose['Quantidade'].max()

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
                    textfont_size=10,
                    cliponaxis=False
                )

                fig_purpose.update_layout(
                    showlegend=False,
                    height=350,
                    margin=dict(t=20, b=30, l=80, r=120),
                    xaxis_title="",
                    yaxis_title="",
                    xaxis=dict(range=[0, max_qtd * 1.3], showticklabels=False, showgrid=False),
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig_purpose, use_container_width=True, key="chart_visao_finalidade")
            else:
                st.warning("Nenhum dado de finalidade encontrado")

        st.markdown("---")

        # ========== GRÁFICO DE SUBFINALIDADE ==========
        st.markdown("### 📌 Amostras por Subfinalidade")

        if subtype_data:
            # Filtrar pelo subtype selecionado se houver
            if selected_subtype:
                # Mapear de volta para nome legível
                subtype_names_inv = {
                    "periodic": "Periódico",
                    "categoryChange": "Mudança de Categoria",
                    "hiring": "Admissão",
                    "renovation": "Renovação",
                    "resignation": "Demissão",
                    "returnToWork": "Retorno ao Trabalho",
                    "random": "Aleatório",
                    "postAccident": "Pós-Acidente",
                    "reasonableCause": "Causa Razoável",
                    "followUp": "Acompanhamento",
                    "preEmployment": "Pré-Admissão"
                }
                selected_subtype_name = subtype_names_inv.get(selected_subtype, selected_subtype)
                if selected_subtype_name in subtype_data:
                    subtype_data_filtered = {selected_subtype_name: subtype_data[selected_subtype_name]}
                else:
                    subtype_data_filtered = subtype_data
            else:
                subtype_data_filtered = subtype_data

            sorted_subtypes = sorted(subtype_data_filtered.items(), key=lambda x: x[1], reverse=True)
            subfinalidades_lista = [s[0] for s in sorted_subtypes]
            quantidades_sub = [s[1] for s in sorted_subtypes]
            total_subfinalidade = sum(quantidades_sub)

            df_subtype = pd.DataFrame({
                'Subfinalidade': subfinalidades_lista,
                'Quantidade': quantidades_sub
            })

            df_subtype['Percentual'] = (df_subtype['Quantidade'] / total_subfinalidade * 100).round(2)
            df_subtype['Texto'] = df_subtype.apply(
                lambda row: f"{row['Quantidade']:,} ({row['Percentual']:.1f}%)".replace(",", "."), axis=1
            )

            max_qtd_sub = df_subtype['Quantidade'].max()

            # Definir cores para diferentes subfinalidades
            cores_subfinalidades = {
                'Periódico': '#00CED1',
                'Admissão': '#4169E1',
                'Demissão': '#FF6B6B',
                'Mudança de Categoria': '#9370DB',
                'Renovação': '#FFD700',
                'Retorno ao Trabalho': '#32CD32',
                'Aleatório': '#FF8C00',
                'Pós-Acidente': '#DC143C',
                'Causa Razoável': '#8B4513',
                'Acompanhamento': '#20B2AA',
                'Pré-Admissão': '#6495ED',
                'Não informado': '#888888'
            }

            df_subtype['Cor'] = df_subtype['Subfinalidade'].map(
                lambda x: cores_subfinalidades.get(x, '#00CED1')
            )

            fig_subtype = go.Figure()

            fig_subtype.add_trace(go.Bar(
                y=df_subtype['Subfinalidade'],
                x=df_subtype['Quantidade'],
                orientation='h',
                text=df_subtype['Texto'],
                textposition='outside',
                textfont=dict(size=10),
                marker_color=df_subtype['Cor']
            ))

            fig_subtype.update_layout(
                showlegend=False,
                height=max(300, len(df_subtype) * 40),
                margin=dict(t=20, b=30, l=120, r=120),
                xaxis_title="",
                yaxis_title="",
                xaxis=dict(range=[0, max_qtd_sub * 1.3], showticklabels=False, showgrid=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_subtype, use_container_width=True, key="chart_visao_subfinalidade")

            # Mostrar resumo em texto
            st.caption(f"Total: {total_subfinalidade:,} amostras em {len(df_subtype)} subfinalidades".replace(",", "."))
        else:
            st.warning("Nenhum dado de subfinalidade encontrado para os filtros selecionados")


def render_perfil_demografico():
    """Página de Perfil Demográfico por Substância"""
    st.title("👤 Perfil Demográfico")
    st.caption("Análise do perfil dos doadores que testaram positivo para cada substância")

    # Filtros
    st.markdown("### 🔍 Filtros")

    col_filtro1, col_filtro2 = st.columns(2)

    # Filtro de Finalidade
    with col_filtro1:
        finalidades = {
            "Todas": None,
            "CNH": "cnh",
            "CLT": "clt",
            "CLT + CNH": "cltCnh",
            "Concurso Público": "civilService",
            "Contra Prova": "againstProof",
        }
        selected_finalidade_name = st.selectbox(
            "Finalidade da Amostra",
            options=list(finalidades.keys()),
            index=0,
            key="demo_filtro_finalidade"
        )
        selected_purpose = finalidades[selected_finalidade_name]

    # Filtro de Laboratório por CNPJ
    with col_filtro2:
        labs_by_cnpj = get_laboratories_by_cnpj()
        cnpj_options = ["Todos"] + sorted(labs_by_cnpj.keys())

        selected_cnpj = st.selectbox(
            "CNPJ Laboratório (PCL)",
            options=cnpj_options,
            index=0,
            key="demo_filtro_cnpj"
        )

        if selected_cnpj and selected_cnpj != "Todos":
            lab_info = labs_by_cnpj.get(selected_cnpj, {})
            selected_lab_id = lab_info.get("id")
            selected_lab_name = lab_info.get("name", "")
            selected_lab_city = lab_info.get("city", "")
            selected_lab_state = lab_info.get("state", "")
        else:
            selected_lab_id = None
            selected_lab_name = None
            selected_lab_city = None
            selected_lab_state = None

    # Exibir informações do laboratório selecionado
    if selected_lab_name:
        location_info = f"{selected_lab_city}/{selected_lab_state}" if selected_lab_city and selected_lab_state else ""
        st.success(f"🏢 **{selected_lab_name}** {f'({location_info})' if location_info else ''}")

    st.markdown("---")

    # Carregar dados demográficos completos
    df_demo_raw = loading_single(
        get_demographic_raw_data,
        "Carregando dados demográficos...",
        selected_lab_id, None, selected_purpose
    )

    if df_demo_raw is not None and not df_demo_raw.empty:
        # Lista de substâncias disponíveis
        substancias_disponiveis = sorted(df_demo_raw["substancia"].unique().tolist())

        # Selectbox para escolher substância
        col_select, col_empty = st.columns([1, 2])
        with col_select:
            # Inicializar session state se necessário
            if "substancia_selecionada" not in st.session_state:
                st.session_state.substancia_selecionada = None

            substancia_escolhida = st.selectbox(
                "Selecione uma substância para análise detalhada:",
                options=["Top 5 Substâncias"] + substancias_disponiveis,
                key="select_substancia_demo"
            )

        # Atualizar session state
        if substancia_escolhida == "Top 5 Substâncias":
            st.session_state.substancia_selecionada = None
        else:
            st.session_state.substancia_selecionada = substancia_escolhida

        

        if st.session_state.substancia_selecionada is None:
            # MODO TOP 5: Mostrar cards das 5 substâncias com mais positivos
            st.markdown("### 🏆 Top 5 Substâncias - Perfil Mais Comum")

            # Contar total de positivos por substância
            contagem_substancias = df_demo_raw.groupby("substancia").size().reset_index(name="total")
            contagem_substancias = contagem_substancias.sort_values("total", ascending=False)
            top_5_substancias = contagem_substancias.head(5)["substancia"].tolist()

            # Criar cards para cada substância do top 5
            for i, substancia in enumerate(top_5_substancias):
                df_sub = df_demo_raw[df_demo_raw["substancia"] == substancia]
                total_positivos = len(df_sub)

                # Perfil mais comum
                perfil = df_sub.groupby(["sexo", "faixa_etaria", "purposeType", "purposeSubType"]).size().reset_index(name="qtd")
                if not perfil.empty:
                    top_perfil = perfil.nlargest(1, "qtd").iloc[0]

                    sexo = top_perfil["sexo"]
                    faixa = top_perfil["faixa_etaria"]
                    tipo_exame = top_perfil["purposeType"]
                    subtipo = top_perfil["purposeSubType"]
                    qtd_perfil = int(top_perfil["qtd"])
                    pct_perfil = (qtd_perfil / total_positivos * 100) if total_positivos > 0 else 0

                    # Formatar subtipo
                    subtipos_map = {
                        "periodic": "Periódico",
                        "hiring": "Admissional",
                        "resignation": "Demissional",
                        "firstLicense": "Primeira Habilitação",
                        "firstCnh": "Primeira Habilitação",
                        "renovation": "Renovação",
                        "categoryChange": "Mudança de Categoria",
                        "functionChange": "Mudança de Função",
                        "return": "Retorno ao Trabalho"
                    }
                    subtipo_texto = f" ({subtipos_map.get(subtipo, subtipo)})" if subtipo else ""

                    # Formatar tipo exame
                    tipos_map = {
                        "cnh": "CNH",
                        "clt": "CLT",
                        "cltCnh": "CLT + CNH",
                        "admissional": "Admissional",
                        "periodico": "Periódico",
                        "demissional": "Demissional"
                    }
                    tipo_formatado = tipos_map.get(tipo_exame, tipo_exame) if tipo_exame else "N/A"

                    # Card visual
                    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                padding: 20px; border-radius: 10px; margin-bottom: 15px;
                                border-left: 4px solid {'#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2 else '#4a4a6a'};">
                        <h4 style="margin: 0; color: white;">{medal} {substancia}</h4>
                        <p style="color: #aaa; margin: 5px 0 0 0; font-size: 14px;">
                            <b>{total_positivos}</b> positivos | Perfil mais comum: <b>{sexo}</b>, <b>{faixa}</b> anos,
                            exame <b>{tipo_formatado}</b>{subtipo_texto} ({qtd_perfil} casos - {pct_perfil:.1f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            # MODO DETALHADO: Análise da substância selecionada
            substancia = st.session_state.substancia_selecionada
            df_sub = df_demo_raw[df_demo_raw["substancia"] == substancia]
            total_positivos = len(df_sub)

            st.markdown(f"### 🔬 Análise Detalhada: {substancia}")

            # Perfil mais comum
            perfil = df_sub.groupby(["sexo", "faixa_etaria", "purposeType", "purposeSubType"]).size().reset_index(name="qtd")
            if not perfil.empty:
                top_perfil = perfil.nlargest(1, "qtd").iloc[0]
                qtd_casos = int(top_perfil['qtd'])
                pct_casos = (qtd_casos / total_positivos * 100) if total_positivos > 0 else 0

                subtipos_map = {
                    "periodic": "Periódico",
                    "hiring": "Admissional",
                    "resignation": "Demissional",
                    "firstLicense": "Primeira Habilitação",
                    "firstCnh": "Primeira Habilitação",
                    "renovation": "Renovação",
                    "categoryChange": "Mudança de Categoria",
                    "functionChange": "Mudança de Função",
                    "return": "Retorno ao Trabalho"
                }
                tipos_map = {
                    "cnh": "CNH",
                    "clt": "CLT",
                    "cltCnh": "CLT + CNH",
                    "admissional": "Admissional",
                    "periodico": "Periódico",
                    "demissional": "Demissional"
                }

                subtipo_texto = subtipos_map.get(top_perfil['purposeSubType'], top_perfil['purposeSubType']) if top_perfil['purposeSubType'] else ""
                tipo_formatado = tipos_map.get(top_perfil['purposeType'], top_perfil['purposeType']) if top_perfil['purposeType'] else "N/A"

                # Card bonito para o perfil mais comum
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                            padding: 25px; border-radius: 12px; margin: 15px 0;
                            border: 1px solid #00CED1;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <p style="color: #888; margin: 0; font-size: 12px; text-transform: uppercase;">Perfil Mais Comum</p>
                            <h3 style="color: white; margin: 8px 0;">👤 {top_perfil['sexo']} ({top_perfil['faixa_etaria']} anos)</h3>
                            <p style="color: #aaa; margin: 0; font-size: 14px;">
                                Exame: <b style="color: #00CED1;">{tipo_formatado}</b>{f' - {subtipo_texto}' if subtipo_texto else ''}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <p style="color: #888; margin: 0; font-size: 12px;">CASOS</p>
                            <h2 style="color: #00CED1; margin: 5px 0;">{qtd_casos:,}</h2>
                            <p style="color: #FFD700; margin: 0; font-size: 16px; font-weight: bold;">{pct_casos:.1f}%</p>
                            <p style="color: #666; margin: 0; font-size: 11px;">do total de {total_positivos:,}</p>
                        </div>
                    </div>
                </div>
                """.replace(",", "."), unsafe_allow_html=True)

            st.markdown("---")

            # Gráficos lado a lado: Estado e Sexo
            col_estado, col_sexo = st.columns(2)

            with col_estado:
                st.markdown("#### 📍 Recorrência por Estado")
                if "estado" in df_sub.columns:
                    # Extrair código do estado (pode vir como dict com code/name)
                    def extrair_estado(val):
                        if isinstance(val, dict):
                            return val.get("code", val.get("name", "N/A"))
                        return val if val else "N/A"

                    df_sub = df_sub.copy()
                    df_sub["estado_str"] = df_sub["estado"].apply(extrair_estado)

                    estado_counts = df_sub["estado_str"].value_counts().reset_index()
                    estado_counts.columns = ["Estado", "Quantidade"]
                    # Filtrar N/A, None e vazios
                    estado_counts = estado_counts[
                        (estado_counts["Estado"] != "N/A") &
                        (estado_counts["Estado"].notna()) &
                        (estado_counts["Estado"] != "")
                    ].head(10)

                    if not estado_counts.empty:
                        fig_estado = px.bar(
                            estado_counts,
                            x="Quantidade",
                            y="Estado",
                            orientation="h",
                            text="Quantidade",
                            color="Quantidade",
                            color_continuous_scale=["#00CED1", "#FF6B6B"]
                        )
                        fig_estado.update_traces(
                            textposition="outside",
                            texttemplate="%{text:,.0f}",
                            cliponaxis=False
                        )
                        fig_estado.update_layout(
                            height=400,
                            margin=dict(t=20, b=20, l=50, r=120),
                            yaxis=dict(autorange="reversed"),
                            coloraxis_showscale=False,
                            xaxis_title="",
                            yaxis_title="",
                            xaxis=dict(range=[0, estado_counts["Quantidade"].max() * 1.25])
                        )
                        st.plotly_chart(fig_estado, use_container_width=True, key="chart_estado_demo")
                    else:
                        st.info("Dados de estado não disponíveis")
                else:
                    st.info("Dados de estado não disponíveis")

            with col_sexo:
                st.markdown("#### 👫 Recorrência por Sexo")
                sexo_counts = df_sub["sexo"].value_counts().reset_index()
                sexo_counts.columns = ["Sexo", "Quantidade"]

                if not sexo_counts.empty:
                    fig_sexo = px.pie(
                        sexo_counts,
                        values="Quantidade",
                        names="Sexo",
                        color="Sexo",
                        color_discrete_map={"Masculino": "#4169E1", "Feminino": "#FF69B4", "N/A": "#808080"},
                        hole=0.4
                    )
                    fig_sexo.update_traces(textposition="inside", textinfo="percent+value")
                    fig_sexo.update_layout(
                        height=350,
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_sexo, use_container_width=True, key="chart_sexo_demo")

            st.markdown("---")

            # Gráficos de idade por sexo
            st.markdown("#### 📊 Distribuição por Faixa Etária")

            col_masc, col_fem = st.columns(2)

            # Ordem das faixas etárias
            ordem_faixas = ["< 18", "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60+"]

            with col_masc:
                st.markdown("##### 👨 Homens")
                df_masc = df_sub[df_sub["sexo"] == "Masculino"]
                if not df_masc.empty:
                    idade_masc = df_masc["faixa_etaria"].value_counts().reset_index()
                    idade_masc.columns = ["Faixa Etária", "Quantidade"]
                    # Ordenar
                    idade_masc["ordem"] = idade_masc["Faixa Etária"].apply(lambda x: ordem_faixas.index(x) if x in ordem_faixas else 99)
                    idade_masc = idade_masc.sort_values("ordem").drop("ordem", axis=1)

                    fig_masc = px.bar(
                        idade_masc,
                        x="Faixa Etária",
                        y="Quantidade",
                        text="Quantidade",
                        color="Quantidade",
                        color_continuous_scale=["#87CEEB", "#4169E1"]
                    )
                    fig_masc.update_traces(textposition="outside")
                    fig_masc.update_layout(
                        height=400,
                        margin=dict(t=30, b=50, l=20, r=20),
                        coloraxis_showscale=False,
                        xaxis_title="",
                        yaxis_title="Quantidade"
                    )
                    st.plotly_chart(fig_masc, use_container_width=True, key="chart_idade_masc")
                else:
                    st.info("Nenhum dado masculino")

            with col_fem:
                st.markdown("##### 👩 Mulheres")
                df_fem = df_sub[df_sub["sexo"] == "Feminino"]
                if not df_fem.empty:
                    idade_fem = df_fem["faixa_etaria"].value_counts().reset_index()
                    idade_fem.columns = ["Faixa Etária", "Quantidade"]
                    # Ordenar
                    idade_fem["ordem"] = idade_fem["Faixa Etária"].apply(lambda x: ordem_faixas.index(x) if x in ordem_faixas else 99)
                    idade_fem = idade_fem.sort_values("ordem").drop("ordem", axis=1)

                    fig_fem = px.bar(
                        idade_fem,
                        x="Faixa Etária",
                        y="Quantidade",
                        text="Quantidade",
                        color="Quantidade",
                        color_continuous_scale=["#FFB6C1", "#FF69B4"]
                    )
                    fig_fem.update_traces(textposition="outside")
                    fig_fem.update_layout(
                        height=400,
                        margin=dict(t=30, b=50, l=20, r=20),
                        coloraxis_showscale=False,
                        xaxis_title="",
                        yaxis_title="Quantidade"
                    )
                    st.plotly_chart(fig_fem, use_container_width=True, key="chart_idade_fem")
                else:
                    st.info("Nenhum dado feminino")

    else:
        st.info("Nenhum dado demográfico disponível para o período selecionado")


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

        # Taxa geral de positividade = positivas confirmatórias / total de amostras (triagem)
        # Fórmula: (Nº de amostras positivas confirmatórias ÷ Nº total de amostras triagem) × 100
        metrics["taxa_geral"] = (metrics["positivas_confirmatorio"] / metrics["total_amostras"] * 100) if metrics["total_amostras"] > 0 else 0.0

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
    Taxa = positivas confirmatórias / total amostras triagem * 100
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

            # Buscar triagem para total de amostras
            monthly_triagem = get_monthly_positivity_data(
                laboratory_ids=lab_ids,
                analysis_type="screening",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            # Buscar confirmatório para positivas
            monthly_conf = get_monthly_positivity_data(
                laboratory_ids=lab_ids,
                analysis_type="confirmatory",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            # Buscar confirmatório THC para positivas
            monthly_conf_thc = get_monthly_positivity_data(
                laboratory_ids=lab_ids,
                analysis_type="confirmatoryTHC",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            # Total de amostras = triagem
            total_triagem = sum(m.get("positivo", 0) + m.get("negativo", 0) for m in monthly_triagem.values())

            # Positivas = confirmatório + confirmatório THC
            positivas_conf = sum(m.get("positivo", 0) for m in monthly_conf.values())
            positivas_conf_thc = sum(m.get("positivo", 0) for m in monthly_conf_thc.values())
            total_positivo = positivas_conf + positivas_conf_thc

            if total_triagem > 0:
                state_data[state] = {
                    "positivo": total_positivo,
                    "negativo": total_triagem - total_positivo,
                    "total": total_triagem,
                    "taxa": (total_positivo / total_triagem * 100)
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
    Taxa = positivas confirmatórias / total amostras triagem * 100
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
            "againstProof": "Contra Prova",
            "periodic": "Periódico",
            "categoryChange": "Mudança de Categoria",
            "hiring": "Admissão",
            "renovation": "Renovação",
            "resignation": "Demissão"
        }

        purpose_data = {}
        for purpose_key, purpose_name in purpose_names.items():
            # Buscar triagem para total de amostras
            monthly_triagem = get_monthly_positivity_data(
                laboratory_ids=laboratory_ids,
                purpose_type=purpose_key,
                analysis_type="screening",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            # Buscar confirmatório para positivas
            monthly_conf = get_monthly_positivity_data(
                laboratory_ids=laboratory_ids,
                purpose_type=purpose_key,
                analysis_type="confirmatory",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            # Buscar confirmatório THC para positivas
            monthly_conf_thc = get_monthly_positivity_data(
                laboratory_ids=laboratory_ids,
                purpose_type=purpose_key,
                analysis_type="confirmatoryTHC",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            # Total de amostras = triagem
            total_triagem = sum(m.get("positivo", 0) + m.get("negativo", 0) for m in monthly_triagem.values())

            # Positivas = confirmatório + confirmatório THC
            positivas_conf = sum(m.get("positivo", 0) for m in monthly_conf.values())
            positivas_conf_thc = sum(m.get("positivo", 0) for m in monthly_conf_thc.values())
            total_positivo = positivas_conf + positivas_conf_thc

            if total_triagem > 0:
                purpose_data[purpose_name] = {
                    "positivo": total_positivo,
                    "negativo": total_triagem - total_positivo,
                    "total": total_triagem,
                    "taxa": (total_positivo / total_triagem * 100)
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
    Taxa = positivas confirmatórias / total amostras triagem * 100
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
            # Buscar triagem para total de amostras
            monthly_triagem = get_monthly_positivity_data(
                laboratory_ids=laboratory_ids,
                renach_status=renach_status,
                analysis_type="screening",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            # Buscar confirmatório para positivas
            monthly_conf = get_monthly_positivity_data(
                laboratory_ids=laboratory_ids,
                renach_status=renach_status,
                analysis_type="confirmatory",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            # Buscar confirmatório THC para positivas
            monthly_conf_thc = get_monthly_positivity_data(
                laboratory_ids=laboratory_ids,
                renach_status=renach_status,
                analysis_type="confirmatoryTHC",
                start_date_filter=start_date,
                end_date_filter=end_date
            )

            # Total de amostras = triagem
            total_triagem = sum(m.get("positivo", 0) + m.get("negativo", 0) for m in monthly_triagem.values())

            # Positivas = confirmatório + confirmatório THC
            positivas_conf = sum(m.get("positivo", 0) for m in monthly_conf.values())
            positivas_conf_thc = sum(m.get("positivo", 0) for m in monthly_conf_thc.values())
            total_positivo = positivas_conf + positivas_conf_thc

            if total_triagem > 0:
                renach_data[renach_name] = {
                    "positivo": total_positivo,
                    "negativo": total_triagem - total_positivo,
                    "total": total_triagem,
                    "taxa": (total_positivo / total_triagem * 100)
                }

        set_cached_data("positivity_renach", cache_key, renach_data)
        return renach_data

    except Exception as e:
        st.error(f"Erro ao buscar dados por RENACH: {e}")
        return {}


def get_demographic_raw_data(laboratory_id: str = None, month: int = None, purpose_type: str = None) -> pd.DataFrame:
    """
    Retorna dados demográficos brutos para análise.
    Inclui: substancia, sexo, faixa_etaria, estado, purposeType, purposeSubType
    """
    selected_start, selected_end = get_selected_period()
    cache_key = generate_cache_key("demographic_raw", laboratory_id, month, purpose_type, selected_start.isoformat(), selected_end.isoformat())
    cached = get_cached_data("demographic_raw", cache_key)
    if cached is not None:
        return cached

    try:
        client = get_mongo_client()
        db = client["ctox"]

        # Período
        if month:
            year = datetime.now().year
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year, 12, 31, 23, 59, 59)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        else:
            start_date = selected_start
            end_date = selected_end

        # Tipos de análise - sempre busca todos para dados demográficos
        analysis_types = ["screening", "confirmatory", "confirmatoryTHC"]

        # Buscar lotes do período
        lots_collection = db["lots"]
        lots_query = {
            "analysisType": {"$in": analysis_types},
            "createdAt": {"$gte": start_date, "$lte": end_date}
        }

        # Filtrar por laboratório se especificado
        if laboratory_id:
            lots_query["_laboratory"] = ObjectId(laboratory_id) if isinstance(laboratory_id, str) else laboratory_id

        lots = list(lots_collection.find(lots_query, {"code": 1}))

        if not lots:
            return pd.DataFrame()

        lot_codes = [lot.get('code') for lot in lots if lot.get('code')]

        # Pipeline de agregação com estado
        pipeline = [
            {"$match": {"_lot": {"$in": lot_codes}}},
            {"$unwind": "$samples"},
            {"$match": {"samples.positive": True}},
            {"$unwind": "$samples._compound"},
            {"$match": {"samples._compound.positive": True}},
            {
                "$lookup": {
                    "from": "compounds",
                    "localField": "samples._compound._id",
                    "foreignField": "_id",
                    "as": "compoundInfo"
                }
            },
            {"$unwind": "$compoundInfo"},
            {
                "$lookup": {
                    "from": "chainofcustodies",
                    "localField": "samples._sample",
                    "foreignField": "sample.code",
                    "as": "chainInfo"
                }
            },
            {"$unwind": {"path": "$chainInfo", "preserveNullAndEmptyArrays": True}},
            {
                "$lookup": {
                    "from": "gatherings",
                    "localField": "chainInfo._id",
                    "foreignField": "_chainOfCustody",
                    "as": "gatheringInfo"
                }
            },
            {"$unwind": {"path": "$gatheringInfo", "preserveNullAndEmptyArrays": True}},
        ]

        # Adicionar filtro de purpose_type se especificado
        if purpose_type:
            pipeline.append({"$match": {"gatheringInfo.purpose.type": purpose_type}})

        pipeline.append({
            "$project": {
                "substancia": "$compoundInfo.name",
                "sexo": "$chainInfo.donor.gender",
                "birthDate": "$chainInfo.donor.birthDate",
                "estado": "$chainInfo.donor.address.state.code",
                "purposeType": "$gatheringInfo.purpose.type",
                "purposeSubType": "$gatheringInfo.purpose.subType"
            }
        })

        results_collection = db["results"]
        results = list(results_collection.aggregate(pipeline, allowDiskUse=True))

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Calcular faixa etária
        hoje = datetime.now()

        def calcular_faixa_etaria(birth_date):
            if pd.isna(birth_date) or birth_date is None:
                return "N/A"
            try:
                idade = (hoje - birth_date).days // 365
                if idade < 18:
                    return "< 18"
                elif idade <= 24:
                    return "18-24"
                elif idade <= 29:
                    return "25-29"
                elif idade <= 34:
                    return "30-34"
                elif idade <= 39:
                    return "35-39"
                elif idade <= 44:
                    return "40-44"
                elif idade <= 49:
                    return "45-49"
                elif idade <= 54:
                    return "50-54"
                elif idade <= 59:
                    return "55-59"
                else:
                    return "60+"
            except:
                return "N/A"

        df["faixa_etaria"] = df["birthDate"].apply(calcular_faixa_etaria)

        # Mapear sexo
        sexo_map = {"m": "Masculino", "f": "Feminino"}
        df["sexo"] = df["sexo"].map(sexo_map).fillna("N/A")

        # Preencher valores nulos
        df["purposeType"] = df["purposeType"].fillna("N/A")
        df["purposeSubType"] = df["purposeSubType"].fillna("")
        df["estado"] = df["estado"].fillna("N/A")

        # Remover coluna birthDate (não precisa mais)
        df = df.drop(columns=["birthDate", "_id"], errors="ignore")

        set_cached_data("demographic_raw", cache_key, df)
        return df

    except Exception as e:
        st.error(f"Erro ao buscar dados demográficos: {e}")
        return pd.DataFrame()


def get_demographic_profile_by_substance(laboratory_id: str = None, month: int = None, analysis_type: str = "all") -> dict:
    """
    Busca o perfil demográfico mais comum para cada substância positiva.
    Retorna dict {substância: {sexo, faixa_etaria, tipo_exame, subtipo, quantidade}}
    """
    selected_start, selected_end = get_selected_period()
    cache_key = generate_cache_key("demographic_profile", laboratory_id, month, analysis_type, selected_start.isoformat(), selected_end.isoformat())
    cached = get_cached_data("demographic_profile", cache_key)
    if cached is not None:
        return cached

    try:
        client = get_mongo_client()
        db = client["ctox"]

        # Período
        if month:
            year = datetime.now().year
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year, 12, 31, 23, 59, 59)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        else:
            start_date = selected_start
            end_date = selected_end

        # Tipos de análise
        if analysis_type == "all":
            analysis_types = ["screening", "confirmatory", "confirmatoryTHC"]
        else:
            analysis_types = [analysis_type]

        # Buscar lotes do período
        lots_collection = db["lots"]
        lots = list(lots_collection.find(
            {
                "analysisType": {"$in": analysis_types},
                "createdAt": {"$gte": start_date, "$lte": end_date}
            },
            {"code": 1}
        ))

        if not lots:
            return {}

        lot_codes = [lot.get('code') for lot in lots if lot.get('code')]

        # Pipeline de agregação
        pipeline = [
            # Filtrar pelos lotes do período
            {"$match": {"_lot": {"$in": lot_codes}}},

            # Desaninhar samples
            {"$unwind": "$samples"},

            # Filtrar apenas amostras positivas
            {"$match": {"samples.positive": True}},

            # Desaninhar compounds
            {"$unwind": "$samples._compound"},

            # Filtrar apenas compounds positivos
            {"$match": {"samples._compound.positive": True}},

            # Lookup para nome do compound
            {
                "$lookup": {
                    "from": "compounds",
                    "localField": "samples._compound._id",
                    "foreignField": "_id",
                    "as": "compoundInfo"
                }
            },
            {"$unwind": "$compoundInfo"},

            # Lookup para chainofcustodies (dados do doador)
            {
                "$lookup": {
                    "from": "chainofcustodies",
                    "localField": "samples._sample",
                    "foreignField": "sample.code",
                    "as": "chainInfo"
                }
            },
            {"$unwind": {"path": "$chainInfo", "preserveNullAndEmptyArrays": True}},

            # Lookup para gatherings (finalidade)
            {
                "$lookup": {
                    "from": "gatherings",
                    "localField": "chainInfo._id",
                    "foreignField": "_chainOfCustody",
                    "as": "gatheringInfo"
                }
            },
            {"$unwind": {"path": "$gatheringInfo", "preserveNullAndEmptyArrays": True}},

            # Projetar campos necessários
            {
                "$project": {
                    "substancia": "$compoundInfo.name",
                    "sexo": "$chainInfo.donor.gender",
                    "birthDate": "$chainInfo.donor.birthDate",
                    "purposeType": "$gatheringInfo.purpose.type",
                    "purposeSubType": "$gatheringInfo.purpose.subType"
                }
            }
        ]

        results_collection = db["results"]
        results = list(results_collection.aggregate(pipeline, allowDiskUse=True))

        if not results:
            return {}

        # Processar resultados
        df = pd.DataFrame(results)

        # Calcular idade e faixa etária
        hoje = datetime.now()

        def calcular_faixa_etaria(birth_date):
            if pd.isna(birth_date) or birth_date is None:
                return "N/A"
            try:
                idade = (hoje - birth_date).days // 365
                if idade < 18:
                    return "< 18"
                elif idade <= 24:
                    return "18-24"
                elif idade <= 29:
                    return "25-29"
                elif idade <= 34:
                    return "30-34"
                elif idade <= 39:
                    return "35-39"
                elif idade <= 44:
                    return "40-44"
                elif idade <= 49:
                    return "45-49"
                elif idade <= 54:
                    return "50-54"
                elif idade <= 59:
                    return "55-59"
                else:
                    return "60+"
            except:
                return "N/A"

        df["faixa_etaria"] = df["birthDate"].apply(calcular_faixa_etaria)

        # Mapear sexo
        sexo_map = {"m": "Masculino", "f": "Feminino"}
        df["sexo"] = df["sexo"].map(sexo_map).fillna("N/A")

        # Preencher valores nulos
        df["purposeType"] = df["purposeType"].fillna("N/A")
        df["purposeSubType"] = df["purposeSubType"].fillna("")

        # Agrupar e contar
        grouped = df.groupby(
            ["substancia", "sexo", "faixa_etaria", "purposeType", "purposeSubType"]
        ).size().reset_index(name="quantidade")

        # Para cada substância, pegar o perfil mais comum
        result = {}
        for substancia in grouped["substancia"].unique():
            df_sub = grouped[grouped["substancia"] == substancia]
            top = df_sub.nlargest(1, "quantidade").iloc[0]
            result[substancia] = {
                "sexo": top["sexo"],
                "faixa_etaria": top["faixa_etaria"],
                "tipo_exame": top["purposeType"],
                "subtipo": top["purposeSubType"],
                "quantidade": int(top["quantidade"])
            }

        # Ordenar por quantidade total de positivos
        result = dict(sorted(result.items(), key=lambda x: x[1]["quantidade"], reverse=True))

        set_cached_data("demographic_profile", cache_key, result)
        return result

    except Exception as e:
        st.error(f"Erro ao buscar perfil demográfico: {e}")
        return {}


def get_demographic_table_data(laboratory_id: str = None, month: int = None, analysis_type: str = "all") -> pd.DataFrame:
    """
    Retorna tabela completa com dados demográficos por substância.
    """
    selected_start, selected_end = get_selected_period()
    cache_key = generate_cache_key("demographic_table", laboratory_id, month, analysis_type, selected_start.isoformat(), selected_end.isoformat())
    cached = get_cached_data("demographic_table", cache_key)
    if cached is not None:
        return cached

    try:
        client = get_mongo_client()
        db = client["ctox"]

        # Período
        if month:
            year = datetime.now().year
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year, 12, 31, 23, 59, 59)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        else:
            start_date = selected_start
            end_date = selected_end

        # Tipos de análise
        if analysis_type == "all":
            analysis_types = ["screening", "confirmatory", "confirmatoryTHC"]
        else:
            analysis_types = [analysis_type]

        # Buscar lotes do período
        lots_collection = db["lots"]
        lots = list(lots_collection.find(
            {
                "analysisType": {"$in": analysis_types},
                "createdAt": {"$gte": start_date, "$lte": end_date}
            },
            {"code": 1}
        ))

        if not lots:
            return pd.DataFrame()

        lot_codes = [lot.get('code') for lot in lots if lot.get('code')]

        # Pipeline de agregação
        pipeline = [
            {"$match": {"_lot": {"$in": lot_codes}}},
            {"$unwind": "$samples"},
            {"$match": {"samples.positive": True}},
            {"$unwind": "$samples._compound"},
            {"$match": {"samples._compound.positive": True}},
            {
                "$lookup": {
                    "from": "compounds",
                    "localField": "samples._compound._id",
                    "foreignField": "_id",
                    "as": "compoundInfo"
                }
            },
            {"$unwind": "$compoundInfo"},
            {
                "$lookup": {
                    "from": "chainofcustodies",
                    "localField": "samples._sample",
                    "foreignField": "sample.code",
                    "as": "chainInfo"
                }
            },
            {"$unwind": {"path": "$chainInfo", "preserveNullAndEmptyArrays": True}},
            {
                "$lookup": {
                    "from": "gatherings",
                    "localField": "chainInfo._id",
                    "foreignField": "_chainOfCustody",
                    "as": "gatheringInfo"
                }
            },
            {"$unwind": {"path": "$gatheringInfo", "preserveNullAndEmptyArrays": True}},
            {
                "$project": {
                    "substancia": "$compoundInfo.name",
                    "sexo": "$chainInfo.donor.gender",
                    "birthDate": "$chainInfo.donor.birthDate",
                    "purposeType": "$gatheringInfo.purpose.type",
                    "purposeSubType": "$gatheringInfo.purpose.subType"
                }
            }
        ]

        results_collection = db["results"]
        results = list(results_collection.aggregate(pipeline, allowDiskUse=True))

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Calcular faixa etária
        hoje = datetime.now()

        def calcular_faixa_etaria(birth_date):
            if pd.isna(birth_date) or birth_date is None:
                return "N/A"
            try:
                idade = (hoje - birth_date).days // 365
                if idade < 18:
                    return "< 18"
                elif idade <= 24:
                    return "18-24"
                elif idade <= 29:
                    return "25-29"
                elif idade <= 34:
                    return "30-34"
                elif idade <= 39:
                    return "35-39"
                elif idade <= 44:
                    return "40-44"
                elif idade <= 49:
                    return "45-49"
                elif idade <= 54:
                    return "50-54"
                elif idade <= 59:
                    return "55-59"
                else:
                    return "60+"
            except:
                return "N/A"

        df["faixa_etaria"] = df["birthDate"].apply(calcular_faixa_etaria)

        # Mapear valores
        sexo_map = {"m": "Masculino", "f": "Feminino"}
        df["sexo"] = df["sexo"].map(sexo_map).fillna("N/A")

        tipos_map = {
            "cnh": "CNH",
            "admissional": "Admissional",
            "periodico": "Periódico",
            "demissional": "Demissional"
        }
        df["purposeType"] = df["purposeType"].map(tipos_map).fillna(df["purposeType"]).fillna("N/A")

        subtipos_map = {
            "firstCnh": "Primeira Habilitação",
            "renovation": "Renovação",
            "categoryChange": "Mudança de Categoria"
        }
        df["purposeSubType"] = df["purposeSubType"].map(subtipos_map).fillna(df["purposeSubType"]).fillna("")

        # Agrupar e contar
        grouped = df.groupby(
            ["substancia", "sexo", "faixa_etaria", "purposeType", "purposeSubType"]
        ).size().reset_index(name="quantidade")

        # Renomear colunas
        grouped.columns = ["Substância", "Sexo", "Faixa Etária", "Tipo Exame", "Subtipo", "Quantidade"]

        # Ordenar
        grouped = grouped.sort_values(["Substância", "Quantidade"], ascending=[True, False])

        set_cached_data("demographic_table", cache_key, grouped)
        return grouped

    except Exception as e:
        st.error(f"Erro ao buscar dados demográficos: {e}")
        return pd.DataFrame()


def get_substance_statistics(laboratory_id: str = None, month: int = None, analysis_type: str = "all") -> dict:
    """
    Busca estatísticas de positividade por substância.
    Retorna dict {substância: {total, positivos, negativos, taxa}}
    """
    # Usar período selecionado na sidebar para cache key
    selected_start, selected_end = get_selected_period()
    cache_key = generate_cache_key("substance_stats", laboratory_id, month, analysis_type, selected_start.isoformat(), selected_end.isoformat())
    cached = get_cached_data("substance_stats", cache_key)
    if cached is not None:
        return cached

    try:
        lots_collection = get_collection("lots")
        results_collection = get_collection("results")

        # Período - usar período selecionado na sidebar
        if month:
            # Se mês específico foi passado, usar o ano atual
            year = datetime.now().year
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year, 12, 31, 23, 59, 59)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        else:
            start_date = selected_start
            end_date = selected_end

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

                # O array de compounds está em '_compound'
                compounds = sample.get('_compound', []) or sample.get('compounds', [])
                for compound in compounds:
                    # O ID do compound está em '_id' (ObjectId do MongoDB)
                    compound_id_raw = compound.get('_id')
                    if isinstance(compound_id_raw, ObjectId):
                        compound_id = str(compound_id_raw)
                    elif isinstance(compound_id_raw, dict) and '$oid' in compound_id_raw:
                        compound_id = compound_id_raw['$oid']
                    else:
                        compound_id = str(compound_id_raw) if compound_id_raw else ''
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

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        analysis_options = {"Todos": "all", "Triagem": "screening", "Confirmatório": "confirmatory", "Contra Prova": "againstProof"}
        selected_analysis = st.selectbox("Tipo de Análise", options=list(analysis_options.keys()), index=0, key="mapa_analysis")
        analysis_type = analysis_options[selected_analysis]

    with col_f2:
        finalidades = {
            "Todas": None,
            "CLT": "clt",
            "CLT + CNH": "cltCnh",
            "Concurso Público": "civilService",
            "Contra Prova": "againstProof",
        }
        selected_finalidade_name = st.selectbox("Finalidade", options=list(finalidades.keys()), index=0, key="mapa_finalidade")
        selected_purpose = finalidades[selected_finalidade_name]

    with col_f3:
        labs_by_cnpj = get_laboratories_by_cnpj()
        cnpj_options = ["Todos"] + sorted(labs_by_cnpj.keys())
        selected_cnpj = st.selectbox("CNPJ Laboratório", options=cnpj_options, index=0, key="mapa_cnpj")

        if selected_cnpj and selected_cnpj != "Todos":
            lab_info = labs_by_cnpj.get(selected_cnpj, {})
            selected_lab_id = lab_info.get("id")
        else:
            selected_lab_id = None

    st.markdown("---")

    geo_data = loading_single(
        get_geographic_data, "Carregando dados geográficos...",
        None, analysis_type, selected_lab_id, selected_purpose
    )

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

    # ========== TOP 5 SUBSTÂNCIAS COM MAIOR TAXA DE POSITIVIDADE ==========
    st.subheader("🧪 Top 5 Substâncias - Taxa de Positividade")

    # Buscar dados de positividade por substância
    start_date, end_date = get_selected_period()
    lab_ids = [selected_lab_id] if selected_lab_id else None
    substance_data = get_positivity_by_substance(lab_ids, start_date, end_date)

    if substance_data:
        # Converter para DataFrame e ordenar por taxa de positividade
        df_substances = pd.DataFrame([
            {
                "Substância": name,
                "Positivos": data["positivo"],
                "Total": data["total"],
                "Taxa (%)": data["percentual"]
            }
            for name, data in substance_data.items()
            if data["total"] > 0
        ])

        if not df_substances.empty:
            # Ordenar por taxa de positividade (maior para menor) e pegar top 5
            df_top5 = df_substances.sort_values("Taxa (%)", ascending=False).head(5)

            # Exibir como cards horizontais
            cols = st.columns(5)
            for i, (idx, row) in enumerate(df_top5.iterrows()):
                with cols[i]:
                    taxa = row["Taxa (%)"]
                    cor_taxa = "#FF6B6B" if taxa > 5 else "#FFD700" if taxa > 2 else "#00CED1"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                padding: 12px; border-radius: 10px; text-align: center; min-height: 110px;">
                        <p style="color: #888; margin: 0; font-size: 11px; font-weight: bold;">#{i+1}</p>
                        <p style="color: #FAFAFA; margin: 5px 0; font-size: 13px; font-weight: 600;">{row['Substância']}</p>
                        <h3 style="color: {cor_taxa}; margin: 5px 0; font-size: 22px;">{taxa:.2f}%</h3>
                        <p style="color: #888; margin: 0; font-size: 10px;">{row['Positivos']:,} / {row['Total']:,}</p>
                    </div>
                    """.replace(",", "."), unsafe_allow_html=True)
        else:
            st.info("Nenhum dado de substância disponível para o período selecionado.")
    else:
        st.info("Nenhum dado de substância disponível para o período selecionado.")

    st.markdown("---")

    # Mapa Choropleth do Brasil
    st.subheader("🗺️ Distribuição Geográfica - Taxa de Positividade por Estado")

    # Mapeamento de nomes completos para siglas
    nome_para_sigla = {
        "Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM",
        "Bahia": "BA", "Ceará": "CE", "Distrito Federal": "DF", "Espírito Santo": "ES",
        "Goiás": "GO", "Maranhão": "MA", "Mato Grosso": "MT", "Mato Grosso do Sul": "MS",
        "Minas Gerais": "MG", "Pará": "PA", "Paraíba": "PB", "Paraná": "PR",
        "Pernambuco": "PE", "Piauí": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
        "Rio Grande do Sul": "RS", "Rondônia": "RO", "Roraima": "RR", "Santa Catarina": "SC",
        "São Paulo": "SP", "Sergipe": "SE", "Tocantins": "TO"
    }

    sigla_para_nome = {v: k for k, v in nome_para_sigla.items()}

    if not df_estado.empty:
        # Preparar dados para o mapa
        mapa_data = []
        for _, row in df_estado.iterrows():
            estado = row["Estado"]
            # Converter nome completo para sigla se necessário
            sigla = nome_para_sigla.get(estado, estado)
            nome = sigla_para_nome.get(sigla, estado)
            mapa_data.append({
                "UF": sigla,
                "Estado": nome,
                "Total": row["Total"],
                "Positivos": row["Positivos"],
                "Taxa": row["Taxa (%)"]
            })

        if mapa_data:
            df_mapa = pd.DataFrame(mapa_data)

            # Usar choropleth com GeoJSON do Brasil
            geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"

            try:
                import requests
                response = requests.get(geojson_url, timeout=10)
                brazil_geojson = response.json()

                # Criar mapa choropleth
                fig_mapa = px.choropleth(
                    df_mapa,
                    geojson=brazil_geojson,
                    locations="UF",
                    featureidkey="properties.sigla",
                    color="Taxa",
                    color_continuous_scale=[
                        [0, "#00CED1"],      # Verde/Azul - baixa taxa
                        [0.5, "#FFD700"],    # Amarelo - média taxa
                        [1, "#FF4444"]       # Vermelho - alta taxa
                    ],
                    hover_name="Estado",
                    hover_data={
                        "UF": False,
                        "Total": ":,",
                        "Positivos": ":,",
                        "Taxa": ":.2f"
                    },
                    labels={"Taxa": "Taxa (%)", "Total": "Total Amostras", "Positivos": "Positivos"}
                )

                fig_mapa.update_geos(
                    fitbounds="locations",
                    visible=False,
                    bgcolor="rgba(0,0,0,0)"
                )

                fig_mapa.update_layout(
                    height=550,
                    margin=dict(t=10, b=10, l=10, r=10),
                    coloraxis_colorbar=dict(
                        title="Taxa (%)",
                        ticksuffix="%",
                        len=0.7,
                        thickness=15,
                        x=0.95
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    geo=dict(
                        bgcolor="rgba(0,0,0,0)",
                        showframe=False
                    )
                )

                # Adicionar anotações com siglas dos estados
                fig_mapa.update_traces(
                    marker_line_color="white",
                    marker_line_width=1
                )

                st.plotly_chart(fig_mapa, use_container_width=True, key="chart_mapa_brasil")

                # Legenda explicativa
                st.caption("🟢 Verde/Azul = Baixa taxa de positividade | 🟡 Amarelo = Taxa média | 🔴 Vermelho = Alta taxa de positividade")

            except Exception as e:
                st.warning(f"Não foi possível carregar o mapa: {e}")
                st.info("Exibindo dados em formato de tabela.")
        else:
            st.info("Nenhum dado de estado encontrado para exibir no mapa")
    else:
        st.info("Nenhum dado de estado encontrado")

    st.markdown("---")

    # Gráficos
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("📊 Amostras por Estado")

        if not df_estado.empty:
            df_estado["Texto"] = df_estado.apply(
                lambda row: f"{row['Total']:,} ({row['Taxa (%)']:.1f}%)".replace(",", "."), axis=1
            )

            df_top_estado = df_estado.head(10).copy()
            max_estado = df_top_estado["Total"].max()

            fig_estado = px.bar(
                df_top_estado,
                y="Estado",
                x="Total",
                orientation="h",
                text="Texto",
                color="Taxa (%)",
                color_continuous_scale=["#00CED1", "#FF6B6B"]
            )

            fig_estado.update_traces(
                textposition="outside",
                textfont_size=10,
                cliponaxis=False
            )
            fig_estado.update_layout(
                height=400,
                margin=dict(t=30, b=30, l=80, r=150),
                xaxis_title="",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                xaxis=dict(range=[0, max_estado * 1.35], showticklabels=False, showgrid=False),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_estado, use_container_width=True, key="chart_mapa_estado")

    with col_g2:
        st.subheader("🏙️ Top 5 Cidades")

        if not df_cidade.empty:
            df_cidade["Texto"] = df_cidade.apply(
                lambda row: f"{row['Total']:,} ({row['Taxa (%)']:.1f}%)".replace(",", "."), axis=1
            )

            df_top_cidade = df_cidade.head(5).copy()
            max_cidade = df_top_cidade["Total"].max()

            fig_cidade = px.bar(
                df_top_cidade,
                y="Cidade",
                x="Total",
                orientation="h",
                text="Texto",
                color="Taxa (%)",
                color_continuous_scale=["#00CED1", "#FF6B6B"]
            )

            fig_cidade.update_traces(
                textposition="outside",
                textfont_size=10,
                cliponaxis=False
            )
            fig_cidade.update_layout(
                height=400,
                margin=dict(t=30, b=30, l=150, r=150),
                xaxis_title="",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                xaxis=dict(range=[0, max_cidade * 1.35], showticklabels=False, showgrid=False),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_cidade, use_container_width=True, key="chart_mapa_cidade")

    st.markdown("---")

    # Tabelas
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.subheader("📋 Dados por Estado")
        st.dataframe(df_estado[["Estado", "Total", "Positivos", "Taxa (%)"]], use_container_width=True, hide_index=True, height=300)

    with col_t2:
        st.subheader("📋 Dados por Cidade")
        st.dataframe(df_cidade[["Cidade", "Estado", "Total", "Positivos", "Taxa (%)"]], use_container_width=True, hide_index=True, height=300)


def get_geographic_data(month: int = None, analysis_type: str = "all", laboratory_id: str = None, purpose_type: str = None) -> dict:
    """
    Busca dados geográficos por estado e cidade.
    """
    # Usar período selecionado na sidebar para cache key
    selected_start, selected_end = get_selected_period()
    cache_key = generate_cache_key("geographic", month, analysis_type, laboratory_id, purpose_type, selected_start.isoformat(), selected_end.isoformat())
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

        # Período - usar período selecionado na sidebar
        if month:
            # Se mês específico foi passado, usar o ano atual
            year = datetime.now().year
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year, 12, 31, 23, 59, 59)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        else:
            start_date = selected_start
            end_date = selected_end

        # Tipos de análise
        if analysis_type == "all":
            analysis_types = ["screening", "confirmatory", "confirmatoryTHC"]
        else:
            analysis_types = [analysis_type]

        lots_collection = get_collection("lots")
        results_collection = get_collection("results")
        gatherings_collection = get_collection("gatherings")

        # Query para gatherings
        gatherings_query = {"createdAt": {"$gte": start_date, "$lte": end_date}}
        if laboratory_id:
            gatherings_query["_laboratory"] = ObjectId(laboratory_id)
        if purpose_type:
            gatherings_query["purpose.type"] = purpose_type

        # Buscar gatherings para mapear chainOfCustody -> laboratory
        gatherings = list(gatherings_collection.find(
            gatherings_query,
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
    Página 4 - Análise Temporal: linha de taxa + barras empilhadas com comparativo MoM
    """
    st.title("📈 Análise Temporal")

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        labs_by_cnpj = get_laboratories_by_cnpj()
        cnpj_options = sorted(labs_by_cnpj.keys())

        selected_cnpjs = st.multiselect(
            "CNPJ Laboratórios",
            options=cnpj_options,
            default=[],
            placeholder="Todos os laboratórios",
            key="temp_lab"
        )

        if selected_cnpjs:
            selected_lab_ids = [labs_by_cnpj[cnpj]["id"] for cnpj in selected_cnpjs if cnpj in labs_by_cnpj]
            lab_names = [labs_by_cnpj[cnpj]["name"] for cnpj in selected_cnpjs if cnpj in labs_by_cnpj]
            if len(lab_names) <= 2:
                st.caption(f"🏢 {', '.join(lab_names)}")
            else:
                st.caption(f"🏢 {len(lab_names)} laboratórios selecionados")
        else:
            selected_lab_ids = None

    with col_f2:
        analysis_options = {"Triagem": "screening", "Confirmatório": "confirmatory", "Todos": "all"}
        selected_analysis = st.selectbox("Tipo de Análise", options=list(analysis_options.keys()), index=0, key="temp_analysis")
        analysis_type = analysis_options[selected_analysis]

    with col_f3:
        view_options = {"Mensal": "monthly", "Semanal": "weekly"}
        selected_view = st.selectbox("Visualização", options=list(view_options.keys()), index=0, key="temp_view")
        view_type = view_options[selected_view]

    st.markdown("---")

    if view_type == "monthly":
        temporal_data = loading_single(
            get_monthly_positivity_data, "Carregando dados temporais...",
            laboratory_ids=selected_lab_ids, analysis_type=analysis_type
        )
    else:
        temporal_data = loading_single(
            get_weekly_data, "Carregando dados temporais...",
            selected_lab_ids, analysis_type
        )

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

    # Ordenar por período
    df_temp = df_temp.sort_values("Período").reset_index(drop=True)

    # Calcular métricas
    total_geral = df_temp["Total"].sum()
    total_positivos = df_temp["Positivos"].sum()
    taxa_media = df_temp["Taxa (%)"].mean() if not df_temp.empty else 0

    # Calcular comparativo MoM (período atual vs anterior)
    if len(df_temp) >= 2:
        periodo_atual = df_temp.iloc[-1]
        periodo_anterior = df_temp.iloc[-2]

        taxa_atual = periodo_atual["Taxa (%)"]
        taxa_anterior = periodo_anterior["Taxa (%)"]
        variacao_taxa = taxa_atual - taxa_anterior

        total_atual = periodo_atual["Total"]
        total_anterior = periodo_anterior["Total"]
        variacao_total = ((total_atual - total_anterior) / total_anterior * 100) if total_anterior > 0 else 0

        nome_atual = periodo_atual["Período"]
        nome_anterior = periodo_anterior["Período"]
    else:
        taxa_atual = df_temp["Taxa (%)"].iloc[-1] if not df_temp.empty else 0
        variacao_taxa = 0
        total_atual = df_temp["Total"].iloc[-1] if not df_temp.empty else 0
        variacao_total = 0
        nome_atual = df_temp["Período"].iloc[-1] if not df_temp.empty else "N/A"
        nome_anterior = "N/A"

    # ========== KPIs COM COMPARATIVO MoM ==========
    st.markdown("### 📊 Indicadores com Comparativo")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    padding: 15px; border-radius: 10px; text-align: center; min-height: 95px; display: flex; flex-direction: column; justify-content: center;">
            <p style="color: #888; margin: 0; font-size: 12px;">Total no Período</p>
            <h2 style="color: #00CED1; margin: 5px 0; font-size: 24px;">{total_geral:,}</h2>
            <p style="color: transparent; margin: 0; font-size: 11px;">&nbsp;</p>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    padding: 15px; border-radius: 10px; text-align: center; min-height: 95px; display: flex; flex-direction: column; justify-content: center;">
            <p style="color: #888; margin: 0; font-size: 12px;">Taxa Média</p>
            <h2 style="color: {'#FF6B6B' if taxa_media > 5 else '#FFD700' if taxa_media > 2 else '#00CED1'}; margin: 5px 0; font-size: 24px;">{taxa_media:.2f}%</h2>
            <p style="color: transparent; margin: 0; font-size: 11px;">&nbsp;</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        cor_variacao = "#FF6B6B" if variacao_taxa > 0 else "#00CED1" if variacao_taxa < 0 else "#888"
        seta = "↑" if variacao_taxa > 0 else "↓" if variacao_taxa < 0 else "→"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    padding: 15px; border-radius: 10px; text-align: center; min-height: 95px; display: flex; flex-direction: column; justify-content: center;">
            <p style="color: #888; margin: 0; font-size: 12px;">Taxa Último Período</p>
            <h2 style="color: {cor_variacao}; margin: 5px 0; font-size: 24px;">{taxa_atual:.2f}%</h2>
            <p style="color: {cor_variacao}; margin: 0; font-size: 11px;">{seta} {abs(variacao_taxa):.2f}pp vs anterior</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        cor_var_total = "#00CED1" if variacao_total > 0 else "#FF6B6B" if variacao_total < 0 else "#888"
        seta_total = "↑" if variacao_total > 0 else "↓" if variacao_total < 0 else "→"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    padding: 15px; border-radius: 10px; text-align: center; min-height: 95px; display: flex; flex-direction: column; justify-content: center;">
            <p style="color: #888; margin: 0; font-size: 12px;">Volume Último Período</p>
            <h2 style="color: #00CED1; margin: 5px 0; font-size: 24px;">{int(total_atual):,}</h2>
            <p style="color: {cor_var_total}; margin: 0; font-size: 11px;">{seta_total} {abs(variacao_total):.1f}% vs anterior</p>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    st.markdown("---")

    # ========== GRÁFICO DE LINHA - TAXA DE POSITIVIDADE ==========
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
        marker=dict(size=10, color="#FF6B6B")
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

    # Adicionar anotações com valores
    fig_linha.update_traces(
        text=df_temp["Taxa (%)"].apply(lambda x: f"{x:.1f}%"),
        textposition="top center",
        textfont=dict(size=10),
        selector=dict(mode='lines+markers')
    )

    fig_linha.update_layout(
        height=400,
        margin=dict(t=40, b=50, l=50, r=50),
        xaxis_title="",
        yaxis_title="Taxa de Positividade (%)",
        xaxis_tickangle=-45,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_linha, use_container_width=True, key="chart_temporal_linha")

    st.markdown("---")

    # ========== GRÁFICO DE BARRAS EMPILHADAS - POSITIVOS VS NEGATIVOS ==========
    st.subheader("📊 Volume de Amostras: Positivos vs Negativos")

    fig_stack = go.Figure()

    # Barras de Negativos
    fig_stack.add_trace(go.Bar(
        name='Negativos',
        x=df_temp["Período"],
        y=df_temp["Negativos"],
        marker_color='#00CED1',
        text=df_temp["Negativos"].apply(lambda x: f"{x:,}".replace(",", ".")),
        textposition='inside',
        textfont=dict(size=10, color='white')
    ))

    # Barras de Positivos
    fig_stack.add_trace(go.Bar(
        name='Positivos',
        x=df_temp["Período"],
        y=df_temp["Positivos"],
        marker_color='#FF6B6B',
        text=df_temp["Positivos"].apply(lambda x: f"{x:,}".replace(",", ".")),
        textposition='inside',
        textfont=dict(size=10, color='white')
    ))

    fig_stack.update_layout(
        barmode='stack',
        height=400,
        margin=dict(t=30, b=50, l=50, r=50),
        xaxis_title="",
        yaxis_title="Quantidade de Amostras",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_stack, use_container_width=True, key="chart_temporal_stack")


def get_weekly_data(laboratory_ids: list = None, analysis_type: str = "screening") -> dict:
    """
    Busca dados semanais de positividade.
    """
    # Usar período selecionado na sidebar
    selected_start, selected_end = get_selected_period()
    cache_key = generate_cache_key("weekly_data", laboratory_ids, analysis_type, selected_start.isoformat(), selected_end.isoformat())
    cached = get_cached_data("weekly_data", cache_key)
    if cached is not None:
        return cached

    try:
        lots_collection = get_collection("lots")
        results_collection = get_collection("results")

        # Usar período selecionado na sidebar
        start_date = selected_start
        end_date = selected_end

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
    Página 5 - Tabela Detalhada com Exportação e Paginação Server-Side
    """
    st.title("📋 Tabela Detalhada")

    # Inicializar estado de paginação
    if 'tabela_page' not in st.session_state:
        st.session_state.tabela_page = 1
    if 'tabela_page_size' not in st.session_state:
        st.session_state.tabela_page_size = 100

    # Filtros
    st.markdown("### 🔍 Filtros")

    # Primeira linha de filtros
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

    # Segunda linha: Filtros em cascata de Finalidade e Tipo de Substância
    col_tipo, col_sub, col_tipo_subst = st.columns(3)

    with col_tipo:
        # Tipo de Exame (CNH, CLT, CLT + CNH)
        tipos_exame_opcoes = ["Todos", "CNH", "CLT", "CLT + CNH"]
        filtro_tipo_exame = st.selectbox("Tipo de Exame", tipos_exame_opcoes, key="det_tipo_exame")

    with col_sub:
        # Subfinalidades dependem do tipo selecionado
        subfinalidades_por_tipo = {
            "CNH": ["Todas", "Primeira Habilitação", "Renovação", "Mudança de Categoria", "Contra Prova"],
            "CLT": ["Todas", "Admissional", "Periódico", "Demissional", "Mudança de Função", "Retorno ao Trabalho", "Contra Prova"],
            "CLT + CNH": ["Todas", "Admissional", "Periódico", "Demissional", "Primeira Habilitação", "Renovação", "Mudança de Categoria", "Mudança de Função", "Retorno ao Trabalho", "Contra Prova"],
            "Todos": ["Todas", "Admissional", "Periódico", "Demissional", "Primeira Habilitação", "Renovação", "Mudança de Categoria", "Mudança de Função", "Retorno ao Trabalho", "Contra Prova"]
        }

        subfinalidades_disponiveis = subfinalidades_por_tipo.get(filtro_tipo_exame, subfinalidades_por_tipo["Todos"])
        filtro_subfinalidade = st.selectbox("Subfinalidade", subfinalidades_disponiveis, key="det_subfinalidade")

    with col_tipo_subst:
        # Tipo de Substância (Ilícito, Medicamento, Psicoativo legal)
        tipos_substancia_opcoes = ["Todos", "Ilícito", "Medicamento"]
        filtro_tipo_substancia = st.selectbox("Tipo de Substância", tipos_substancia_opcoes, key="det_tipo_substancia")

    st.markdown("---")

    # Preparar filtros para paginação server-side
    filters = {
        'tipo_exame': filtro_tipo_exame if filtro_tipo_exame != 'Todos' else None,
        'subfinalidade': filtro_subfinalidade if filtro_subfinalidade != 'Todas' else None,
        'resultado': filtro_positivas if filtro_positivas != 'Todos' else None
    }

    # Resetar página se filtros mudaram
    filter_key = f"{filtro_tipo_exame}_{filtro_subfinalidade}_{filtro_positivas}_{filtro_tipo_substancia}"
    if 'last_filter_key' not in st.session_state or st.session_state.last_filter_key != filter_key:
        st.session_state.tabela_page = 1
        st.session_state.last_filter_key = filter_key

    # Carregar dados com paginação
    with st.spinner("Carregando dados..."):
        df, total_count, total_pages = get_substance_data_paginated(
            page=st.session_state.tabela_page,
            page_size=st.session_state.tabela_page_size,
            filters=filters
        )

    if df.empty and total_count == 0:
        st.warning("⚠️ Nenhum dado encontrado")
        return

    # Colunas de substâncias (excluindo colunas de metadados)
    substance_cols = [col for col in df.columns if col not in ['Data', 'Lote', 'Tipo de Lote', 'Tipo Exame', 'Subfinalidade', 'Amostra']]

    # Filtrar colunas de substâncias por tipo selecionado
    if filtro_tipo_substancia != 'Todos':
        # Obter nomes das substâncias do tipo selecionado
        substancias_do_tipo = get_compound_names_by_type(filtro_tipo_substancia)
        # Filtrar apenas as colunas que correspondem ao tipo
        substance_cols = [col for col in substance_cols if col in substancias_do_tipo]

    # Aplicar filtros locais (lote, amostra, substância específica)
    df_filtrado = df.copy()

    if filtro_lote:
        df_filtrado = df_filtrado[df_filtrado['Lote'].str.contains(filtro_lote, case=False, na=False)]

    if filtro_amostra:
        df_filtrado = df_filtrado[df_filtrado['Amostra'].astype(str).str.contains(filtro_amostra, case=False, na=False)]

    if filtro_substancia != 'Todas' and filtro_substancia in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado[filtro_substancia] == 'Positivo']

    # ============================================
    # COMPARATIVO DE CONCENTRAÇÃO (quando filtrar por amostra específica)
    # ============================================
    if filtro_amostra and len(filtro_amostra.strip()) > 0:
        # Verificar se existe a amostra nos resultados
        if len(df_filtrado) >= 1:
            amostra_code = str(df_filtrado.iloc[0]['Amostra'])
        else:
            # Usar o filtro digitado como código da amostra
            amostra_code = filtro_amostra.strip()

        st.markdown(f"### 🧪 Comparativo de Quantidades - Amostra {amostra_code}")

        with st.spinner("Carregando dados de concentração..."):
            # Buscar concentrações da amostra
            concentration_data = get_sample_concentration_data(amostra_code)
            # Buscar médias gerais do período
            avg_data = get_average_concentrations()
            # Buscar laboratório da amostra
            lab_info = get_sample_laboratory(amostra_code)
            # Buscar médias do laboratório específico
            avg_data_lab = {}
            if lab_info.get("lab_id"):
                avg_data_lab = get_average_concentrations_by_lab(lab_info["lab_id"])

        # Mostrar informação do laboratório
        if lab_info.get("lab_name"):
            lab_location = f"{lab_info.get('lab_city', '')}/{lab_info.get('lab_state', '')}" if lab_info.get('lab_city') else ""
            lab_has_avg = len(avg_data_lab) > 0
            lab_avg_status = "✅ Média disponível" if lab_has_avg else "⚠️ Sem dados de média no período"
            st.info(f"🏢 **Laboratório:** {lab_info['lab_name']} {f'({lab_location})' if lab_location else ''} | {lab_avg_status}")
        else:
            st.warning(f"⚠️ Não foi possível identificar o laboratório da amostra {amostra_code}")

        if concentration_data:
            # Filtrar apenas substâncias positivas na amostra
            positive_compounds = {k: v for k, v in concentration_data.items() if v.get("positive") == True}

            if positive_compounds:
                st.markdown("#### Substâncias Positivas Encontradas")

                for compound_name, data in positive_compounds.items():
                    conc_amostra = data.get("concentration", 0) or 0
                    # Média geral
                    avg_info = avg_data.get(compound_name, {})
                    conc_media_geral = avg_info.get("avg_concentration", 0) or 0
                    # Média do laboratório
                    avg_info_lab = avg_data_lab.get(compound_name, {})
                    conc_media_lab = avg_info_lab.get("avg_concentration", 0) or 0
                    # Máximo para escala do gráfico
                    conc_max = max(
                        avg_info.get("max_concentration", 0) or 0,
                        avg_info_lab.get("max_concentration", 0) or 0
                    )

                    # Card com informações da droga
                    col_info, col_chart = st.columns([1, 2])

                    with col_info:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                    padding: 15px; border-radius: 10px; margin-bottom: 10px;
                                    border: 1px solid #FFD700;">
                            <p style="color: #FFD700; margin: 0; font-size: 12px; font-weight: bold;">Droga:</p>
                            <p style="color: #E8E8E8; margin: 5px 0; font-size: 16px;">{compound_name.upper()}</p>
                            <p style="color: #888; margin: 10px 0 0 0; font-size: 12px;">Quant. Encontrada:</p>
                            <p style="color: #00CED1; margin: 0; font-size: 18px; font-weight: bold;">{conc_amostra:.4f} ng/mg</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_chart:
                        # Gráfico de barras comparativo com 3 barras
                        has_media_geral = conc_media_geral > 0
                        has_media_lab = conc_media_lab > 0

                        if has_media_geral or has_media_lab:
                            fig_comp = go.Figure()

                            # Barra da amostra
                            fig_comp.add_trace(go.Bar(
                                name='Encontrado nesta amostra',
                                y=[''],
                                x=[conc_amostra],
                                orientation='h',
                                marker_color='#FFD700',
                                text=f'{conc_amostra:.4f}',
                                textposition='outside',
                                textfont=dict(size=10)
                            ))

                            # Barra da média do laboratório
                            if has_media_lab:
                                fig_comp.add_trace(go.Bar(
                                    name='Média do laboratório',
                                    y=[''],
                                    x=[conc_media_lab],
                                    orientation='h',
                                    marker_color='#00CED1',
                                    text=f'{conc_media_lab:.4f}',
                                    textposition='outside',
                                    textfont=dict(size=10)
                                ))

                            # Barra da média geral
                            if has_media_geral:
                                fig_comp.add_trace(go.Bar(
                                    name='Média geral',
                                    y=[''],
                                    x=[conc_media_geral],
                                    orientation='h',
                                    marker_color='#4169E1',
                                    text=f'{conc_media_geral:.4f}',
                                    textposition='outside',
                                    textfont=dict(size=10)
                                ))

                            max_val = max(conc_amostra, conc_media_geral, conc_media_lab, conc_max) * 1.3
                            fig_comp.update_layout(
                                height=140,
                                margin=dict(t=10, b=10, l=10, r=80),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                barmode='group',
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9)),
                                xaxis=dict(range=[0, max_val], showticklabels=False, showgrid=False),
                                yaxis=dict(showticklabels=False)
                            )

                            st.plotly_chart(fig_comp, use_container_width=True, key=f"comp_det_{compound_name}")
                        else:
                            st.info(f"Não há dados de média para {compound_name} no período selecionado.")

                        # Indicador de padrão de consumo (prioriza média do laboratório)
                        media_referencia = conc_media_lab if conc_media_lab > 0 else conc_media_geral
                        tipo_media = "do laboratório" if conc_media_lab > 0 else "geral"

                        if media_referencia > 0:
                            if conc_amostra > media_referencia * 1.5:
                                padrao_texto = f"A quantidade encontrada está **ACIMA** do padrão médio {tipo_media} de consumo."
                                padrao_cor = "#FF6B6B"
                            elif conc_amostra < media_referencia * 0.5:
                                padrao_texto = f"A quantidade encontrada está **ABAIXO** do padrão médio {tipo_media} de consumo."
                                padrao_cor = "#4CAF50"
                            else:
                                padrao_texto = f"A quantidade encontrada está **DENTRO** do padrão médio {tipo_media} de consumo."
                                padrao_cor = "#FFD700"

                            st.markdown(f"""
                            <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;
                                        border-left: 4px solid {padrao_cor}; margin-bottom: 15px;">
                                <p style="color: #E8E8E8; margin: 0; font-size: 13px;">{padrao_texto}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("---")
            else:
                st.info("Esta amostra não possui substâncias positivas para exibir o comparativo de concentração.")
        else:
            st.warning(f"Não foi possível encontrar dados de concentração para a amostra '{amostra_code}'. Verifique se o código está correto.")

    # Estatísticas (baseadas no total, não apenas na página atual)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de Amostras", f"{total_count:,}".replace(",", "."))

    with col2:
        total_lotes = df_filtrado['Lote'].nunique() if not df_filtrado.empty else 0
        st.metric("Lotes (página)", f"{total_lotes:,}".replace(",", "."))

    with col3:
        if substance_cols and not df_filtrado.empty:
            amostras_positivas = df_filtrado[df_filtrado[substance_cols].apply(lambda row: (row == 'Positivo').any(), axis=1)].shape[0]
        else:
            amostras_positivas = 0
        st.metric("Positivas (página)", f"{amostras_positivas:,}".replace(",", "."))

    # Resumo de substâncias positivas na página
    if substance_cols and not df_filtrado.empty:
        substancias_positivas = {}
        for col in substance_cols:
            count = (df_filtrado[col] == 'Positivo').sum()
            if count > 0:
                substancias_positivas[col] = count

        if substancias_positivas:
            # Ordenar por quantidade (maior para menor)
            sorted_subs = sorted(substancias_positivas.items(), key=lambda x: x[1], reverse=True)
            # Criar badges para cada substância
            badges_html = " ".join([
                f'<span style="display: inline-block; margin: 2px 4px; padding: 4px 10px; background: #dc3545; color: white; border-radius: 12px; font-size: 0.85rem; font-weight: 500;">{sub}: {count}</span>'
                for sub, count in sorted_subs
            ])
            st.markdown(f"""
            <div style="margin-top: 10px;">
                <span style="color: #FAFAFA; font-size: 0.9rem; margin-right: 8px;">🔴 Substâncias positivas na página:</span>
                {badges_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="margin-top: 10px;">
                <span style="color: #28a745; font-size: 0.9rem;">✅ Nenhuma substância positiva nesta página</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Função de estilização para a tabela
    def style_table(df):
        """
        Aplica estilização à tabela:
        - Linhas com pelo menos uma substância positiva: fundo vermelho, texto preto
        - Células de substâncias positivas: texto branco, negrito
        """
        sub_cols = [col for col in df.columns if col not in ['Data', 'Lote', 'Tipo de Lote', 'Tipo Exame', 'Subfinalidade', 'Amostra']]
        styles = pd.DataFrame('', index=df.index, columns=df.columns)

        for idx in df.index:
            row = df.loc[idx]
            has_positive = any(row[col] == 'Positivo' for col in sub_cols if col in row.index)

            if has_positive:
                for col in df.columns:
                    styles.loc[idx, col] = 'background-color: #ffcccc; color: black;'
                for col in sub_cols:
                    if col in row.index and row[col] == 'Positivo':
                        styles.loc[idx, col] = 'background-color: #dc3545; color: white; font-weight: bold;'

        return styles

    # Filtrar colunas do dataframe para exibição quando filtro de tipo de substância estiver ativo
    if filtro_tipo_substancia != 'Todos':
        # Definir colunas de metadados que sempre aparecem
        meta_cols = ['Data', 'Lote', 'Tipo de Lote', 'Tipo Exame', 'Subfinalidade', 'Amostra']
        # Filtrar para mostrar apenas as colunas de metadados + substâncias do tipo selecionado
        cols_to_show = [col for col in df_filtrado.columns if col in meta_cols or col in substance_cols]
        df_exibicao = df_filtrado[cols_to_show]
    else:
        df_exibicao = df_filtrado

    # Tabela com estilo
    if not df_exibicao.empty:
        styled_df = df_exibicao.style.apply(lambda _: style_table(df_exibicao), axis=None)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)
    else:
        st.dataframe(df_exibicao, use_container_width=True, hide_index=True, height=500)

    # Controles de paginação (abaixo da tabela)
    st.markdown("---")
    col_pag1, col_pag2, col_pag3, col_pag4, col_pag5 = st.columns([1, 1, 2, 1, 1])

    with col_pag1:
        if st.button("⏮️ Primeira", disabled=st.session_state.tabela_page <= 1, use_container_width=True):
            st.session_state.tabela_page = 1
            st.rerun()

    with col_pag2:
        if st.button("◀️ Anterior", disabled=st.session_state.tabela_page <= 1, use_container_width=True):
            st.session_state.tabela_page -= 1
            st.rerun()

    with col_pag3:
        st.markdown(f"""
        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 5px;">
            <span style="color: #FAFAFA;">Página <strong>{st.session_state.tabela_page}</strong> de <strong>{total_pages}</strong></span>
        </div>
        """, unsafe_allow_html=True)

    with col_pag4:
        if st.button("Próxima ▶️", disabled=st.session_state.tabela_page >= total_pages, use_container_width=True):
            st.session_state.tabela_page += 1
            st.rerun()

    with col_pag5:
        if st.button("Última ⏭️", disabled=st.session_state.tabela_page >= total_pages, use_container_width=True):
            st.session_state.tabela_page = total_pages
            st.rerun()

    # Seletor de itens por página
    col_size, col_goto = st.columns([1, 1])
    with col_size:
        new_page_size = st.selectbox(
            "Itens por página",
            [50, 100, 200, 500],
            index=[50, 100, 200, 500].index(st.session_state.tabela_page_size),
            key="page_size_select"
        )
        if new_page_size != st.session_state.tabela_page_size:
            st.session_state.tabela_page_size = new_page_size
            st.session_state.tabela_page = 1
            st.rerun()

    with col_goto:
        goto_page = st.number_input(
            "Ir para página",
            min_value=1,
            max_value=max(1, total_pages),
            value=st.session_state.tabela_page,
            key="goto_page_input"
        )
        if goto_page != st.session_state.tabela_page:
            st.session_state.tabela_page = goto_page
            st.rerun()

    st.markdown("---")

    # Exportação
    st.markdown("### ⬇️ Exportar Dados")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        # Excel da página atual
        output_pagina = io.BytesIO()
        with pd.ExcelWriter(output_pagina, engine='openpyxl') as writer:
            df_exibicao.to_excel(writer, index=False, sheet_name='Amostras')
        excel_pagina = output_pagina.getvalue()

        st.download_button(
            "📄 Excel (página atual)",
            excel_pagina,
            "amostras_pagina_atual.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Exporta apenas os registros exibidos na página atual"
        )

    with col_exp2:
        # Botão para exportar todos os dados com filtros aplicados
        if st.button("📊 Excel (filtros aplicados)", use_container_width=True, help="Exporta todos os resultados do período com os filtros atuais"):
            with st.spinner("Carregando todos os dados com filtros aplicados..."):
                df_completo = get_substance_data()
                if not df_completo.empty:
                    # Aplicar mesmos filtros
                    if filtro_tipo_exame != 'Todos':
                        df_completo = df_completo[df_completo['Tipo Exame'] == filtro_tipo_exame]
                    if filtro_subfinalidade != 'Todas':
                        df_completo = df_completo[df_completo['Subfinalidade'] == filtro_subfinalidade]
                    if filtro_positivas == 'Positivo':
                        sub_cols_exp = [col for col in df_completo.columns if col not in ['Data', 'Lote', 'Tipo de Lote', 'Tipo Exame', 'Subfinalidade', 'Amostra']]
                        df_completo = df_completo[df_completo[sub_cols_exp].apply(lambda row: (row == 'Positivo').any(), axis=1)]
                    elif filtro_positivas == 'Negativo':
                        sub_cols_exp = [col for col in df_completo.columns if col not in ['Data', 'Lote', 'Tipo de Lote', 'Tipo Exame', 'Subfinalidade', 'Amostra']]
                        df_completo = df_completo[df_completo[sub_cols_exp].apply(lambda row: (row == 'Negativo').all(), axis=1)]

                    # Aplicar filtro de tipo de substância nas colunas
                    if filtro_tipo_substancia != 'Todos':
                        meta_cols_exp = ['Data', 'Lote', 'Tipo de Lote', 'Tipo Exame', 'Subfinalidade', 'Amostra']
                        substancias_filtradas = get_compound_names_by_type(filtro_tipo_substancia)
                        cols_exp = [col for col in df_completo.columns if col in meta_cols_exp or col in substancias_filtradas]
                        df_completo = df_completo[cols_exp]

                    # Gerar Excel
                    output_filtrado = io.BytesIO()
                    with pd.ExcelWriter(output_filtrado, engine='openpyxl') as writer:
                        df_completo.to_excel(writer, index=False, sheet_name='Amostras')
                    excel_filtrado = output_filtrado.getvalue()

                    st.download_button(
                        "⬇️ Baixar Excel (filtros aplicados)",
                        excel_filtrado,
                        "amostras_filtradas.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_filtrado"
                    )
                    st.success(f"Pronto! {len(df_completo):,} registros disponíveis para download.".replace(",", "."))
                else:
                    st.warning("Nenhum dado para exportar.")

    with col_exp3:
        # Botão para exportar todos os dados de 2026 em diante
        if st.button("📥 Excel (2026)", use_container_width=True, help="Exporta todos os registros a partir de 01/01/2026"):
            with st.spinner("Carregando todos os dados de 2026 em diante..."):
                # Buscar dados desde 01/01/2026
                data_inicio_2026 = datetime(2026, 1, 1, 0, 0, 0)
                data_fim = datetime.now().replace(hour=23, minute=59, second=59)

                df_2026 = get_substance_data_full(data_inicio_2026, data_fim)
                if not df_2026.empty:
                    # Gerar Excel
                    output_2026 = io.BytesIO()
                    with pd.ExcelWriter(output_2026, engine='openpyxl') as writer:
                        df_2026.to_excel(writer, index=False, sheet_name='Amostras_2026')
                    excel_2026 = output_2026.getvalue()

                    st.download_button(
                        "⬇️ Baixar Excel (2026+)",
                        excel_2026,
                        "amostras_2026_em_diante.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_2026"
                    )
                    st.success(f"Pronto! {len(df_2026):,} registros disponíveis para download.".replace(",", "."))
                else:
                    st.warning("Nenhum dado encontrado para 2026 em diante.")

    st.caption(f"Exibindo {len(df_exibicao):,} de {total_count:,} registros".replace(",", "."))


def get_shewhart_data() -> dict:
    """
    Calcula dados para o gráfico de controle estatístico (Shewhart).
    Retorna dados mensais com limites de controle ±2σ e ±3σ.
    """
    start_date, end_date = get_selected_period()
    cache_key = generate_cache_key("shewhart", start_date.isoformat(), end_date.isoformat())
    cached = get_cached_data("shewhart_data", cache_key)
    if cached is not None:
        return cached

    try:
        # Buscar dados mensais passando o período selecionado
        monthly_data = get_monthly_positivity_data(
            analysis_type="screening",
            start_date_filter=start_date,
            end_date_filter=end_date
        )

        if not monthly_data:
            return {}

        # Calcular taxas mensais
        meses = []
        taxas = []
        totais = []

        for mes, data in monthly_data.items():
            total = data.get("positivo", 0) + data.get("negativo", 0)
            if total > 0:
                taxa = data.get("taxa", 0)
                meses.append(mes)
                taxas.append(taxa)
                totais.append(total)

        if len(taxas) < 3:
            return {}

        # Calcular estatísticas
        media = sum(taxas) / len(taxas)
        variancia = sum((t - media) ** 2 for t in taxas) / len(taxas)
        desvio = variancia ** 0.5

        # Limites de controle
        ucl_2sigma = media + 2 * desvio  # Upper Control Limit 2σ
        lcl_2sigma = max(0, media - 2 * desvio)  # Lower Control Limit 2σ
        ucl_3sigma = media + 3 * desvio  # Upper Control Limit 3σ
        lcl_3sigma = max(0, media - 3 * desvio)  # Lower Control Limit 3σ

        # Identificar pontos fora de controle
        pontos_fora = []
        for i, (mes, taxa) in enumerate(zip(meses, taxas)):
            if taxa > ucl_3sigma or taxa < lcl_3sigma:
                pontos_fora.append({
                    "mes": mes,
                    "taxa": taxa,
                    "severidade": "critico",
                    "desvio": abs(taxa - media) / desvio if desvio > 0 else 0
                })
            elif taxa > ucl_2sigma or taxa < lcl_2sigma:
                pontos_fora.append({
                    "mes": mes,
                    "taxa": taxa,
                    "severidade": "alerta",
                    "desvio": abs(taxa - media) / desvio if desvio > 0 else 0
                })

        result = {
            "meses": meses,
            "taxas": taxas,
            "totais": totais,
            "media": media,
            "desvio": desvio,
            "ucl_2sigma": ucl_2sigma,
            "lcl_2sigma": lcl_2sigma,
            "ucl_3sigma": ucl_3sigma,
            "lcl_3sigma": lcl_3sigma,
            "pontos_fora": pontos_fora
        }

        set_cached_data("shewhart_data", cache_key, result)
        return result

    except Exception as e:
        st.error(f"Erro ao calcular dados Shewhart: {e}")
        return {}


def get_temporal_comparison() -> dict:
    """
    Compara período atual com período anterior (MoM - Month over Month).
    """
    start_date, end_date = get_selected_period()
    cache_key = generate_cache_key("temporal_comparison", start_date.isoformat(), end_date.isoformat())
    cached = get_cached_data("temporal_comparison", cache_key)
    if cached is not None:
        return cached

    try:
        # Calcular período anterior com mesma duração
        duracao = (end_date - start_date).days
        periodo_anterior_fim = start_date - timedelta(days=1)
        periodo_anterior_inicio = periodo_anterior_fim - timedelta(days=duracao)

        # Dados do período atual
        dados_atual = get_monthly_positivity_data(
            analysis_type="screening",
            start_date_filter=start_date,
            end_date_filter=end_date
        )

        # Dados do período anterior
        dados_anterior = get_monthly_positivity_data(
            analysis_type="screening",
            start_date_filter=periodo_anterior_inicio,
            end_date_filter=periodo_anterior_fim
        )

        # Calcular totais
        total_atual = sum(d.get("positivo", 0) + d.get("negativo", 0) for d in dados_atual.values())
        positivos_atual = sum(d.get("positivo", 0) for d in dados_atual.values())
        taxa_atual = (positivos_atual / total_atual * 100) if total_atual > 0 else 0

        total_anterior = sum(d.get("positivo", 0) + d.get("negativo", 0) for d in dados_anterior.values())
        positivos_anterior = sum(d.get("positivo", 0) for d in dados_anterior.values())
        taxa_anterior = (positivos_anterior / total_anterior * 100) if total_anterior > 0 else 0

        # Calcular variações
        variacao_taxa = taxa_atual - taxa_anterior
        variacao_volume = ((total_atual - total_anterior) / total_anterior * 100) if total_anterior > 0 else 0

        result = {
            "periodo_atual": {
                "inicio": start_date,
                "fim": end_date,
                "total": total_atual,
                "positivos": positivos_atual,
                "taxa": taxa_atual
            },
            "periodo_anterior": {
                "inicio": periodo_anterior_inicio,
                "fim": periodo_anterior_fim,
                "total": total_anterior,
                "positivos": positivos_anterior,
                "taxa": taxa_anterior
            },
            "variacao_taxa": variacao_taxa,
            "variacao_volume": variacao_volume,
            "tendencia": "alta" if variacao_taxa > 0.5 else "baixa" if variacao_taxa < -0.5 else "estavel"
        }

        set_cached_data("temporal_comparison", cache_key, result)
        return result

    except Exception as e:
        st.error(f"Erro ao calcular comparação temporal: {e}")
        return {}


def get_lab_outliers() -> list:
    """
    Identifica laboratórios outliers (fora da curva) usando z-score.
    Inclui CNPJ, cidade e estado de cada laboratório.
    """
    start_date, end_date = get_selected_period()
    cache_key = generate_cache_key("lab_outliers", start_date.isoformat(), end_date.isoformat())
    cached = get_cached_data("lab_outliers", cache_key)
    if cached is not None:
        return cached

    try:
        lab_data = get_positivity_by_laboratory(start_date, end_date)
        labs_map = get_laboratories_map()

        # Criar mapa de ID para informações completas (CNPJ, cidade, estado)
        labs_full_info = get_laboratories_with_address()
        labs_info_map = {lab["id"]: lab for lab in labs_full_info}

        # Filtrar labs com mínimo de amostras
        labs_validos = []
        for lab_id, data in lab_data.items():
            total = data.get("total", 0)
            if total >= 50:  # Mínimo de 50 amostras
                lab_info = labs_info_map.get(lab_id, {})
                labs_validos.append({
                    "id": lab_id,
                    "nome": labs_map.get(lab_id, "Desconhecido"),
                    "cnpj": lab_info.get("cnpj", ""),
                    "cidade": lab_info.get("city", ""),
                    "estado": lab_info.get("state", ""),
                    "total": total,
                    "positivos": data.get("positivos", 0),
                    "taxa": data.get("taxa", 0)
                })

        if len(labs_validos) < 3:
            return []

        # Calcular estatísticas
        taxas = [l["taxa"] for l in labs_validos]
        media = sum(taxas) / len(taxas)
        variancia = sum((t - media) ** 2 for t in taxas) / len(taxas)
        desvio = variancia ** 0.5

        # Identificar outliers
        outliers = []
        for lab in labs_validos:
            if desvio > 0:
                z_score = (lab["taxa"] - media) / desvio
                if abs(z_score) > 2:  # Mais de 2 desvios padrão
                    lab["z_score"] = z_score
                    lab["media_nacional"] = media
                    lab["desvio_nacional"] = desvio
                    lab["severidade"] = "critico" if abs(z_score) > 3 else "alto"
                    lab["posicao"] = "acima" if z_score > 0 else "abaixo"
                    outliers.append(lab)

        # Ordenar por z-score (mais extremo primeiro)
        outliers.sort(key=lambda x: abs(x["z_score"]), reverse=True)

        set_cached_data("lab_outliers", cache_key, outliers)
        return outliers

    except Exception as e:
        st.error(f"Erro ao identificar outliers: {e}")
        return []


def get_substance_trends() -> list:
    """
    Analisa tendências de crescimento/queda por substância.
    """
    start_date, end_date = get_selected_period()
    cache_key = generate_cache_key("substance_trends", start_date.isoformat(), end_date.isoformat())
    cached = get_cached_data("substance_trends", cache_key)
    if cached is not None:
        return cached

    try:
        # Calcular período anterior
        duracao = (end_date - start_date).days
        periodo_anterior_fim = start_date - timedelta(days=1)
        periodo_anterior_inicio = periodo_anterior_fim - timedelta(days=duracao)

        # Dados atuais
        stats_atual = get_substance_statistics(month=None, analysis_type="all")

        # Para período anterior, precisamos ajustar a consulta
        # Por simplicidade, usar os dados atuais e calcular tendências
        trends = []
        for subst, data in stats_atual.items():
            if data.get("total", 0) >= 50:  # Mínimo de análises
                trends.append({
                    "substancia": subst,
                    "total": data.get("total", 0),
                    "positivos": data.get("positivos", 0),
                    "taxa": data.get("taxa", 0),
                    "percentual_total": round(data.get("positivos", 0) / max(sum(d.get("positivos", 0) for d in stats_atual.values()), 1) * 100, 2)
                })

        # Ordenar por taxa (maior primeiro)
        trends.sort(key=lambda x: x["taxa"], reverse=True)

        set_cached_data("substance_trends", cache_key, trends)
        return trends

    except Exception as e:
        st.error(f"Erro ao analisar tendências de substâncias: {e}")
        return []


def calculate_health_score(anomalias: dict, shewhart: dict, outliers: list) -> dict:
    """
    Calcula um score de saúde geral do sistema baseado nas anomalias detectadas.
    Score de 0 a 100 (100 = perfeito).
    """
    score = 100
    detalhes = []

    # Penalizar por anomalias
    total_anomalias = sum(len(v) for v in anomalias.values())
    if total_anomalias > 0:
        penalidade = min(30, total_anomalias * 2)
        score -= penalidade
        detalhes.append(f"-{penalidade} pts: {total_anomalias} anomalias detectadas")

    # Penalizar por pontos fora de controle (Shewhart)
    if shewhart.get("pontos_fora"):
        criticos = len([p for p in shewhart["pontos_fora"] if p.get("severidade") == "critico"])
        alertas = len([p for p in shewhart["pontos_fora"] if p.get("severidade") == "alerta"])
        penalidade = criticos * 10 + alertas * 5
        score -= min(30, penalidade)
        if penalidade > 0:
            detalhes.append(f"-{min(30, penalidade)} pts: {criticos} pontos críticos, {alertas} alertas no controle")

    # Penalizar por outliers
    if outliers:
        criticos = len([o for o in outliers if o.get("severidade") == "critico"])
        altos = len([o for o in outliers if o.get("severidade") == "alto"])
        penalidade = criticos * 8 + altos * 4
        score -= min(20, penalidade)
        if penalidade > 0:
            detalhes.append(f"-{min(20, penalidade)} pts: {len(outliers)} laboratórios outliers")

    score = max(0, score)

    # Determinar status
    if score >= 80:
        status = "Excelente"
        cor = "#28A745"
    elif score >= 60:
        status = "Bom"
        cor = "#17A2B8"
    elif score >= 40:
        status = "Atenção"
        cor = "#FFC107"
    else:
        status = "Crítico"
        cor = "#DC3545"

    return {
        "score": score,
        "status": status,
        "cor": cor,
        "detalhes": detalhes
    }


def render_auditoria():
    """
    Página 6 - Auditoria e Anomalias
    Implementa controle estatístico, comparação temporal, análise de outliers e alertas.
    """
    st.title("🔍 Auditoria e Anomalias")

    # Carregar todos os dados necessários
    tasks = [
        ("Anomalias", detect_anomalies, (), {}),
        ("Shewhart", get_shewhart_data, (), {}),
        ("Comparação Temporal", get_temporal_comparison, (), {}),
        ("Outliers", get_lab_outliers, (), {}),
        ("Tendências Substâncias", get_substance_trends, (), {}),
    ]

    results = loading_with_progress(tasks, "Analisando dados de auditoria...")

    anomalias = results.get("Anomalias", {})
    shewhart = results.get("Shewhart", {})
    comparacao = results.get("Comparação Temporal", {})
    outliers = results.get("Outliers", [])
    tendencias_subst = results.get("Tendências Substâncias", [])

    # Calcular score de saúde
    health = calculate_health_score(anomalias, shewhart, outliers)

    # ============================================
    # KPIs RESUMO
    # ============================================
    total_anomalias = sum(len(v) for v in anomalias.values())
    criticos = len(shewhart.get("pontos_fora", [])) + len([o for o in outliers if o.get("severidade") == "critico"])
    alertas = len([a for cat in anomalias.values() for a in cat]) + len([o for o in outliers if o.get("severidade") == "alto"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {health['cor']} 0%, {health['cor']}CC 100%); padding: 18px; border-radius: 12px; text-align: center;">
            <div style="color: rgba(255,255,255,0.9); font-size: 12px; text-transform: uppercase;">Score de Saúde</div>
            <div style="color: white; font-size: 32px; font-weight: bold;">{health['score']}</div>
            <div style="color: rgba(255,255,255,0.9); font-size: 14px;">{health['status']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%); padding: 18px; border-radius: 12px; text-align: center;">
            <div style="color: #888; font-size: 12px; text-transform: uppercase;">Total Anomalias</div>
            <div style="color: white; font-size: 32px; font-weight: bold;">{total_anomalias}</div>
            <div style="color: #888; font-size: 12px;">detectadas</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #DC3545 0%, #B02A37 100%); padding: 18px; border-radius: 12px; text-align: center;">
            <div style="color: rgba(255,255,255,0.9); font-size: 12px; text-transform: uppercase;">Críticos</div>
            <div style="color: white; font-size: 32px; font-weight: bold;">{criticos}</div>
            <div style="color: rgba(255,255,255,0.9); font-size: 12px;">requerem ação</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFC107 0%, #E0A800 100%); padding: 18px; border-radius: 12px; text-align: center;">
            <div style="color: rgba(0,0,0,0.7); font-size: 12px; text-transform: uppercase;">Alertas</div>
            <div style="color: #1A1A2E; font-size: 32px; font-weight: bold;">{alertas}</div>
            <div style="color: rgba(0,0,0,0.6); font-size: 12px;">monitorar</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================================
    # GRÁFICO DE CONTROLE ESTATÍSTICO (SHEWHART)
    # ============================================
    st.subheader("📈 Gráfico de Controle Estatístico (Shewhart)")

    if shewhart and shewhart.get("meses"):
        # Criar gráfico
        fig_shewhart = go.Figure()

        # Linha dos dados
        fig_shewhart.add_trace(go.Scatter(
            x=shewhart["meses"],
            y=shewhart["taxas"],
            mode='lines+markers',
            name='Taxa de Positividade',
            line=dict(color='#0066CC', width=2),
            marker=dict(size=8)
        ))

        # Linha central (média)
        fig_shewhart.add_hline(
            y=shewhart["media"],
            line_dash="solid",
            line_color="#28A745",
            line_width=2,
            annotation_text=f"Média: {shewhart['media']:.2f}%",
            annotation_position="right"
        )

        # Limites ±2σ
        fig_shewhart.add_hline(
            y=shewhart["ucl_2sigma"],
            line_dash="dash",
            line_color="#FFC107",
            line_width=1,
            annotation_text=f"+2σ: {shewhart['ucl_2sigma']:.2f}%",
            annotation_position="right"
        )
        fig_shewhart.add_hline(
            y=shewhart["lcl_2sigma"],
            line_dash="dash",
            line_color="#FFC107",
            line_width=1,
            annotation_text=f"-2σ: {shewhart['lcl_2sigma']:.2f}%",
            annotation_position="right"
        )

        # Limites ±3σ
        fig_shewhart.add_hline(
            y=shewhart["ucl_3sigma"],
            line_dash="dot",
            line_color="#DC3545",
            line_width=1,
            annotation_text=f"+3σ: {shewhart['ucl_3sigma']:.2f}%",
            annotation_position="right"
        )
        fig_shewhart.add_hline(
            y=shewhart["lcl_3sigma"],
            line_dash="dot",
            line_color="#DC3545",
            line_width=1,
            annotation_text=f"-3σ: {shewhart['lcl_3sigma']:.2f}%",
            annotation_position="right"
        )

        # Destacar pontos fora de controle
        for ponto in shewhart.get("pontos_fora", []):
            idx = shewhart["meses"].index(ponto["mes"]) if ponto["mes"] in shewhart["meses"] else -1
            if idx >= 0:
                cor = "#DC3545" if ponto["severidade"] == "critico" else "#FFC107"
                fig_shewhart.add_trace(go.Scatter(
                    x=[ponto["mes"]],
                    y=[ponto["taxa"]],
                    mode='markers',
                    marker=dict(size=14, color=cor, symbol='x'),
                    name=f'{ponto["severidade"].capitalize()}: {ponto["mes"]}',
                    showlegend=False
                ))

        fig_shewhart.update_layout(
            height=450,
            margin=dict(t=30, b=50, l=50, r=100),
            xaxis_title="Período",
            yaxis_title="Taxa de Positividade (%)",
            yaxis_ticksuffix="%",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig_shewhart, use_container_width=True, key="chart_shewhart")

        # Legenda explicativa
        st.caption("**Interpretação:** Pontos dentro dos limites ±2σ (amarelo) estão sob controle normal. Pontos fora dos limites ±3σ (vermelho) indicam comportamento atípico que requer investigação.")
    else:
        st.info("Dados insuficientes para gerar o gráfico de controle estatístico.")

    st.markdown("---")

    # ============================================
    # COMPARATIVO TEMPORAL E LABORATÓRIOS OUTLIERS
    # ============================================
    col_temp, col_outliers = st.columns(2)

    with col_temp:
        st.subheader("📊 Comparativo Temporal")

        if comparacao:
            atual = comparacao.get("periodo_atual", {})
            anterior = comparacao.get("periodo_anterior", {})

            # Cards de comparação
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Período Atual**")
                st.markdown(f"📅 {atual.get('inicio', '').strftime('%d/%m/%Y') if atual.get('inicio') else '-'} a {atual.get('fim', '').strftime('%d/%m/%Y') if atual.get('fim') else '-'}")
                st.metric("Taxa", f"{atual.get('taxa', 0):.2f}%")
                st.metric("Volume", f"{atual.get('total', 0):,}".replace(",", "."))

            with col_b:
                st.markdown("**Período Anterior**")
                st.markdown(f"📅 {anterior.get('inicio', '').strftime('%d/%m/%Y') if anterior.get('inicio') else '-'} a {anterior.get('fim', '').strftime('%d/%m/%Y') if anterior.get('fim') else '-'}")
                st.metric("Taxa", f"{anterior.get('taxa', 0):.2f}%")
                st.metric("Volume", f"{anterior.get('total', 0):,}".replace(",", "."))

            # Variações
            st.markdown("---")
            var_taxa = comparacao.get("variacao_taxa", 0)
            var_volume = comparacao.get("variacao_volume", 0)
            tendencia = comparacao.get("tendencia", "estavel")

            cor_taxa = "#DC3545" if var_taxa > 0 else "#28A745" if var_taxa < 0 else "#6C757D"
            cor_volume = "#28A745" if var_volume > 0 else "#DC3545" if var_volume < 0 else "#6C757D"

            st.markdown(f"""
            <div style="display: flex; gap: 10px;">
                <div style="flex: 1; background: #1A1A2E; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #888; font-size: 11px;">Variação Taxa</div>
                    <div style="color: {cor_taxa}; font-size: 20px; font-weight: bold;">{'+' if var_taxa > 0 else ''}{var_taxa:.2f}%</div>
                </div>
                <div style="flex: 1; background: #1A1A2E; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #888; font-size: 11px;">Variação Volume</div>
                    <div style="color: {cor_volume}; font-size: 20px; font-weight: bold;">{'+' if var_volume > 0 else ''}{var_volume:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            icone_tendencia = "📈" if tendencia == "alta" else "📉" if tendencia == "baixa" else "➡️"
            st.markdown(f"**Tendência:** {icone_tendencia} {tendencia.capitalize()}")
        else:
            st.info("Dados insuficientes para comparação temporal.")

    with col_outliers:
        st.subheader("🏢 Laboratórios Outliers")

        if outliers:
            for lab in outliers[:5]:  # Top 5 outliers
                cor = "#DC3545" if lab.get("severidade") == "critico" else "#FFC107"
                icone = "🔴" if lab.get("severidade") == "critico" else "🟠"
                direcao = "↑" if lab.get("posicao") == "acima" else "↓"
                cnpj = lab.get('cnpj', '')
                cidade = lab.get('cidade', '')
                estado = lab.get('estado', '')
                localizacao = f"{cidade}/{estado}" if cidade and estado else cidade or estado or ""

                st.markdown(f"""
                <div style="background: #1A1A2E; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {cor};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="color: white; font-weight: bold;">{icone} {lab.get('nome', 'Desconhecido')[:30]}</div>
                            <div style="color: #00CED1; font-size: 11px; font-family: monospace;">{cnpj if cnpj else 'CNPJ não informado'}</div>
                            <div style="color: #888; font-size: 11px;">{localizacao} • {lab.get('total', 0):,} amostras</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: {cor}; font-size: 18px; font-weight: bold;">{lab.get('taxa', 0):.2f}% {direcao}</div>
                            <div style="color: #888; font-size: 11px;">z-score: {lab.get('z_score', 0):.2f}σ</div>
                        </div>
                    </div>
                </div>
                """.replace(",", "."), unsafe_allow_html=True)

            st.caption(f"**Média nacional:** {outliers[0].get('media_nacional', 0):.2f}% | **Desvio padrão:** {outliers[0].get('desvio_nacional', 0):.2f}%")
        else:
            st.success("Nenhum laboratório outlier identificado.")

    st.markdown("---")

    # ============================================
    # ALERTAS POR SEVERIDADE
    # ============================================
    st.subheader("⚠️ Alertas por Severidade")

    # Organizar alertas por severidade
    alertas_criticos = []
    alertas_altos = []
    alertas_medios = []

    # Pontos fora de controle (Shewhart)
    for ponto in shewhart.get("pontos_fora", []):
        alerta = {
            "tipo": "Controle Estatístico",
            "descricao": f"Taxa de {ponto['taxa']:.2f}% em {ponto['mes']} ({ponto['desvio']:.1f}σ do normal)",
            "acao": "Investigar causas do desvio"
        }
        if ponto.get("severidade") == "critico":
            alertas_criticos.append(alerta)
        else:
            alertas_altos.append(alerta)

    # Laboratórios outliers
    for lab in outliers:
        alerta = {
            "tipo": "Laboratório Outlier",
            "descricao": f"{lab.get('nome', 'Desconhecido')} com taxa {lab.get('taxa', 0):.2f}% ({lab.get('posicao', '')} da média)",
            "acao": "Auditar processos do laboratório"
        }
        if lab.get("severidade") == "critico":
            alertas_criticos.append(alerta)
        else:
            alertas_altos.append(alerta)

    # Anomalias detectadas
    for taxa in anomalias.get("taxas_extremas", []):
        alertas_medios.append({
            "tipo": "Taxa Extrema",
            "descricao": f"{taxa.get('Laboratório', 'Desconhecido')}: {taxa.get('Taxa (%)', '0%')} ({taxa.get('Tipo', '')})",
            "acao": "Verificar dados"
        })

    for volume in anomalias.get("volumes_atipicos", []):
        alertas_medios.append({
            "tipo": "Volume Atípico",
            "descricao": f"{volume.get('Mês', '')}: {volume.get('Volume', 0)} amostras ({volume.get('Desvio', '')})",
            "acao": "Verificar sazonalidade"
        })

    # Exibir alertas em expanders
    if alertas_criticos:
        with st.expander(f"🔴 Críticos ({len(alertas_criticos)})", expanded=True):
            for alerta in alertas_criticos:
                st.markdown(f"""
                <div style="background: rgba(220, 53, 69, 0.1); border-left: 4px solid #DC3545; padding: 10px; margin-bottom: 8px; border-radius: 4px;">
                    <strong>{alerta['tipo']}</strong><br>
                    <span style="color: #888;">{alerta['descricao']}</span><br>
                    <em style="color: #DC3545;">Ação: {alerta['acao']}</em>
                </div>
                """, unsafe_allow_html=True)

    if alertas_altos:
        with st.expander(f"🟠 Altos ({len(alertas_altos)})", expanded=len(alertas_criticos) == 0):
            for alerta in alertas_altos:
                st.markdown(f"""
                <div style="background: rgba(255, 193, 7, 0.1); border-left: 4px solid #FFC107; padding: 10px; margin-bottom: 8px; border-radius: 4px;">
                    <strong>{alerta['tipo']}</strong><br>
                    <span style="color: #888;">{alerta['descricao']}</span><br>
                    <em style="color: #E0A800;">Ação: {alerta['acao']}</em>
                </div>
                """, unsafe_allow_html=True)

    if alertas_medios:
        with st.expander(f"🟡 Médios ({len(alertas_medios)})", expanded=False):
            for alerta in alertas_medios:
                st.markdown(f"""
                <div style="background: rgba(23, 162, 184, 0.1); border-left: 4px solid #17A2B8; padding: 10px; margin-bottom: 8px; border-radius: 4px;">
                    <strong>{alerta['tipo']}</strong><br>
                    <span style="color: #888;">{alerta['descricao']}</span><br>
                    <em style="color: #17A2B8;">Ação: {alerta['acao']}</em>
                </div>
                """, unsafe_allow_html=True)

    if not alertas_criticos and not alertas_altos and not alertas_medios:
        st.success("✅ Nenhum alerta detectado no período selecionado!")

    st.markdown("---")

    # ============================================
    # ANÁLISE DE SUBSTÂNCIAS
    # ============================================
    st.subheader("🧪 Análise de Substâncias")

    if tendencias_subst:
        col_graf, col_tab = st.columns([2, 1])

        with col_graf:
            # Top 10 substâncias por taxa
            top_subst = tendencias_subst[:10]

            df_subst = pd.DataFrame(top_subst)
            df_subst = df_subst.sort_values("taxa", ascending=True)

            fig_subst = px.bar(
                df_subst,
                y="substancia",
                x="taxa",
                orientation="h",
                title="Top 10 Substâncias por Taxa de Positividade",
                text=df_subst["taxa"].apply(lambda x: f"{x:.2f}%"),
                color="taxa",
                color_continuous_scale=["#00CED1", "#DC3545"]
            )

            fig_subst.update_traces(textposition="outside")
            fig_subst.update_layout(
                height=400,
                margin=dict(t=40, b=20, l=10, r=10),
                xaxis_title="Taxa (%)",
                yaxis_title="",
                showlegend=False,
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_subst, use_container_width=True, key="chart_subst_audit")

        with col_tab:
            st.markdown("**Resumo das Substâncias**")
            df_display = pd.DataFrame(tendencias_subst[:10])
            df_display["taxa"] = df_display["taxa"].apply(lambda x: f"{x:.2f}%")
            df_display["positivos"] = df_display["positivos"].apply(lambda x: f"{x:,}".replace(",", "."))
            df_display["total"] = df_display["total"].apply(lambda x: f"{x:,}".replace(",", "."))
            df_display["percentual_total"] = df_display["percentual_total"].apply(lambda x: f"{x:.1f}%")

            df_display = df_display.rename(columns={
                "substancia": "Substância",
                "taxa": "Taxa",
                "positivos": "Positivos",
                "total": "Total",
                "percentual_total": "% Total"
            })

            st.dataframe(
                df_display[["Substância", "Taxa", "Positivos", "% Total"]],
                use_container_width=True,
                hide_index=True,
                height=350
            )
    else:
        st.info("Nenhum dado de substância disponível.")

    st.markdown("---")

    # ============================================
    # EXPORTAR RELATÓRIO
    # ============================================
    st.subheader("📥 Exportar Relatório de Auditoria")

    # Preparar dados para exportação
    export_data = {
        "Data da Análise": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Score de Saúde": health["score"],
        "Status": health["status"],
        "Total Anomalias": total_anomalias,
        "Alertas Críticos": len(alertas_criticos),
        "Alertas Altos": len(alertas_altos),
        "Alertas Médios": len(alertas_medios)
    }

    # Criar workbook Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Resumo
        df_resumo = pd.DataFrame([export_data])
        df_resumo.to_excel(writer, sheet_name='Resumo', index=False)

        # Outliers
        if outliers:
            df_outliers = pd.DataFrame(outliers)
            df_outliers.to_excel(writer, sheet_name='Outliers', index=False)

        # Alertas
        all_alertas = []
        for a in alertas_criticos:
            a["severidade"] = "Crítico"
            all_alertas.append(a)
        for a in alertas_altos:
            a["severidade"] = "Alto"
            all_alertas.append(a)
        for a in alertas_medios:
            a["severidade"] = "Médio"
            all_alertas.append(a)

        if all_alertas:
            df_alertas = pd.DataFrame(all_alertas)
            df_alertas.to_excel(writer, sheet_name='Alertas', index=False)

        # Substâncias
        if tendencias_subst:
            df_subst_exp = pd.DataFrame(tendencias_subst)
            df_subst_exp.to_excel(writer, sheet_name='Substancias', index=False)

    output.seek(0)
    st.download_button(
        "📊 Download Relatório Excel",
        output,
        f"auditoria_{datetime.now().strftime('%Y%m%d')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False
    )

    # Expander explicativo das métricas
    st.markdown("---")
    with st.expander("📖 Glossário: Entenda as Métricas de Auditoria", expanded=False):
        st.markdown("""
        ### Score de Saúde (0-100)
        Indicador geral da qualidade dos dados e processos. Calculado a partir das anomalias detectadas:
        - **80-100 (Excelente):** Sistema operando dentro dos parâmetros esperados
        - **60-79 (Bom):** Algumas anomalias detectadas, mas sob controle
        - **40-59 (Atenção):** Múltiplas anomalias requerem investigação
        - **0-39 (Crítico):** Situação grave que demanda ação imediata

        ---

        ### Z-Score (Desvio Padrão - σ)
        Mede quantos desvios padrão um valor está distante da média. Usado para identificar outliers:
        - **|z| < 2σ:** Valor normal, dentro da variação esperada
        - **|z| entre 2σ e 3σ:** Valor atípico, merece atenção (Alerta Alto)
        - **|z| > 3σ:** Valor muito extremo, fora do padrão (Crítico)

        *Exemplo: Se a média nacional é 5% e o desvio padrão é 1%, um laboratório com 8% tem z-score = 3σ (crítico).*

        ---

        ### Gráfico de Controle (Shewhart)
        Técnica estatística para monitorar processos ao longo do tempo:
        - **Linha Central (verde):** Média histórica da taxa de positividade
        - **Limites ±2σ (amarelo):** Zona de alerta - valores fora requerem monitoramento
        - **Limites ±3σ (vermelho):** Zona crítica - valores fora indicam processo fora de controle

        *Pontos consecutivos próximos aos limites também podem indicar tendências problemáticas.*

        ---

        ### Níveis de Severidade

        | Nível | Cor | Critério | Ação Recomendada |
        |-------|-----|----------|------------------|
        | 🔴 **Crítico** | Vermelho | z-score > 3σ ou processo fora de controle | Investigação imediata |
        | 🟠 **Alto** | Laranja | z-score entre 2σ e 3σ | Monitorar de perto |
        | 🟡 **Médio** | Amarelo | Anomalias pontuais ou tendências | Acompanhar evolução |

        ---

        ### Laboratórios Outliers
        Laboratórios identificados com taxas de positividade significativamente diferentes da média nacional:
        - **Acima da média (↑):** Taxa maior que o esperado - pode indicar problemas de qualidade ou perfil regional específico
        - **Abaixo da média (↓):** Taxa menor que o esperado - pode indicar subnotificação ou processos diferenciados

        ---

        ### Comparativo Temporal
        Compara o período atual com o período anterior de mesma duração:
        - **Variação de Taxa:** Diferença percentual na taxa de positividade
        - **Variação de Volume:** Diferença percentual no número de amostras
        - **Tendência:** Alta (↑), Baixa (↓) ou Estável (→)
        """)


def detect_anomalies() -> dict:
    """
    Detecta anomalias nos dados.
    Otimizado para usar dados agregados em vez de consultas individuais por laboratório.
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
        # 1. Verificar taxas extremas por laboratório usando dados agregados
        # Em vez de chamar get_metrics_data para cada lab, usar get_positivity_by_laboratory
        start_date, end_date = get_selected_period()
        lab_data = get_positivity_by_laboratory(start_date, end_date)
        labs_map = get_laboratories_map()

        for lab_id, data in lab_data.items():
            lab_name = labs_map.get(lab_id, "Desconhecido")
            total = data.get("total", 0)
            taxa = data.get("taxa", 0)

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
    Taxa de positividade = positivas confirmatórias / total amostras triagem * 100
    """
    comparison_data = []

    for lab in network_labs:
        lab_id = lab.get('id')
        lab_name = lab.get('name', 'Desconhecido')
        lab_city = lab.get('city', '')
        lab_state = lab.get('state', '')

        # Buscar métricas do laboratório
        metrics = get_metrics_data(laboratory_ids=[lab_id])

        # Total de amostras = triagem
        total_amostras = metrics.get("total_amostras", 0)
        # Positivas = confirmatório
        positivas_confirmatorio = metrics.get("positivas_confirmatorio", 0)
        # Total confirmatório (para exibição)
        total_confirmatorio = metrics.get("positivas_confirmatorio", 0) + metrics.get("negativas_confirmatorio", 0)
        # Taxa = positivas confirmatórias / total amostras triagem
        taxa_positividade = (positivas_confirmatorio / total_amostras * 100) if total_amostras > 0 else 0.0

        comparison_data.append({
            "id": lab_id,
            "nome": lab_name,
            "cidade": lab_city,
            "estado": lab_state,
            "total_amostras": total_amostras,
            "positivas_triagem": metrics.get("positivas_triagem", 0),
            "negativas_triagem": metrics.get("negativas_triagem", 0),
            "positivas_confirmatorio": positivas_confirmatorio,
            "negativas_confirmatorio": metrics.get("negativas_confirmatorio", 0),
            "total_confirmatorio": total_confirmatorio,
            "taxa_positividade": taxa_positividade
        })

    return comparison_data


def render_rede():
    """
    Página 7 - Comparação entre Filiais da Rede
    """
    st.title("🏢 Rede - Comparação entre Filiais")

    # Buscar redes identificadas por CNPJ
    networks = loading_single(get_networks_by_cnpj, "Identificando redes por CNPJ...")

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
    comparison_data = loading_single(
        get_network_comparison_data, "Carregando dados das filiais...",
        labs_da_rede
    )

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

    # KPIs resumidos da rede
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    total_amostras_rede = df_comparison['total_amostras'].sum()
    total_confirmatorio_rede = df_comparison['total_confirmatorio'].sum() if 'total_confirmatorio' in df_comparison.columns else 0
    total_positivos_conf = df_comparison['positivas_confirmatorio'].sum()

    with col_kpi1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%); padding: 15px; border-radius: 10px; text-align: center;">
            <div style="color: #888; font-size: 12px;">Total Amostras (Rede)</div>
            <div style="color: white; font-size: 24px; font-weight: bold;">{total_amostras_rede:,}</div>
        </div>
        """.replace(',', '.'), unsafe_allow_html=True)

    with col_kpi2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0F3460 0%, #16213E 100%); padding: 15px; border-radius: 10px; text-align: center;">
            <div style="color: #888; font-size: 12px;">Total Confirmatório</div>
            <div style="color: #00CED1; font-size: 24px; font-weight: bold;">{total_confirmatorio_rede:,}</div>
        </div>
        """.replace(',', '.'), unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #DC3545 0%, #B02A37 100%); padding: 15px; border-radius: 10px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">Positivos Confirmatório</div>
            <div style="color: white; font-size: 24px; font-weight: bold;">{total_positivos_conf:,}</div>
        </div>
        """.replace(',', '.'), unsafe_allow_html=True)

    with col_kpi4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFA500 0%, #E69500 100%); padding: 15px; border-radius: 10px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">Taxa Média (Rede)</div>
            <div style="color: white; font-size: 24px; font-weight: bold;">{taxa_media_rede:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico de barras horizontal - mais visual
    st.subheader("📊 Comparativo: Taxa de Positividade por Filial (Confirmatório)")

    # Ordenar por taxa de positividade
    df_chart_taxa = df_comparison.sort_values('taxa_positividade', ascending=True)

    # Cores das barras baseadas na média
    cores_barras = []
    for taxa in df_chart_taxa['taxa_positividade']:
        if taxa > taxa_media_rede * 1.5:
            cores_barras.append('#DC3545')  # Vermelho
        elif taxa > taxa_media_rede:
            cores_barras.append('#FFA500')  # Laranja
        else:
            cores_barras.append('#28A745')  # Verde

    fig_taxa = go.Figure()

    fig_taxa.add_trace(go.Bar(
        y=df_chart_taxa['label'],
        x=df_chart_taxa['taxa_positividade'],
        orientation='h',
        marker_color=cores_barras,
        text=df_chart_taxa['taxa_positividade'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        textfont=dict(size=12)
    ))

    # Linha vertical da média
    fig_taxa.add_vline(
        x=taxa_media_rede,
        line_dash="dash",
        line_color="#666",
        line_width=2,
        annotation_text=f"Média: {taxa_media_rede:.1f}%",
        annotation_position="top"
    )

    max_taxa = df_chart_taxa['taxa_positividade'].max()
    fig_taxa.update_layout(
        height=max(350, len(df_chart_taxa) * 45),
        margin=dict(t=40, b=40, l=150, r=80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Taxa de Positividade (%)",
            range=[0, max(max_taxa * 1.3, 10)],
            gridcolor='rgba(128,128,128,0.3)'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11)
        )
    )

    st.plotly_chart(fig_taxa, use_container_width=True, key="chart_rede_taxa")

    # Legenda de cores
    st.markdown("""
    <div style="display: flex; gap: 20px; justify-content: center; margin: 10px 0; font-size: 13px;">
        <span><strong>Taxa Confirmatório:</strong></span>
        <span style="color: #28A745;">● Abaixo da média</span>
        <span style="color: #FFA500;">● Acima da média</span>
        <span style="color: #DC3545;">● Muito acima (+50%)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Gráfico de volume por filial
    st.subheader("📈 Volume de Amostras por Filial")

    df_chart_vol = df_comparison.sort_values('total_amostras', ascending=True)

    fig_vol = go.Figure()

    fig_vol.add_trace(go.Bar(
        y=df_chart_vol['label'],
        x=df_chart_vol['total_amostras'],
        orientation='h',
        marker_color='#00CED1',
        text=df_chart_vol['total_amostras'].apply(lambda x: f'{x:,}'.replace(',', '.')),
        textposition='outside',
        textfont=dict(size=12)
    ))

    max_vol = df_chart_vol['total_amostras'].max()
    fig_vol.update_layout(
        height=max(350, len(df_chart_vol) * 45),
        margin=dict(t=40, b=40, l=150, r=80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Total de Amostras",
            range=[0, max_vol * 1.25],
            gridcolor='rgba(128,128,128,0.3)',
            showticklabels=False,
            showgrid=False
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11)
        )
    )

    st.plotly_chart(fig_vol, use_container_width=True, key="chart_rede_volume")

    st.markdown("---")

    # Tabela detalhada
    st.subheader("📋 Tabela Comparativa Detalhada")

    # Preparar DataFrame para exibição com métricas de confirmatório
    df_display = df_comparison[[
        'nome', 'cidade', 'estado', 'total_amostras',
        'total_confirmatorio', 'positivas_confirmatorio', 'negativas_confirmatorio', 'taxa_positividade'
    ]].copy()

    df_display.columns = [
        'Filial', 'Cidade', 'Estado', 'Total Amostras',
        'Total Confirm.', 'Positivos Confirm.', 'Negativos Confirm.', 'Taxa Positividade (%)'
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

    csv_buffer = io.StringIO()
    df_display.to_csv(csv_buffer, index=False, sep=';', encoding='utf-8')
    st.download_button(
        "📊 Download CSV",
        csv_buffer.getvalue(),
        f"rede_{cnpj_base}_comparacao.csv",
        "text/csv",
        use_container_width=True
    )


if __name__ == "__main__":
    main()
