import os
import glob
import time
import chromadb
import pymupdf as fitz
import streamlit as st
from google import genai
from google.genai import types

# =====================================================================
# 1. CONFIGURAÇÃO DA PÁGINA & IDENTIDADE VISUAL (PRETO & AZUL)
# =====================================================================
st.set_page_config(
    page_title="Tutor Acadêmico IA | Ciência da Computação",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #09090B;
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid #1E293B;
    }

    /* Cabeçalho Preto & Azul */
    .tutor-header {
        background: linear-gradient(135deg, #0D1117 0%, #0F2544 100%);
        border: 1px solid #1D4ED8;
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.2);
    }
    .tutor-title {
        font-size: 2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .tutor-title span {
        color: #38BDF8;
    }
    .tutor-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 4px;
    }
    .tutor-badge {
        background-color: rgba(37, 99, 235, 0.2);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Botões em Azul */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(90deg, #1D4ED8 0%, #2563EB 100%);
        color: #FFFFFF;
        border: 1px solid #3B82F6;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #2563EB 0%, #0284C7 100%);
        border-color: #38BDF8;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.35);
        transform: translateY(-1px);
    }

    /* Slider em Azul */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #2563EB !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.7) !important;
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #1D4ED8 0%, #38BDF8 100%) !important;
    }
    [data-testid="stSliderThumbValue"] {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }

    /* Radio Button em Azul */
    [data-testid="stRadio"] div[role="radiogroup"] label div:first-child {
        border-color: #3B82F6 !important;
    }

    /* Abas estilizadas em Preto & Azul */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0D1117;
        border-radius: 8px 8px 0 0;
        color: #94A3B8;
        border: 1px solid #1E293B;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0F2544 !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

PASTA_MATERIAS = "./materias_faculdade"
PASTA_BANCO = "./banco_estudos"

@st.cache_resource
def inicializar_recursos():
    client = genai.Client()
    chroma_client = chromadb.PersistentClient(path=PASTA_BANCO)
    collection = chroma_client.get_or_create_collection(name="materias_faculdade")
    return client, collection

client, collection = inicializar_recursos()

# =====================================================================
# 2. PIPELINE DE INDEXAÇÃO & GERENCIAMENTO DE ARQUIVOS
# =====================================================================

def descrever_pagina_imagem(pagina_fitz) -> str:
    """Converte uma página do PDF em PNG e usa o Gemini para extrair texto e diagramas."""
    pix = pagina_fitz.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    
    modelos = ["gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-2.5-flash"]
    for modelo in modelos:
        try:
            response = client.models.generate_content(
                model=modelo,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    "Transcreva todo o texto deste slide e descreva detalhadamente qualquer diagrama, gráfico ou fluxo visual presente nele para fins de estudo universitário."
                ]
            )
            return response.text or ""
        except Exception:
            time.sleep(1.5)
    return ""


def executar_indexacao_visual(pasta_raiz: str = PASTA_MATERIAS):
    """Renderiza um painel de status ao vivo na tela principal acompanhando cada página."""
    os.makedirs(pasta_raiz, exist_ok=True)
    arquivos_pdf = glob.glob(os.path.join(pasta_raiz, "**", "*.pdf"), recursive=True)

    with st.status("⚙️ Processando e Indexando PDFs no Banco Vetorial...", expanded=True) as status_box:
        if not arquivos_pdf:
            st.warning(f"Nenhum PDF encontrado em '{pasta_raiz}'.")
            status_box.update(label="⚠️ Nenhum arquivo encontrado", state="error")
            return

        ids_existentes = set(collection.get()["ids"])
        barra_progresso = st.progress(0.0, text="Iniciando leitura dos arquivos...")
        log_container = st.empty()
        total_novas_paginas = 0
        resumo_arquivos = []

        for idx_arq, caminho_pdf in enumerate(arquivos_pdf):
            materia = os.path.basename(os.path.dirname(caminho_pdf))
            nome_arquivo = os.path.basename(caminho_pdf)
            
            try:
                doc_fitz = fitz.open(caminho_pdf)
            except Exception as e:
                st.error(f"Erro ao abrir `{nome_arquivo}`: {e}")
                continue

            total_pags_pdf = len(doc_fitz)
            documentos, ids, metadados = [], [], []
            novas_neste_pdf = 0

            for num_pagina in range(total_pags_pdf):
                doc_id = f"{materia}_{nome_arquivo}_pag_{num_pagina + 1}"
                
                progresso_atual = (idx_arq + ((num_pagina + 1) / max(1, total_pags_pdf))) / len(arquivos_pdf)
                barra_progresso.progress(
                    min(1.0, progresso_atual),
                    text=f"📂 [{materia}] {nome_arquivo} — Página {num_pagina + 1} de {total_pags_pdf}"
                )

                if doc_id in ids_existentes:
                    continue

                pagina = doc_fitz[num_pagina]
                texto = pagina.get_text().strip()

                if len(texto) < 60:
                    log_container.info(f"👁️ **Visão Computacional IA:** Lendo slide visual `{nome_arquivo}` (Pág. {num_pagina + 1}/{total_pags_pdf})...")
                    descricao_img = descrever_pagina_imagem(pagina)
                    texto = f"{texto}\n[Conteúdo Visual do Slide]: {descricao_img}".strip()
                    time.sleep(1.5)
                else:
                    log_container.caption(f"📄 Extraindo texto de `{nome_arquivo}` — Pág. {num_pagina + 1}/{total_pags_pdf}...")

                if texto:
                    texto_enriquecido = f"[Matéria: {materia} | Arquivo: {nome_arquivo} | Pág. {num_pagina + 1}]\n{texto}"
                    documentos.append(texto_enriquecido)
                    ids.append(doc_id)
                    metadados.append({
                        "materia": materia,
                        "arquivo": nome_arquivo,
                        "pagina": num_pagina + 1
                    })
                    novas_neste_pdf += 1

            doc_fitz.close()

            if documentos:
                log_container.info(f"💾 Salvando {len(documentos)} páginas de `{nome_arquivo}` no ChromaDB...")
                collection.upsert(documents=documentos, ids=ids, metadatas=metadados)
                total_novas_paginas += novas_neste_pdf
                resumo_arquivos.append(f"✅ **{materia} / {nome_arquivo}**: {novas_neste_pdf} páginas novas indexadas.")
            else:
                resumo_arquivos.append(f"⏭️ **{materia} / {nome_arquivo}**: já estava 100% indexado ({total_pags_pdf} págs).")

        barra_progresso.progress(1.0, text="Concluído!")
        log_container.empty()

        for linha in resumo_arquivos:
            st.markdown(linha)

        st.success(f"🎉 Banco atualizado! Total geral agora: **{collection.count()} páginas**.")
        status_box.update(
            label=f"✅ Indexação concluída! ({total_novas_paginas} páginas novas adicionadas)",
            state="complete",
            expanded=True
        )


def listar_materias_disponiveis() -> list:
    """Retorna as matérias encontradas nas pastas e nos metadados do ChromaDB."""
    materias = set()
    if os.path.exists(PASTA_MATERIAS):
        for nome in os.listdir(PASTA_MATERIAS):
            if os.path.isdir(os.path.join(PASTA_MATERIAS, nome)):
                materias.add(nome)
    try:
        dados = collection.get(include=["metadatas"])
        for meta in dados.get("metadatas", []):
            if meta and meta.get("materia"):
                materias.add(meta["materia"])
    except Exception:
        pass
    return sorted(list(materias))


def listar_arquivos_da_materia(materia: str) -> dict:
    """Retorna um dicionário {nome_arquivo: qtd_paginas_indexadas} para a matéria selecionada."""
    arquivos_info = {}
    
    # 1. Verifica arquivos salvos na pasta local
    pasta_mat = os.path.join(PASTA_MATERIAS, materia)
    if os.path.exists(pasta_mat):
        for arq in os.listdir(pasta_mat):
            if arq.lower().endswith(".pdf"):
                arquivos_info[arq] = 0

    # 2. Conta quantas páginas de cada arquivo estão indexadas no ChromaDB
    try:
        dados = collection.get(where={"materia": materia}, include=["metadatas"])
        for meta in dados.get("metadatas", []):
            if meta and meta.get("arquivo"):
                nome_arq = meta["arquivo"]
                arquivos_info[nome_arq] = arquivos_info.get(nome_arq, 0) + 1
    except Exception:
        pass

    return dict(sorted(arquivos_info.items()))


def excluir_pdf_especifico(materia: str, nome_arquivo: str) -> tuple[int, bool]:
    """Remove os vetores de um PDF específico do ChromaDB e apaga o arquivo físico local."""
    paginas_removidas = 0
    try:
        dados = collection.get(
            where={"$and": [{"materia": materia}, {"arquivo": nome_arquivo}]},
            include=[]
        )
        ids_para_deletar = dados.get("ids", [])
        if ids_para_deletar:
            collection.delete(ids=ids_para_deletar)
            paginas_removidas = len(ids_para_deletar)
    except Exception as e:
        st.error(f"Erro ao remover do banco vetorial: {e}")

    arquivo_apagado = False
    caminho_fisico = os.path.join(PASTA_MATERIAS, materia, nome_arquivo)
    if os.path.exists(caminho_fisico):
        try:
            os.remove(caminho_fisico)
            arquivo_apagado = True
        except Exception as e:
            st.error(f"Erro ao apagar arquivo local: {e}")

    return paginas_removidas, arquivo_apagado


# =====================================================================
# 3. FERRAMENTAS DO AGENTE (FUNCTION CALLING)
# =====================================================================

def gerar_lista_exercicios(tema: str, quantidade: int = 5, materia: str = "") -> str:
    """
    Busca conteúdos no material da faculdade sobre um tema e retorna o contexto
    para criar uma lista de exercícios ou simulado com gabarito comentado.

    Args:
        tema: O assunto ou tópico da matéria para focar nas questões (ex: 'Árvore Rubro-Negra', 'Scrum').
        quantidade: Número de questões desejadas (padrão: 5).
        materia: Nome da matéria para filtrar (opcional).
    """
    st.toast(f"📝 Buscando material para {quantidade} questões sobre '{tema}'...")
    filtro = {"materia": materia} if materia and materia != "Todas as matérias" else None
    
    resultados = collection.query(
        query_texts=[tema],
        n_results=8,
        where=filtro
    )
    
    if not resultados["documents"] or not resultados["documents"][0]:
        return "Nenhum conteúdo encontrado no banco para gerar exercícios sobre esse tema."

    trechos = "\n---\n".join(resultados["documents"][0])
    return f"MATERIAL BASE PARA CRIAR {quantidade} QUESTÕES SOBRE '{tema}':\n{trechos}"


# =====================================================================
# 4. BARRA LATERAL (FILTRO, UPLOAD & GERENCIADOR DE PDFs)
# =====================================================================
with st.sidebar:
    st.markdown("### 🎓 Painel de Estudos")
    metric_placeholder = st.empty()
    metric_placeholder.metric("Páginas Indexadas (ChromaDB)", f"{collection.count()} págs")

    st.divider()
    materias_lista = listar_materias_disponiveis()
    opcoes_filtro = ["Todas as matérias"] + materias_lista
    materia_selecionada = st.selectbox("📚 Filtrar Matéria Ativa", opcoes_filtro)

    st.divider()
    st.subheader("📤 Adicionar Novos Slides / PDFs")
    destino_opcao = st.radio(
        "Destino do arquivo",
        ["Matéria existente", "Criar nova matéria"],
        horizontal=True
    )

    if destino_opcao == "Matéria existente" and materias_lista:
        materia_upload = st.selectbox("Selecione a pasta", materias_lista)
    else:
        materia_upload = st.text_input("Nome da nova matéria", placeholder="Ex: ED2")

    arquivos_enviados = st.file_uploader(
        "Arraste seus PDFs aqui",
        type=["pdf"],
        accept_multiple_files=True
    )

    acao_salvar = st.button("⚡ Salvar e Indexar PDFs", use_container_width=True)
    acao_reindexar = st.button("🔄 Reindexar Pasta Local", use_container_width=True)

    # Novo Bloco: Visualizar e Excluir PDFs Específicos
    st.divider()
    with st.expander("🗂️ Gerenciar / Excluir PDFs", expanded=False):
        if not materias_lista:
            st.caption("Nenhuma matéria cadastrada ainda.")
        else:
            mat_gerenciar = st.selectbox(
                "Matéria",
                materias_lista,
                key="select_mat_gerenciar"
            )
            arquivos_dict = listar_arquivos_da_materia(mat_gerenciar)

            if not arquivos_dict:
                st.caption("Nenhum PDF nesta matéria.")
            else:
                opcoes_pdfs = list(arquivos_dict.keys())
                pdf_escolhido = st.selectbox(
                    "Selecione o PDF",
                    opcoes_pdfs,
                    format_func=lambda nome: f"{nome} ({arquivos_dict[nome]} págs)",
                    key="select_pdf_excluir"
                )
                if st.button("🗑️ Excluir PDF Selecionado", use_container_width=True):
                    pags_del, arq_del = excluir_pdf_especifico(mat_gerenciar, pdf_escolhido)
                    st.session_state["feedback_exclusao"] = (
                        f"🗑️ PDF **{pdf_escolhido}** removido de **{mat_gerenciar}** "
                        f"({pags_del} páginas apagadas do banco)."
                    )
                    st.rerun()

    st.divider()
    if st.button("🧹 Limpar Histórico do Chat", use_container_width=True):
        st.session_state.mensagens_tutor = []
        st.rerun()

# =====================================================================
# 5. CABEÇALHO PRINCIPAL & MONITOR DE INDEXAÇÃO
# =====================================================================
st.markdown(f"""
<div class="tutor-header">
    <div>
        <h1 class="tutor-title">TUTOR ACADÊMICO <span>RAG MULTIMODAL</span></h1>
        <div class="tutor-subtitle">Monitoria Inteligente baseada nos seus Slides, Apostilas e Diagramas</div>
    </div>
    <div class="tutor-badge">📖 Foco: {materia_selecionada}</div>
</div>
""", unsafe_allow_html=True)

# Exibe confirmação de exclusão caso um PDF tenha acabado de ser removido
if st.session_state.get("feedback_exclusao"):
    st.info(st.session_state.pop("feedback_exclusao"))

# Executa a indexação na área principal da tela
if acao_salvar:
    if not arquivos_enviados:
        st.error("⚠️ Selecione ou arraste pelo menos um arquivo `.pdf` na barra lateral antes de clicar em Salvar.")
    elif not materia_upload or not materia_upload.strip():
        st.error("⚠️ Digite o nome da matéria (ex: `ED2`) antes de clicar em Salvar.")
    else:
        pasta_destino = os.path.join(PASTA_MATERIAS, materia_upload.strip())
        os.makedirs(pasta_destino, exist_ok=True)
        for arq in arquivos_enviados:
            caminho_salvo = os.path.join(pasta_destino, arq.name)
            with open(caminho_salvo, "wb") as f:
                f.write(arq.getbuffer())
        executar_indexacao_visual(PASTA_MATERIAS)
        metric_placeholder.metric("Páginas Indexadas (ChromaDB)", f"{collection.count()} págs")

if acao_reindexar:
    executar_indexacao_visual(PASTA_MATERIAS)
    metric_placeholder.metric("Páginas Indexadas (ChromaDB)", f"{collection.count()} págs")

if "mensagens_tutor" not in st.session_state:
    st.session_state.mensagens_tutor = []

aba_duvidas, aba_simulado = st.tabs(["💬 Monitoria & Tirar Dúvidas", "📝 Gerador de Simulados & Exercícios"])

# --- ABA 1: CHAT RAG COM CITAÇÕES ---
with aba_duvidas:
    c1, c2, c3 = st.columns(3)
    atalho = None
    if c1.button("📌 Resumir Tópicos Principais", use_container_width=True):
        atalho = "Faça um resumo estruturado em tópicos dos conceitos mais importantes presentes no material desta matéria."
    if c2.button("🧠 Explicar com Analogia Prática", use_container_width=True):
        atalho = "Explique o conceito central destes slides de forma didática usando uma analogia do mundo real."
    if c3.button("⚡ 3 Questões Rápidas de Revisão", use_container_width=True):
        atalho = "Crie 3 exercícios de revisão estilo prova sobre os temas principais deste material com gabarito comentado."

    for msg in st.session_state.mensagens_tutor:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("fontes"):
                with st.expander("📚 Fontes consultadas nos seus PDFs"):
                    for f in msg["fontes"]:
                        st.markdown(f"- `{f}`")

    pergunta_input = st.chat_input("Pergunte algo sobre os slides ou peça exercícios de um tópico...")
    prompt_final = atalho or pergunta_input

    if prompt_final:
        st.session_state.mensagens_tutor.append({"role": "user", "content": prompt_final})
        with st.chat_message("user"):
            st.markdown(prompt_final)

        with st.chat_message("assistant"):
            with st.spinner("Consultando banco vetorial e analisando slides..."):
                if collection.count() == 0:
                    st.error("O banco vetorial está vazio! Adicione PDFs na barra lateral primeiro.")
                else:
                    filtro = None
                    if materia_selecionada != "Todas as matérias":
                        filtro = {"materia": materia_selecionada}
                    else:
                        for mat in materias_lista:
                            if mat.lower() in prompt_final.lower():
                                filtro = {"materia": mat}
                                break

                    resultados = collection.query(
                        query_texts=[prompt_final],
                        n_results=5,
                        where=filtro
                    )

                    docs_encontrados = resultados["documents"][0] if resultados["documents"] else []
                    metas_encontradas = resultados["metadatas"][0] if resultados["metadatas"] else []

                    trechos = "\n---\n".join(docs_encontrados)
                    fontes_formatadas = sorted(list({
                        f"[{m['materia']}] {m['arquivo']} — pág. {m['pagina']}"
                        for m in metas_encontradas
                    }))
                    fontes_str = ", ".join(fontes_formatadas)

                    prompt_sistema = """
                    Você é um monitor universitário de Ciência da Computação.
                    Responda à dúvida do aluno usando APENAS os trechos de material fornecidos.
                    Se a resposta não estiver no material, diga explicitamente que não encontrou nos slides.
                    Seja didático, objetivo e estruture a resposta em tópicos quando ajudar na revisão.
                    Se o aluno pedir exercícios, simulados ou questões de revisão, use a ferramenta 'gerar_lista_exercicios'
                    e entregue questões bem elaboradas com o gabarito comentado no final.
                    """

                    conteudo_usuario = f"""
                    MATÉRIA ATUAL: {materia_selecionada}
                    MATERIAL DE APOIO (Fontes: {fontes_str}):
                    {trechos}

                    PEDIDO DO ALUNO:
                    {prompt_final}
                    """

                    modelos_reserva = [
                        "gemini-3.5-flash-lite",
                        "gemini-3.7-flash",
                        "gemini-2.5-flash",
                        "gemini-3.8-flash",
                    ]

                    resposta_texto = None
                    for modelo in modelos_reserva:
                        try:
                            chat = client.chats.create(
                                model=modelo,
                                config=types.GenerateContentConfig(
                                    system_instruction=prompt_sistema,
                                    temperature=0.3,
                                    tools=[gerar_lista_exercicios],
                                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                        maximum_remote_calls=10
                                    )
                                )
                            )
                            resp = chat.send_message(conteudo_usuario)
                            if resp and resp.text:
                                resposta_texto = resp.text
                                break
                        except Exception as e:
                            st.warning(f"Modelo {modelo} indisponível ({e}), tentando próximo...")

                    if resposta_texto:
                        st.markdown(resposta_texto)
                        if fontes_formatadas:
                            with st.expander("📚 Fontes consultadas nos seus PDFs"):
                                for f in fontes_formatadas:
                                    st.markdown(f"- `{f}`")

                        st.session_state.mensagens_tutor.append({
                            "role": "assistant",
                            "content": resposta_texto,
                            "fontes": fontes_formatadas
                        })
                    else:
                        st.error("Nenhum modelo respondeu agora. Aguarde alguns segundos e tente novamente.")

# --- ABA 2: GERADOR ESTRUTURADO DE SIMULADOS ---
with aba_simulado:
    st.subheader("📝 Gerar Lista de Exercícios Personalizada")
    with st.form("form_simulado"):
        col_a, col_b = st.columns([2, 1])
        tema_simulado = col_a.text_input(
            "Tema ou Tópico da Prova",
            placeholder="Ex: Árvores Balanceadas AVL, Processos de Software, Ponteiros em C..."
        )
        qtd_questoes = col_b.slider("Número de Questões", min_value=2, max_value=10, value=5)
        
        col_c, col_d = st.columns(2)
        tipo_questao = col_c.selectbox(
            "Formato das Questões",
            ["Múltipla Escolha (A, B, C, D)", "Discursivas / Código", "Misto (Objetivas + Discursivas)"]
        )
        nivel_dificuldade = col_d.selectbox(
            "Nível de Dificuldade",
            ["Graduação (Padrão Prova)", "Fixação (Conceitos Básicos)", "Desafio (Casos de Borda & Análise)"]
        )

        gerar_simulado_btn = st.form_submit_button("🚀 Gerar Simulado com Gabarito")

    if gerar_simulado_btn and tema_simulado.strip():
        if collection.count() == 0:
            st.error("Indexe pelo menos um PDF na barra lateral antes de gerar exercícios.")
        else:
            with st.spinner("Extraindo trechos relevantes e elaborando questões..."):
                contexto_base = gerar_lista_exercicios(
                    tema=tema_simulado,
                    quantidade=qtd_questoes,
                    materia=materia_selecionada
                )

                prompt_prova = f"""
                Você é um professor universitário de Ciência da Computação.
                Com base EXCLUSIVAMENTE no material abaixo, elabore um simulado com {qtd_questoes} questões sobre '{tema_simulado}'.
                - Formato: {tipo_questao}
                - Nível: {nivel_dificuldade}
                - Ao final, inclua uma seção separada '## ✅ Gabarito Comentado' explicando o porquê da resposta correta de cada questão.

                {contexto_base}
                """

                modelos_reserva = ["gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-2.5-flash"]
                simulado_md = None
                for m in modelos_reserva:
                    try:
                        r = client.models.generate_content(
                            model=m,
                            contents=prompt_prova,
                            config=types.GenerateContentConfig(temperature=0.35)
                        )
                        if r and r.text:
                            simulado_md = r.text
                            break
                    except Exception:
                        pass

                if simulado_md:
                    st.session_state["ultimo_simulado"] = simulado_md
                else:
                    st.error("Falha ao gerar o simulado agora. Tente novamente em alguns segundos.")

    if st.session_state.get("ultimo_simulado"):
        st.divider()
        st.markdown(st.session_state["ultimo_simulado"])
        st.download_button(
            label="📥 Baixar Simulado e Gabarito (.MD)",
            data=st.session_state["ultimo_simulado"].encode("utf-8"),
            file_name="simulado_revisao.md",
            mime="text/markdown"
        )