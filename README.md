# 🎓 Tutor Acadêmico RAG Multimodal

Assistente de estudos universitários baseado em **Retrieval-Augmented Generation (RAG)** multimodal, desenvolvido em **Python**, **ChromaDB**, **PyMuPDF** e **Google Gemini API (`google-genai`)**.

## 🚀 Funcionalidades
- **Pipeline Multimodal Híbrido:** Extrai texto nativo de slides e apostilas em PDF com PyMuPDF (`fitz`) e aciona visão computacional via Gemini Vision automaticamente em páginas com diagramas, fluxogramas ou pouco texto (`< 60` caracteres).
- **Banco Vetorial Persistente:** Indexação local no ChromaDB com metadados por disciplina, arquivo e número da página para respostas fundamentadas e com citação exata das fontes.
- **Gerenciamento Granular:** Upload de múltiplos PDFs por matéria e exclusão cirúrgica de documentos específicos do banco vetorial.
- **Gerador de Simulados:** Criação automatizada de provas práticas (múltipla escolha, discursivas ou mistas) com gabarito comentado e exportação em Markdown.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.13
- **IA & Visão:** Google GenAI SDK (`gemini-3.8-flash` / `gemini-3.5-flash-lite`)
- **Banco Vetorial & PDF:** ChromaDB, PyMuPDF (`fitz`)
- **Interface:** Streamlit

## ⚙️ Como Executar Localmente
1. Clone o repositório:
   ```bash
   git clone [https://github.com/itsyann/tutor-academico-rag.git](https://github.com/itsyann/tutor-academico-rag.git)
   cd tutor-academico-rag