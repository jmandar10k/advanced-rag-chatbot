from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from dotenv import load_dotenv
import tempfile, os, warnings

warnings.filterwarnings("ignore")
load_dotenv()

# -------------------- GLOBALS --------------------
vector_store = None
bm25_retriever = None
multiquery_retriever = None

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

llm = ChatGroq(model='llama-3.1-8b-instant')
flashrank = FlashrankRerank(top_n=3)

# -------------------- LOAD + CHUNK --------------------
def load_and_chunk(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "]
    )

    return splitter.split_documents(documents)

# -------------------- BUILD RETRIEVERS --------------------
def build_retrievers(chunks):
    global vector_store, bm25_retriever, multiquery_retriever

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "lambda_mult": 0.5}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="Generate 3 variations of the given question.\nQuestion: {question}"
    )

    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_retriever,
        llm=llm,
        prompt=QUERY_PROMPT
    )

# -------------------- HYBRID RETRIEVER --------------------
def hybrid_retriever(query):
    docs_vector = multiquery_retriever.invoke(query)
    docs_bm25 = bm25_retriever.invoke(query)

    all_docs = docs_vector + docs_bm25

    seen, unique_docs = set(), []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    return unique_docs

# -------------------- FORMAT --------------------
def format_docs(docs):
    seen, final = set(), []
    for doc in docs:
        text = doc.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            final.append(text)
    return "\n\n".join(final)

# -------------------- PROMPT --------------------
prompt_template = PromptTemplate(
    template="""
You are a helpful AI assistant.

Answer ONLY using the given context.

Instructions:
- Use chat history for better understanding
- Be clear and natural
- If not found, say "I don't know"

Chat History:
{history}

Context:
{context}

Question:
{query}
""",
    input_variables=['query', 'context', 'history']
)

# -------------------- MAIN FUNCTION --------------------
def get_rag_response(query, chat_history):

    docs = hybrid_retriever(query)
    docs = flashrank.compress_documents(docs, query)

    context = format_docs(docs)

    history_text = "\n".join(
        [f"{msg['role']}: {msg['message']}" for msg in chat_history[-6:]]
    )

    final_prompt = prompt_template.format(
        query=query,
        context=context,
        history=history_text
    )

    response = llm.invoke(final_prompt)

    return response.content, docs