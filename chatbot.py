import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

DB_FAISS_PATH = "vectorstore/db_faiss"
UPLOAD_FOLDER = "uploaded_docs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def process_and_add_to_vectorstore(file_path, vectorstore, embedding_model):
    ext = file_path.split('.')[-1].lower()

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    else:
        st.sidebar.error("Unsupported file type.")
        return

    try:
        with st.sidebar:
            with st.spinner("Processing document and adding to vectorstore..."):
                documents = loader.load()
                vectorstore.add_documents(documents)
                vectorstore.save_local(DB_FAISS_PATH)
        st.sidebar.success(f"Document '{os.path.basename(file_path)}' successfully added to vectorstore!")
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")


def format_response(result, source_documents):
    formatted_response = f"**Answer:**\n{result}\n\n"
    if source_documents:
        formatted_response += "**Source Documents:**\n"
        for i, doc in enumerate(source_documents, 1):
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', 'Unknown page')
            content = ' '.join(doc.page_content.strip().split())
            if len(content) > 200:
                content = content[:200] + "..."
            formatted_response += f"\n**Source {i}:**\n"
            formatted_response += f"- **File:** {source.split('/')[-1] if '/' in source else source}\n"
            formatted_response += f"- **Page:** {page}\n"
            formatted_response += f"- **Content:** {content}\n"
    return formatted_response


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.5
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    return llm


def main():
    st.title("Ask Chatbot!")

    st.sidebar.title("Upload Documents")
    uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.info(f"File '{uploaded_file.name}' uploaded successfully!")

        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = get_vectorstore()
        process_and_add_to_vectorstore(save_path, vectorstore, embedding_model)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't make up an answer.
        Only respond with what's in the context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )


            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            formatted_response = format_response(result, source_documents)

            st.chat_message('assistant').markdown(formatted_response)
            st.session_state.messages.append({'role': 'assistant', 'content': formatted_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()