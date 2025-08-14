import streamlit as st
import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 챗봇 제목 설정
st.title("RAG 기반 챗봇")

# 1. 임베딩 모델 및 LLM 로드 (단 한 번만 실행)
@st.cache_resource
def load_models():
    """
    RAG 시스템에 필요한 모델과 벡터 DB를 로드합니다.
    """
    # Hugging Face 토큰을 secrets에서 가져옵니다.
    # 'HUGGINGFACEHUB_API_TOKEN'은 secrets.toml에 설정한 키와 일치해야 합니다.
    hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    embedding_model_name = "jhgan/ko-sbert-nli"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # 수정: from_pretrained 함수에 토큰을 명시적으로 전달합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # 수정: from_pretrained 함수에 토큰을 명시적으로 전달합니다.
        token=hf_token
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.5,
        top_p=0.95,
        # 수정: pipeline에도 토큰을 전달합니다.
        token=hf_token
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return llm, vector_db

# 2. RAG 체인 설정 (단 한 번만 실행)
# 수정: llm과 vector_db 인자명 앞에 모두 밑줄('_')을 붙여 캐싱에서 제외합니다.
@st.cache_resource
def setup_rag_chain(_llm, _vector_db):
    """
    로드된 모델과 벡터 DB를 사용하여 RAG 체인을 설정합니다.
    """
    prompt_template = """
    다음 문맥(Context)을 사용하여 질문에 답변해주세요.
    만약 문맥에서 답을 찾을 수 없다면, 모른다고 답해주세요.

    문맥(Context): {context}
    질문(Question): {question}

    답변:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        retriever=_vector_db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- 메인 애플리케이션 로직 ---

llm, vector_db = load_models()
rag_chain = setup_rag_chain(llm, vector_db)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            full_response = rag_chain.invoke({"query": prompt})["result"]

            answer_prefix = "답변:"
            if answer_prefix in full_response:
                final_response = full_response.split(answer_prefix, 1)[1].strip()
            else:
                final_response = full_response.strip()

            st.markdown(final_response)
    
    st.session_state.messages.append({"role": "assistant", "content": final_response})