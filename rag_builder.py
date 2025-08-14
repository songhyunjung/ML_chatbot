import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. 임베딩 모델 로드
# 선택한 한국어 임베딩 모델을 불러옵니다.
embedding_model_name = "jhgan/ko-sbert-nli"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# 2. Vector DB 생성 및 저장
def create_and_save_vector_db(texts):
    """
    주어진 텍스트 덩어리들을 벡터화하여 FAISS Vector DB를 생성하고 저장합니다.
    """
    vector_db = FAISS.from_texts(texts, embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

# 3. RAG 체인 구축
def setup_rag_chain(vector_db):
    """
    Retrieval-Augmented Generation (RAG) 체인을 설정합니다.
    """
    # 수정: HuggingFacePipeline을 사용하여 로컬에서 LLM을 로드합니다.
    # 이 방식은 Hugging Face API 의존성을 줄이고, 로컬 GPU를 직접 활용합니다.
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # 토크나이저와 모델을 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 텍스트 생성 파이프라인을 구축합니다.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.5,
        top_p=0.95
    )
    
    # HuggingFacePipeline을 langchain의 LLM으로 사용합니다.
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template = """
    다음 문맥(Context)을 사용하여 질문에 답변해주세요.
    만약 문맥에서 답을 찾을 수 없다면, 모른다고 답해주세요.

    문맥(Context): {context}
    질문(Question): {question}

    답변:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# 예시 실행
if __name__ == "__main__":
    dummy_texts = [
        "머신러닝은 인공지능의 한 분야로, 데이터로부터 학습하여 예측을 수행하는 알고리즘을 개발하는 학문이다.",
        "딥러닝은 머신러닝의 하위 분야로, 인공 신경망을 사용하여 복잡한 패턴을 학습한다.",
        "자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 생성하도록 하는 기술이다.",
        "지도학습은 정답 라벨이 있는 데이터를 사용하여 모델을 훈련시키는 방법이다. 대표적인 예로 분류와 회귀가 있다.",
        "비지도학습은 정답 라벨이 없는 데이터를 사용하여 데이터의 구조나 패턴을 찾는 방법이다. 클러스터링이 대표적인 예이다.",
        "강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동을 학습하는 방법이다. 게임 AI에 주로 사용된다."
    ]

    print("Creating and saving Vector DB...")
    vector_db = create_and_save_vector_db(dummy_texts)
    print("Vector DB created and saved successfully.")

    print("Setting up RAG chain...")
    rag_chain = setup_rag_chain(vector_db)
    print("RAG chain setup complete.")

    question = "지도학습과 비지도학습의 차이점은 무엇인가요?"
    print(f"\nQuestion: {question}")
    answer = rag_chain.invoke({"query": question})
    print(f"Answer: {answer['result']}")