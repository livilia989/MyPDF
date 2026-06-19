import streamlit as st
import fitz

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# OpenAI API Key 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="보험 약관 AI 분석기",
    page_icon="📄",
    layout="wide"
)

st.title("📄 보험 약관 AI 분석기")
st.write("보험 약관 PDF를 업로드하고 질문해보세요.")

# PDF 업로드
uploaded_file = st.file_uploader(
    "보험 약관 PDF 업로드",
    type=["pdf"]
)

# 사이드바
with st.sidebar:

    st.header("질문 예시")

    st.write("✔ 월 납입 보험료는 얼마인가?")
    st.write("✔ 암 진단금은 얼마인가?")
    st.write("✔ 보장 내용을 요약해줘")
    st.write("✔ 해약환급금은 얼마인가?")
    st.write("✔ 보험기간은 몇 년인가?")

# 벡터DB 생성
@st.cache_resource
def build_vectorstore(pdf_bytes):

    pdf_doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in pdf_doc:
        text += page.get_text()
        text += "\n\n"

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    vectorstore = FAISS.from_texts(
        chunks,
        OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
    )

    return vectorstore

# 질문 입력
question = st.text_input(
    "질문을 입력하세요",
    placeholder="예) 암보험의 월 납입 보험료는 얼마인가?"
)

# 분석 버튼
if st.button("🔍 분석하기"):

    if uploaded_file is None:
        st.warning("PDF 파일을 업로드하세요.")
        st.stop()

    if not question:
        st.warning("질문을 입력하세요.")
        st.stop()

    with st.spinner("보험 약관 분석 중..."):

        vectorstore = build_vectorstore(
            uploaded_file.read()
        )

        docs = vectorstore.similarity_search_with_score(
            question,
            k=5
        )

        context = ""

        for doc, score in docs:
            context += doc.page_content
            context += "\n\n"

        prompt = ChatPromptTemplate.from_template(
            """
다음 보험약관을 참고하여 질문에 답변해주세요.

보험약관
{context}

========================

질문
{question}

답변 형식

1. 결론
2. 상세설명
3. 근거
"""
        )

        llm = ChatOpenAI(
            model="gpt-5.5",
            temperature=0
        )

        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

    st.success("분석 완료")

    st.subheader("📌 답변")
    st.write(answer)

    with st.expander("📚 검색된 근거 보기"):

        for idx, (doc, score) in enumerate(docs):

            st.markdown(f"### 근거 {idx + 1}")
            st.write(doc.page_content)
            st.divider()
