import sys
import streamlit as st
import time
from pathlib import Path
import json
import pandas as pd
import re
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from konlpy.tag import Okt
import spacy
import seaborn as sns
import matplotlib.ticker as mticker

from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from citationlinker import CitationLinker  
from prompt import * 

st.set_page_config(page_title="📃 CitationLinkerGPT", page_icon="📃")

@st.cache_resource(show_spinner='Embedding file....')
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_path}")
    splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 600,
        chunk_overlap= 50,
    )

    # 해당 loader는 pdf, txt, docx와 모두 호환됩니다.
    # 여기가 특화 데이터셋입니다.
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 20})
    return retriever

def save_message(message, role):
    st.session_state['messages'].append({'message':message,'role':role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def paint_history():
    for message in st.session_state['messages']:
        send_message(message['message'], message['role'], save=False)

def file_exists(file_path):
    return os.path.isfile(file_path)

def load_config():
    config_path = "config/config.json"
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "arxiv_id": "",
        "essay_dir": "reference",
        "model": "gpt-4o-mini",
        "preprocess_threhsold": 25,
        "reference_ratio": 0.2,
        "reference_condition": "- 인용 표시: 논문 본문에서 참고문헌이 **[숫자]** 형식 ([5], [27] 등)으로 인용된 경우만 추출하세요",
        "result_dir": "result",
        "content_keys": {}
    }

def load_policy_config():
    config_path = "config/config-policy.json"
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return json.load(f)

def save_config(config, path):
    with open(path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def load_json(file_path):
    if Path(file_path).exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return None

def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def generate_wordcloud(text):
    wordcloud = WordCloud(
        font_path="font/NanumGothic-Regular.ttf",
        width=900,
        height=450,
        max_words=100,  # 표시할 최대 단어 수
        background_color="white",  # 배경색 설정
        colormap="Set2",  # 컬러맵 적용
        relative_scaling=0.3,  # 글자 크기 조정
        contour_color="steelblue",  # 테두리 색상
        contour_width=2  # 테두리 두께
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off") 
    return fig

def extract_nouns(text):
    """
    입력된 텍스트에서 한글 및 영어 명사만 추출하는 함수
    """
    # 한글 명사 추출
    korean_nouns = okt.nouns(text)
    korean_nouns = [noun for noun in korean_nouns if len(noun)>=2]

    # 영어 단어만 필터링 (정규식)
    english_text = " ".join(re.findall(r"[a-zA-Z]+", text))

    # 영어 명사 추출
    doc = nlp(english_text)
    english_nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    # 한글과 영어 명사 결합
    nouns = korean_nouns + english_nouns
    return nouns

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.schema import LLMResult
from pydantic import Field
from typing import List, Optional
import torch

# 아래 클래스는 허깅페이스의 모델 사용법을 그대로 참조하여 클래스화 시킨 것입니다.
class KoreanLlamaPipeline:
    def __init__(self, model, tokenizer, max_new_tokens=1024, temperature=0.6, top_p=0.9):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, instruction, **kwargs):
        messages = [{"role": "user", "content": instruction}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            **kwargs
        )

        # Decode the output
        return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

# 아래는 아무리 해도 문서를 제대로 찾을 수 없어서 chat gpt에게 코드를 짜달라고 했습니다.
# langchain과 huggingface 라이브러리를 연결하기 위한 클래스여서 huggingfacepipeline을 상속받아야 한다고 전달받았습니다.
# 반드시 _generate 함수와 stop 토큰을 처리하는 아래와 같은 로직이 필요하다고 전달을 받았습니다.
# 다만 제가 구현한 것이 아니라 헨들링을 잘 하기는 어렵습니다.
class KoreanLlamaLangChainLLM(HuggingFacePipeline):
    pipeline: KoreanLlamaPipeline = Field()

    def _generate(self, prompt: str, stop: Optional[List[str]] = None) -> LLMResult:
        response = self.pipeline(prompt)
        if stop:
            for token in stop:
                response = response.split(token)[0]
        return LLMResult(generations=[[{"text": response}]])

okt = Okt()  
nlp = spacy.load("en_core_web_sm") 

config = load_config()
config['arxiv_id'] = st.sidebar.text_input("🔍 arXiv 논문 ID 입력", config.get("arxiv_id", ""))
config['essay_dir'] = f"reference/{config['arxiv_id'].replace('.','')}"
config['result_dir'] = f"result/{config['arxiv_id'].replace('.','')}"
config['model'] = st.sidebar.selectbox("🧠 사용할 모델", ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"], index=["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"].index(config.get("model", "gpt-4o-mini")))
config['preprocess_threhsold'] = st.sidebar.slider("📏 최소 문장 길이", 10, 100, config.get("preprocess_threhsold", 25))
config['reference_ratio'] = st.sidebar.slider("📊 참고문헌 분석 비율", 0.1, 1.0, config.get("reference_ratio", 0.2))
config['reference_condition'] = st.sidebar.text_area("🔗 참고문헌 인용 조건", config.get("reference_condition", ""))

content_keys_dict = config.get("content_keys", {})
content_keys_list = []
for key, value in content_keys_dict.items():
    content_keys_list.append({"ID": key, "name": value["name"], "forward_deliminators": value["deliminators"]["forward"], "backward_deliminators": value["deliminators"]["backward"]})

content_keys_df = pd.DataFrame(content_keys_list)
edited_df = st.sidebar.data_editor(content_keys_df, num_rows="dynamic")
new_content_keys = {}
for i, row in edited_df.iterrows():
    new_content_keys[str(i+1)] = {
        "name": row["name"],
        "deliminators": {
            "forward": row["forward_deliminators"],
            "backward": row["backward_deliminators"]
        }
    }
config['content_keys'] = new_content_keys

st.title("📃 CitationLinkerGPT")
st.sidebar.header("Config Setting")
st.sidebar.subheader("📑 논문 섹션 설정")

analyze_essay_tab, basic_review_tab, chatbot_tab = st.tabs(['Analyze Essay', 'Review based on References', "CitationGPT"])

if st.sidebar.button("💾 설정 저장"):
    config_path = f"config/config-{config['arxiv_id'].replace('.','')}.json"
    if file_exists(config_path):
        st.sidebar.warning("⚠️ 이미 분석이 끝난 논문입니다. '확인 탭'에서 arXiv ID를 입력하세요.")

    else:
        CitationLinker._create_directory_if_not_exists(config['essay_dir'])
        CitationLinker._create_directory_if_not_exists(config['result_dir'])
        save_config(config, path=config_path)
        st.sidebar.success("설정이 저장되었습니다!")

with analyze_essay_tab:
    if st.button("🚀 논문 분석 시작"):
        if not config['arxiv_id']:
            st.warning("❗ 논문 ID를 입력하세요!")
        else:
            citation_linker = CitationLinker(config)

            with st.status("논문 분석을 시작합니다...", expanded=True) as status:
                try:
                    st.write("📥 **1단계: 논문 검색 및 다운로드 중...**")
                    _, elapsed_time = measure_time(
                        citation_linker._search_and_download_essay,
                        arxiv_id=config['arxiv_id']
                    )
                    # To Do: 종료 후 citation_linker.title / citation_linker.authors의 값이 논문 다운로드 완료 하단에 표시되어야 함함
                    st.success(f"✅ 논문 다운로드 완료! 🕒 소요 시간: {elapsed_time:.2f}초")

                    st.write("📃 **2단계: 논문 전처리 진행 중...**")
                    processed_output, elapsed_time = measure_time(
                        citation_linker._preprocess,
                        save_path=citation_linker.essay_dir / f"0-{citation_linker.title}.pdf"
                    )
                    # To Do: config['preprocess_threhsold'] 이하의 text는 모두 삭제되었음을 논문 전처리 완료 하단에 표시
                    st.success(f"✅ 논문 전처리 완료! 🕒 소요 시간: {elapsed_time:.2f}초")

                    st.write("📝 **3단계: 논문 요약 진행 중...**")
                    summary, elapsed_time = measure_time(
                        citation_linker._basic_summarize,
                        basic_summarize_template=basic_summarize_template,
                        processed_output=processed_output
                    )
                    with open(citation_linker.result_dir / "basic_summary.json", 'w', encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False, indent=4)
                    # To Do: Summary의 print 결과를 Toggle로 보여줄 수 있어야 함
                    st.success(f"✅ 논문 요약 완료! 🕒 소요 시간: {elapsed_time:.2f}초")

                    st.write("📑 **4단계: 참고문헌 목록화 중...**")
                    (processed_output, reference_dict), elapsed_time = measure_time(
                        citation_linker._reference_preprocess,
                        reference_extraction_template=reference_extraction_template,
                        processed_output=processed_output
                    )
                    # To Do: reference_dict의 key, title, author를 표 형태로 보여줘야 하고 결과를 Toggle로 보여줄 수 있어야 함
                    st.success(f"✅ 참고문헌 목록화 완료! 🕒 소요 시간: {elapsed_time:.2f}초")

                    st.write("📑 **5단계: 참고문헌 인용 횟수 세는 중...**")
                    processed_output, elapsed_time = measure_time(
                        citation_linker._reference_counting,
                        reference_count_template_dict=reference_count_template_dict,
                        processed_output=processed_output,
                        reference_dict=reference_dict
                    )
                    
                    st.success(f"✅ 횟수 세기 완료! 🕒 소요 시간: {elapsed_time:.2f}초")

                    st.write("📊 **6단계: 인용 논문 다운로드 중...**")
                    (related_reference, total_related_reference, processed_output), elapsed_time = measure_time(
                        citation_linker._download_reference,
                        processed_output=processed_output
                    )
                    
                    with open(citation_linker.result_dir / "reference_count.json", 'w', encoding="utf-8") as f:
                        json.dump(total_related_reference, f, ensure_ascii=False, indent=4)
                    st.success(f"✅ 인용 논문 다운로드 완료! 🕒 소요 시간: {elapsed_time:.2f}초")

                    st.write("📊 **7단계: 인용 논문과의 접점 세밀화 중...**")
                    related_reference, elapsed_time = measure_time(
                        citation_linker._reduce_questions,
                        question_reduction_template=question_reduction_template,
                        related_reference=related_reference
                    )
                    st.success(f"✅ 인용 논문과의 접점 세밀화 완료! 🕒 소요 시간: {elapsed_time:.2f}초")

                    st.write("📊 **8단계: 연구 확장성 파악 중...**")
                    related_reference, elapsed_time = measure_time(
                        citation_linker._find_connection_from_reference,
                        reference_qna_template=reference_qna_template,
                        # research_progress_template=research_progress_template,
                        # processed_output=processed_output,
                        related_reference=related_reference
                    )
                    with open(citation_linker.result_dir / "reference_qna.json", 'w', encoding="utf-8") as f:
                        json.dump(related_reference, f, ensure_ascii=False, indent=4)
                    # To Do: related_references.items()를 for문 돌리면서 value['summary']와  value['summary_qna]'결과를 모두 print하여여 toggle로 보여줄 수 있어야 함.
                    st.success(f"✅ 연구 확장성 분석 완료! 🕒 소요 시간: {elapsed_time:.2f}초")
                    
                    status.update(label="🎉 논문 분석이 완료되었습니다!", state="complete")
                    # Check Review 탭으로 이동 안내 메시지 추가
                    st.info(f"✅ 전체 논문 분석이 완료되었습니다! 'Check Review' 탭에 {config['arxiv_id']}를 입력하세요.")
                except Exception as e:
                    status.update(label=f"❌ 오류 발생: {e}", state="error")

with basic_review_tab:
    # 사용자 입력 (기본값: config['arxiv_id'])
    arxiv_id = st.text_input("📌 arXiv 논문 ID 입력", value=config.get("arxiv_id", ""))
    arxiv_id = arxiv_id.replace('.','')
    # 버튼 클릭 시 JSON 파일 불러오기
    if st.button("🔍 기본 리뷰 확인"):
        basic_summary_path = Path("result") / arxiv_id / "basic_summary.json"
        basic_summary = load_json(basic_summary_path)
        basic_summary_json = {
            1: basic_summary.split("###")[1],
            2 : basic_summary.split("###")[2],
            3 : basic_summary.split("###")[3],
            4 : basic_summary.split("###")[4],
            5: basic_summary.split("###")[5],
        }
        basic_1 = "###" + basic_summary_json[1]
        basic_2 = "###" + "###".join([basic_summary_json[i] for i in range(2, len(basic_summary_json)+1)])
        nouns = extract_nouns(basic_summary)

        reference_count_path = Path("result") / arxiv_id / "reference_count.json"
        reference_qna_path = Path("result") / arxiv_id / "reference_qna.json"

        reference_count = load_json(reference_count_path)
        reference_qna = load_json(reference_qna_path)
        data_dict = {}
        for i, (key, value_dict) in enumerate(reference_count.items()):
            data_dict[i] = {
                "Title":value_dict["Title"],
                "Counter":value_dict["Counter"],
                "Index":key
            }
        reference_connection_dict = {}
        for key, value_dict in reference_qna.items():
            reference_connection_dict[key] = {
                "Title" : value_dict["Title"], 
                "Index" : key,
                "reference_connection" : value_dict["Summary"]}

        df = pd.DataFrame(data_dict).T
        df["Counter"] = df["Counter"].apply(lambda x: round(x/3, 3))
        max_value = df.iloc[0]["Counter"]
        df['Score'] = (df["Counter"] / max_value).round(2) 

        nums = len(reference_qna)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📖 논문 요약")
            st.write(basic_1)

        with col2:
            st.subheader("🎨 워드 클라우드")
            st.pyplot(generate_wordcloud(" ".join(nouns)))

        st.subheader("🔍 연구 정보")
        st.write(basic_2)
        
        # 🌟 데이터 테이블 표시
        sns.set_theme(style="whitegrid")

        fig, ax = plt.subplots(figsize=(16, 9))
        sns.barplot(x="Index", y="Score", data=df.iloc[:nums], palette="coolwarm", ax=ax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
        ax.set_yticklabels(ax.get_yticks(), fontsize=11)

        # 라벨 추가
        ax.set_xlabel("Title Index", fontsize=13, fontweight='bold')
        ax.set_ylabel("Citation Score", fontsize=13, fontweight='bold')
        ax.set_title("Citation Score of References", fontsize=15, fontweight='bold')

        # 막대 위에 값 표시 (숫자)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight="bold", color="black")
            
        st.markdown("### 📊 데이터 테이블")
        st.dataframe(df, use_container_width=True, height=400)
        st.markdown("### 📉 인용 점수 Plot")
        st.pyplot(fig, use_container_width=True)
        st.subheader("📌 참고 문헌과의 접점 정리")
        for index, content in reference_connection_dict.items():
            with st.expander(f"📖(Index: {index}) {content['Title']}", expanded=False):
                st.write(f"**Reference Summary:**")
                st.text_area("Reference Details", content["reference_connection"], height=300)

with chatbot_tab:
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None
    if "model" not in st.session_state:
        st.session_state["model"] = None

    # arXiv ID 입력
    arxiv_id = st.text_input("📌arXiv 논문 ID 입력", value=config.get("arxiv_id", ""))
    arxiv_id = arxiv_id.replace('.', '')

    # 챗봇 응답 정책 선택

    # 논문 데이터 로드
    basic_summary_path = Path("result") / arxiv_id / "basic_summary.json"
    basic_summary = load_json(basic_summary_path)

    reference_qna_path = Path("result") / arxiv_id / "reference_qna.json"
    reference_qna = load_json(reference_qna_path)

    # 논문 파일 확인 후 임베딩 처리
    target_essay = Path("reference") / arxiv_id
    pdf_files_path = list(target_essay.glob("0*.pdf"))

    if not pdf_files_path:
        st.error("📂 논문 파일을 찾을 수 없습니다.")
        st.stop()
    
    if st.session_state["retriever"] is None:
        st.session_state["retriever"] = embed_file(pdf_files_path[0])
    
    retriever = st.session_state["retriever"]

    if not basic_summary or not reference_qna:
        st.error("📂 필수 데이터가 없습니다. 논문 ID를 확인하세요.")
        st.stop()

    # 요약 텍스트 정리
    merged_text = "\n".join(
        f"{value_dict['Summary']}"
        for value_dict in reference_qna.values()
    )
    policy_config = load_policy_config()
    degree = policy_config['chatbot']

    # 챗봇 응답을 위한 프롬프트 생성
    if degree == "high":
        portion = len(basic_summary)
        prompt_text = basic_summary + "\n" + merged_text[:portion*5]
    elif degree == "middle":
        portion = len(basic_summary)
        prompt_text = basic_summary + "\n" + merged_text[:portion*3]
    else:
        prompt_text = ""

    if st.session_state["model"] is None:

        if degree in ['high', 'middle']:
            llm = ChatOpenAI(temperature=0.1, streaming=True)

        else:
            # 로컬 모델에서 사용할 문장 생성 모델입니다.
            model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )
            model = model.to('cuda')

            custom_pipeline = KoreanLlamaPipeline(model, tokenizer)
            llm = KoreanLlamaLangChainLLM(pipeline=custom_pipeline)
        st.session_state["model"] = llm
    else: 
        llm = st.session_state["model"]

    # ChatPromptTemplate 구성
    prompt = ChatPromptTemplate.from_messages([
        ('system','''당신은 ai 논문에 대하여 수업하는 교수님입니다. 
        아래는 AI를 공부하는 학생과 대화하는 상황을 설계하였습니다.
        세 가지 정보가 제공이 됩니다.
        1. 학생과 이전에 대화했던 내용.
        2. 논문의 요약본
        3. 논문의 본문 일부
         
        학생이 ai에 관한 질문을 했다면 절대로 내용을 지어내지 말고 반드시 이전 대화 내용과 요약문과 본문 일부에서 답을 찾아주세요.
        학생이 ai에 관한 질문을 했는데 이전 대화 내용, 요약문, 본문에서 대답을 찾을 수 없다면 교수님도 잘 모르겠다는 어투로 대답을 해주세요.
        
        학생이 ai와 관련이 없는 일상적인 질문을 한다면, 이전 대화 내용을 참고하여 자유롭게 대답해주세요. 그러면서 수업에 좀 더 집중해야 성적 잘 줄거라고 농담도 함께 던져주세요.
        세부 사항도 곁들여 설명하되 한 번에 6문장 이하로 대답하세요.
        1. 학생과 이전에 대화했던 내용: {context}
        2. 논문의 요약본: {summary}
        3. 논문의 본문 일부: {essay}
        '''),
        ('human', '{question}')
    ])

    # 기본 메시지 추가 (세션 상태를 활용하여 중복 방지)
    if not st.session_state["messages"]:
        st.session_state["messages"].append(
            {"message": "수업 시작하겠습니다. 질문하세요!", "role": "ai"}
        )

    # 기존 메시지 출력
    paint_history()

    # 사용자 입력 받기
    message = st.chat_input('논문에 관한 질문...')
    if message:
        send_message(message, "human")
        context = "\n".join([f"{msg['role']}: {msg['message']}" for msg in st.session_state['messages']])
        # LLM 체인 실행
        chain = {
            'context': RunnableLambda(lambda _: context),
            'summary': RunnableLambda(lambda _: prompt_text),
            'essay': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        } | prompt | llm
        response = chain.invoke(message)
        if degree in ['high','middle']:
            send_message(response.content, 'ai')
        else:
            send_message(response, 'ai')
