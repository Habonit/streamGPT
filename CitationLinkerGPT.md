# CitationLinker

CitationLinker는 논문 요약을 넘어, 본문에서 인용된 논문의 빈도를 분석하여 연관성을 평가하고, 이를 반영한 요약을 제공하는 프로젝트입니다. 특히 최근접 인용 논문과의 연관도를 고려하여 보다 정밀한 논문 요약을 목표로 합니다.
---
## 🔍 프로젝트 배경

AI를 활용한 연구 및 개발 과정에서 논문을 참고해야 하는 경우가 많습니다. 그러나 논문의 난해함보다, 해당 논문이 기반으로 삼고 있는 배경지식을 몰라 독해가 어려운 경우가 빈번합니다. 

이를 해결하려면 논문에서 중요한 인용 논문이 무엇인지, 그리고 본문과의 접점이 무엇인지 자동으로 분석하는 서비스가 필요합니다. CitationLinker는 이러한 문제를 해결하고자, 단순 요약을 넘어 인용 논문을 고려한 논문 요약 서비스를 구현하는 것을 목표로 합니다.
---
## 🛠️ 구현 전략

논문의 인용 방식은 저널이나 학회마다 다르기 때문에, 범용적인 분석을 위해 OpenAI API가 지원하는 강력한 언어 모델을 활용합니다. 이를 통해 다양한 논문의 인용 데이터를 일관된 방식으로 처리할 수 있습니다.

따라서 온프레미스 환경에서 직접 모델을 학습하기보다는, API를 적극 활용하여 프롬프트 기반의 데이터 처리 방식을 채택할 계획입니다.

소스코드는 citationlinker.py에서 확인 가능합니다. 
---
## 🛠️ 구현 검증 방법

1차 목표는 **2024년 NeurIPS 논문**인 *How Do Large Language Models Acquire Factual Knowledge During Pretraining?*을 대상으로 참고 문헌 기반 요약을 수행하는 것입니다. 

- **1월 31일까지**: 기본 기능 구현 완료
- **2월 7일까지**: Streamlit을 활용한 UI 개발
- **2월 14일까지**: IEEE, ETRI 논문 등 다양한 논문에 대한 적용성 검토 및 최종 점검
---
## 🔧 구현 세부 전략

1. **논문 기본 요약**
2. **참고 문헌 구조화**
3. **참고 문헌 인용 횟수 카운트**
4. **최다 인용 논문 필터링 및 논문과의 접점 정리**
5. **해당 논문이 최다 인용 논문을 어떻게 활용하고 연구를 발전시키는지 분석**
6. **1~5번 내용을 Streamlit UI로 구현**
7. **1~5번 기반으로 대화가 가능한 챗봇 구축**
---
## 📌 1차 구현(구현 세부 전략의 1~5번 결과물) 사용 방법

### 1️⃣ `config.json` 작성  
`config.json` 파일을 생성하고 아래 내용을 작성합니다.
config 작성법은 아래를 따르며 content_keys만 아래 사항을 따라 주의하여 작성합니다.
1) Title, Authors, Submitted, Abstract는 아무것도 수정하지 않습니다. 이는 모든 논문의 공통적인 항목입니다.
2) 그 외에 항목은 모두 리뷰하려는 논문에 따라 달라져야 합니다. 이 때 deliminators_forward는 상단 구분자, backward는 하단 구분자입니다.
3) 모든 구분자는 항목을 구분하기 위한 논문 내의 유일한 표현이어야 합니다.

```json
{
    "arxiv_id": "리뷰하려는 논문의 arxiv ID",
    "essay_dir": "논문을 저장할 디렉토리 경로",
    "model": "openai 모델 (예: gpt-4o-mini 추천)",
    "preprocess_threhsold": 25,
    "reference_ratio": 0.2,
    "reference_condition": "참고문헌 인용 표시 방법 안내",
    "result_dir": "결과 파일 저장 디렉토리",
    "content_keys": {
        "1": { "name": "Title", "deliminators": { "forward": null, "backward": null } },
        "2": { "name": "Authors", "deliminators": { "forward": null, "backward": null } },
        "3": { "name": "Submitted", "deliminators": { "forward": null, "backward": null } },
        "4": { "name": "Abstract", "deliminators": { "forward": null, "backward": null } },
        "5": { "name": "Introduction", "deliminators": { "forward": "Introduction\n\n", "backward": "Related Work\n\n"} },
        "6": { "name": "Related Work", "deliminators": { "forward": "Related Work\n\n", "backward": "Experimental Setup\n\n"} },
        "7": { "name": "Experimental Setup", "deliminators": { "forward": "Experimental Setup\n\n", "backward": "Results\n\n"} },
        "8": { "name": "Results", "deliminators": { "forward": "Results\n\n", "backward": "Discussion and Conclusions\n\n"} },
        "9": { "name": "Discussion and Conclusions", "deliminators": { "forward": "Discussion and Conclusions\n\n", "backward": "References\n\n"} },
        "10": { "name": "References", "deliminators": { "forward": "References\n\n", "backward": "Appendix\n\n"} }
    }
}
```
### 2️⃣ .env 파일 작성
.env 파일을 생성하고 아래 내용을 입력합니다.

```text 
OPENAI_API_KEY=your_openai_api_key
```

### 3️⃣ 실행
```bash
python citationlinker.py
```
### 4️⃣ 실행 완료 확인
실행이 정상적으로 완료되면 result_dir 내에 아래 파일이 생성됩니다.

basic_summary.json : 논문의 기본 요약
reference_count.json : 참고문헌 인용 횟수
reference_qna.json : 참고문헌과의 접점 정리
실행 시간은 약 25분 소요됩니다.

## 📜 프롬프트 상세보기
프롬프트 원문은 prompt.py에서 확인 가능합니다.

### 프롬프트(1) - 논문 기본 요약
```text
논문: {essay}

다음 논문의 내용을 요약해주세요. 항목은 다음과 같습니다.

## 항목 정보
1. 기본 정보: 논문의 제목과 저자, 모두 영문으로 작성합니다. 
2. 연구 목적: 이 논문이 가진 문제의식과 이에 대한 설명을 작성합니다. 이때 문제의식은 명사형으로 끝내고 설명은 300자에서 500자로 서술합니다. 
3. 연구 방법: 해당 논문이 연구한 실험 방법과 데이터, 모델, 분석 방법 등을 작성합니다. 연구 방법은 가능한 한 구체적으로 기술합니다.
4. 주요 결과: 논문의 연구 성과를 요약합니다. 이때 연구의 핵심적인 발견과 논문의 주요 기여를 강조합니다.
5. 결론 및 시사점: 논문의 결론과 연구의 의미를 정리합니다. 또한 연구의 한계점과 향후 연구 방향을 포함하여 설명합니다.

## 응답 예시

### 1. 기본 정보
1) 제목: <<논문 제목(영문)>>
2) 저자: <<논문 저자(영문)>>

### 2. 연구 목적
1) 문제의식: <<문제 의식을 명사형으로 구성>>
2) 설명: <<문제 의식에 대한 설명을 300자에서 500자로 서술>>

### 3. 연구 방법
1) 실험 방법: <<연구에서 사용한 실험 방법을 기술>>
2) 데이터: <<사용된 데이터의 출처 및 특성을 설명>>
3) 모델 및 분석 방법: <<적용된 모델과 분석 기법을 설명>>

### 4. 주요 결과
1) 연구의 주요 발견: <<논문의 핵심 결과를 요약>>
2) 기여 및 성과: <<연구가 기존 연구 대비 기여한 점을 설명>>

### 5. 결론 및 시사점
1) 결론: <<연구의 주요 결론을 요약>>
2) 시사점: <<연구가 시사하는 바와 활용 가능성을 설명>>
3) 연구의 한계: <<논문에서 언급한 연구의 한계를 기술>>
4) 향후 연구 방향: <<추가 연구가 필요한 부분을 설명>>

가능한 한 논문의 핵심 내용을 유지하면서 간결하고 명확하게 요약해주세요.
"""

```

### 프롬프트(2) - 참고 문헌 구조화
```text 
"""
다음은 논문에서 reference 부분만 발췌한 것입니다. 
이를 보고 논문의 제목과 저자를 출력 형식에 따라 JSON 형태로 정리해주세요.

References:
{references}

출력 형식:

{{
1 : {{"Title":<<논문의 제목1>>, "Authors":<<논문의 저자1>>}},
2 : {{"Title":<<논문의 제목2>>, "Authors":<<논문의 저자2>>}},
3 : {{"Title":<<논문의 제목3>>, "Authors":<<논문의 저자3>>}}
}}

"""
```
### 프롬프트(3) - 참고 문헌 인용 횟수 count (일부만 수록)

```text 
"""
논문의 일부와 참고문헌 목록이 제공되었습니다. 
논문 본문에서 특정 참고문헌이 인용되었다면, 해당 참고문헌의 **인용 횟수**를 `"Counter"`에 기록하고, 
**인용된 문장**을 `"Context"`에 최대 300자 이내로 발췌하여 저장하세요.  
**한 번도 인용되지 않은 참고문헌은 출력에서 제외됩니다.**

## 조건
{condition}
- 동일한 참고문헌이 여러 번 인용되면 `"Context"` 배열에 모든 인용 부분을 포함합니다.
- `"Context"`의 개별 항목은 **최대 300자 이하**를 유지해야 합니다.

## 출력 예시

{{
    "10": {{
        "Title": "참고문헌 제목1",
        "Counter": 2,
        "Context": ["첫 번째 인용 문장", "두 번째 인용 문장"]
    }},
    "3": {{
        "Title": "참고문헌 제목2",
        "Counter": 1,
        "Context": ["인용된 문장"]
    }}
}}


## References
{references}

## Essay
{essay}
"""
```

### 프롬프트(4) - 질문 축약 

```text
"""
다음 문장들을 분석하여 의미적으로 유사한 문장은 모두 제거하고, 서로 독립적인 문장만 남겨주세요.

### 입력 문장 목록:
{text_list}

### 출력 형식:
- 중복되거나 유사한 문장을 제거한 후, 의미적으로 독립적인 문장만 번호와 함께 남겨주세요.
- 출력 결과는 원래 문장의 의미를 유지하면서 중복을 피하도록 정제해주세요.

### 결과 예시 :
1. [27] reported that the performance of LLMs adheres to a scaling law, correlating positively with both the model size and the size of the pretraining corpus.
2. In Eq.1, the definition of the local acquisition maxima is also dependent on the injected knowledge k and the window size tw, but we write tLAM(q, i) for brevity.
3. While our finding that effectivity remains unchanged for different stages of pretraining may seem contradictory to the widely known observation that the amount of pretraining data is a critical factor in the performance of LLMs [23, 27], we suggest a plausible hypothesis based on further observations in §4.3.

위와 같은 방식으로 유사한 문장을 제거하고 정제된 결과를 반환해주세요.

"""

```

### 프롬프트(5) - 인용 논문과의 접점 정리-1 
```text
"""
다음 문서를 보고 질문은 한국어로 번역하여 출력해주세요.
질문에 대한 답변 및 관련 지식을 한국어로 요약하고, 답변의 근거가 되는 부분을 논문에서 그대로 발췌해 주세요.
그리고 모든 질문에 대한 답변을 500자 내외의 한 단락으로 정리해주세요.
### 인용 논문 제목 (Title): {title}
### 질문 목록 (Questions): {questions}
### 문서 (Document): {essay}
### 답변 예시 (Example Answer):
인용 논문 제목 (Title): <<논문 제목>>
1. **질문 :** <<질문1-한국어로 번역된 질문>>
   - **답변 :** <<질문1에 대한 한국어 답변이나 관련 지식 한 단락(300자 내외)>>
   - **근거 :** <<인용 논문에서 해당 답변을 뒷받침하는 문장>>

2. **질문 :** <<질문2-한국어로 번역된 질문>>
   - **답변 :** <<질문2에 대한 한국어 답변이나 관련 지식 한 단락(300자 내외)>>
   - **근거 :** <<인용 논문에서 해당 답변을 뒷받침하는 문장>>
...

답변 요약 
: <<전체 답변에 대한 요약 한 단락(500자 내외)>>
### 답변 요약 (Summary of Answers)
"""
```
### 프롬프트(6) - 인용 논문과의 접점 정리-2
```text 
"""
다음은 논문과 해당 논문의 참고문헌에 관한 내용입니다.
해당 논문이 인용 논문에서의 주요 질의 응답에 대하여 어떻게 연구를 발전시켰는지 한 단락으로 서술해주세요.

### 해당 논문 제목 (Title): {title}
### 해당 논문 내용 (Essay): {essay}
### 참고 문헌에 대한 주요 질의 응답(QnA): {qna}

### 답변 예시 (Example Answer):
인용 논문 제목 (Title): <<논문 제목>>
<<참고 문헌에 대한 주요 질의 응답에 대하여 어떻게 논문이 연구를 발전시켰는지 300자 내외로 서술>>
"""
```

## 남은 기간 🚀 추가 과제 및 향후 계획

### ✅ 1️⃣ Streamlit 통합 및 ImageGPT 연계  
- 현재 구현한 CitationLinker를 **Streamlit**을 활용하여 UI 형태로 구현  
- 기존 **ImageGPT** 프로젝트와 통합하여 하나의 **Streamlit 기반 논문 분석 프로젝트**로 완성  

### ✅ 2️⃣ 논문 리뷰 기반 챗봇 개발  
- 구현된 논문 분석 기능을 활용하여 **논문 리뷰 내용을 기반으로 대화 가능한 챗봇** 개발  
- 논문의 요약 및 참고 문헌 정보를 활용한 질의응답 기능 추가  

### ✅ 3️⃣ 다양한 저널 논문 호환성 강화  
- 여러 저널 및 학회의 논문에서도 **호환될 수 있도록 프롬프트 최적화**  
- 논문 구조가 다를 경우에도 자동으로 분석할 수 있도록 **유연한 데이터 처리 방식 적용**  