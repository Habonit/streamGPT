basic_summarize_template = """
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

reference_extraction_template = """
다음은 논문에서 reference 부분만 발췌한 것입니다. 
이를 보고 논문의 제목과 저자를 출력 형식에 따라 JSON 형태로 정리해주세요.

References:
{references}

출력 형식:
```json
{{
1 : {{"Title":<<논문의 제목1>>, "Authors":<<논문의 저자1>>}},
2 : {{"Title":<<논문의 제목2>>, "Authors":<<논문의 저자2>>}},
3 : {{"Title":<<논문의 제목3>>, "Authors":<<논문의 저자3>>}}
}}
```
"""

reference_count_template_dict = {
    '0': """
다음은 논문의 일부와 참고문헌 목록을 정리한 것입니다. 
논문 본문에서 참고문헌이 인용되었다면, 해당 참고문헌의 "Counter"에 인용 횟수를 기록하고,  
"Context"에는 해당 참고문헌이 인용된 부분을 **300자 이내**로 발췌해서 저장하세요.  
**인용되지 않은 참고문헌은 출력하지 말고, 인용된 참고문헌만 JSON 형식으로 출력하세요.**

**조건**
{condition}
- 참고문헌이 여러 번 인용되었을 경우, 각 인용 부분을 `"Context"` 배열에 저장하세요.
- `"Context"`에 포함되는 인용된 문장은 **최대 300자 이내**로 유지하세요.

**출력 형식**
```json
{{
    "27": {{
        "Title": "참고문헌 제목1",
        "Counter": 3,
        "Context": ["인용된 부분1", "인용된 부분2", "인용된 부분3"]
    }},
    "5": {{
        "Title": "참고문헌 제목2",
        "Counter": 1,
        "Context": ["인용된 부분1"]
    }}
}}
```

**References**
{references}

**Essay**
{essay}

""",
    "1" : """
논문의 일부와 참고문헌 목록이 제공되었습니다. 
논문 본문에서 특정 참고문헌이 인용되었다면, 해당 참고문헌의 **인용 횟수**를 `"Counter"`에 기록하고, 
**인용된 문장**을 `"Context"`에 최대 300자 이내로 발췌하여 저장하세요.  
**한 번도 인용되지 않은 참고문헌은 출력에서 제외됩니다.**

## 조건
{condition}
- 동일한 참고문헌이 여러 번 인용되면 `"Context"` 배열에 모든 인용 부분을 포함합니다.
- `"Context"`의 개별 항목은 **최대 300자 이하**를 유지해야 합니다.

## 출력 예시
```json
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
```

## References
{references}

## Essay
{essay}
""",

    "2": """
논문의 본문과 참고문헌 목록을 정리해야 합니다.  
논문에서 특정 참고문헌이 인용되었을 경우,
해당 참고문헌의 **인용 횟수(`"Counter"`)**와 **인용 문장(`"Context"`)**을 JSON으로 출력하세요.  
**본문에서 한 번도 인용되지 않은 참고문헌은 출력하지 않습니다.**

## 조건
{condition}
- 동일한 참고문헌이 여러 번 인용되었을 경우, `"Context"` 배열에 각 인용 문장을 저장합니다.
- `"Context"`의 각 문장은 **300자 이내**로 제한해야 합니다.

## 출력 예시
```json
{{
    "7": {{
        "Title": "참고문헌 제목1",
        "Counter": 3,
        "Context": ["첫 번째 인용 문장", "두 번째 인용 문장", "세 번째 인용 문장"]
    }},
    "19": {{
        "Title": "참고문헌 제목2",
        "Counter": 1,
        "Context": ["인용된 문장"]
    }}
}}
```

## References
{references}

## Essay
{essay}
"""}

question_reduction_template = """
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

reference_qna_template = """
다음 문서를 보고 질문 목록에 대한 대답이나 관련 지식을 5문장으로 된 한 문단으로 요약해주세요.

### 인용 논문 제목 (Title): {title}
### 질문 목록 (Questions): {questions}
### 문서 (Document): {essay}

"""

# research_progress_template = """
# 다음은 논문과 해당 논문의 참고문헌에 관한 내용입니다.
# 해당 논문이 인용 논문에서의 주요 질의 응답에 대하여 어떻게 연구를 발전시켰는지 한 단락으로 서술해주세요.

# ### 해당 논문 제목 (Title): {title}
# ### 해당 논문 내용 (Essay): {essay}
# ### 참고 문헌에 대한 주요 질의 응답(QnA): {qna}

# ### 답변 예시 (Example Answer):
# 인용 논문 제목 (Title): <<논문 제목>>
# <<참고 문헌에 대한 주요 질의 응답에 대하여 어떻게 논문이 연구를 발전시켰는지 300자 내외로 서술>>
# """