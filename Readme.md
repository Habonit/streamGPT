# Sparta GPT Portfolio

## 개요
- Sparta GPT Portfolio는 스트림릿(Streamlit)을 기반으로 챗봇 서비스를 구현하는 프로젝트입니다. 향후 AWS를 통해 배포까지 진행할 예정입니다.

---

## 현재 구현 상황

### 1. ImageGPT ✅ (완료)
이미지를 업로드하여 이를 기반으로 대화할 수 있는 챗봇 서비스입니다.
- **소스 코드 위치**
  - `home.py`
  - `pages/ImageGPT.py`
- **설명 문서**: [ImageGPT.md](./ImageGPT.md)

### 2. CitationLinkerGPT ⚙️ (개발 중)
참고문헌의 내용을 바탕으로 논문을 요약하고 유사도 점수를 산출하며, 이를 기반으로 챗봇 서비스를 구현하는 기능을 개발 중입니다.
- **소스 코드 위치**
  - `citationlinker.py`
  - `prompt.py`
- **설명 문서**: [CitationLinkerGPT.md](./CitationLinkerGPT.md)

---

## 프로젝트 목표
1. 스트림릿을 활용한 직관적인 챗봇 서비스 구축
2. 다양한 AI 모델을 연동하여 GPT 기반 챗봇 기능 확장
3. AWS를 이용한 배포 및 서비스 운영 실험
4. 연구 및 논문 분석을 위한 AI 기반 챗봇 개발 (CitationLinkerGPT)

---

## 프로젝트 실행
```bash
streamlit run home.py
```
---

## 파일 구조
```bash

SpartaGPTPortfolio/
│── home.py  # 메인 실행 파일
│── README.md  # 프로젝트 설명 문서
│
├── pages/
│   ├── ImageGPT.py  # ImageGPT 관련 코드
│   ├── CitationGPT.py # CitationLinkerGPT 관련 코드
│
├── citationlinker.py  # CitationLinkerGPT의 핵심 로직
├── prompt.py  # GPT 챗봇 프롬프트 관련 코드
│
├── week6_basic_ImageGpt.mp4 # ImageGPT 구동 영상
│
├── docs/
│   ├── ImageGPT.md  # ImageGPT 설명 문서
│   ├── CitationLinkerGPT.md  # CitationLinkerGPT 설명 문서
│ 
├── image/ # md에서 사용된 이미지 저장소소
│   ├── citation_image.png 
│
├── reference/ # citationlinker.py에서 추출한 논문들이 저장되는 공간
├── resultl/
│   ├── basic_summary.json # citationlinker.py의 논문에 대한 기본 요약
│   ├── reference_count.json # citationlinker.py의 citation_score를 내기 위한 자료
│   ├── reference_qna.json # citationlinker.py의 참고문헌에 근거한 요약
```