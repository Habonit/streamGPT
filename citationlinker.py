import json
import arxiv
import requests
from pathlib import Path
import os
import re
from dotenv import load_dotenv
from copy import deepcopy
from tqdm import tqdm

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from openai import OpenAI
import openai
from prompt import * 

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

class CitationLinker():
    def __init__ (self, config):
        self.target_id = config['arxiv_id']
        self.preprocess_threhsold = config['preprocess_threhsold']
        self.reference_ratio = config['reference_ratio']
        self.reference_condition = config['reference_condition']
        self.model = config['model']

        self.essay_dir = Path(config['essay_dir'])
        self.result_dir = Path(config['result_dir'])

        self.title = None
        self.authors = None
        self.submitted = None
        self.abstract = None
        self.pdf_url = None

        self.basic_keys = ["Title", "Authors", "Submitted" ,"Abstract"]
        self.references = ['References']
        
        self.content_config = config['content_keys']
        self.content_keys = [value_dict["name"] for _, value_dict in self.content_config.items()]
        
    @staticmethod
    def _create_directory_if_not_exists(directory_path):
        if directory_path.split("/")[-1] == "target-id":
            pass

        elif not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"디렉토리 '{directory_path}'가 생성되었습니다.")
        else:
            raise FileExistsError(f"에러: '{directory_path}' 디렉토리가 이미 존재합니다.")

    def _search_arxiv_pdf(self, arxiv_id):
        search = arxiv.Search(id_list=[arxiv_id])
        for result in search.results():
            break
        print(f"📌 Title: {result.title}")
        print(f"📝 Authors: {', '.join([author.name for author in result.authors])}")
        print(f"📅 Submitted: {result.published}")
        print(f"🔗 PDF Link: {result.pdf_url}")
        print(f"📝 Abstract:\n{result.summary}")
        self.title = result.title
        self.authors = ', '.join([author.name for author in result.authors])
        self.submitted = str(result.published)
        self.abstract = result.summary
        self.pdf_url = result.pdf_url

    def _download_arxiv_pdf(self, pdf_url, save_path):
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
        print("논문 저장 완료!")

    def _fetch_arxiv_paper(self, title, max_results=30):
        
        search = arxiv.Search(
            query=title,
            max_results=max_results, 
            sort_by=arxiv.SortCriterion.Relevance
        )

        for result in search.results():
            if title[10:-10].lower().replace(" ", "") in result.title.lower().replace(" ", ""):
                return ( {
                    "title": result.title,
                    "abstract": result.summary,
                    "pdf_url": result.pdf_url
                })
            
        return None 
    
    @staticmethod
    def _message_to_openai(message, model):
        response = client.chat.completions.create(
            model=model,
            store=True,
            messages=[{"role": "user", "content": message}],
            temperature=0.5
        )
        return response
    
    def _search_and_download_essay(self, arxiv_id):
        self._search_arxiv_pdf(arxiv_id=arxiv_id)
        self._download_arxiv_pdf(
            pdf_url=self.pdf_url, 
            save_path=self.essay_dir / f"0-{self.title}.pdf"
        )
    def _preprocess(self, save_path):
        # 데이터를 불러와 섹션 별로 나눕니다.
        loader = UnstructuredPDFLoader(save_path)
        documents = loader.load()
        processed_output = {}
        for key, value_dict in self.content_config.items():
            if value_dict['name'] == "Title":
                processed_output[value_dict['name']]=self.title
            elif value_dict['name'] == "Authors":
                processed_output[value_dict['name']]=self.authors
            elif value_dict['name'] == "Submitted":
                processed_output[value_dict['name']]=self.submitted
            elif value_dict['name'] == "Abstract":    
                processed_output[value_dict['name']]=self.abstract
            else:
                processed_output[value_dict['name']]=documents[0].page_content.split(value_dict['deliminators']['forward'])[-1].split(value_dict['deliminators']['backward'])[0]

        # basic_key가 아닌 섹션 중 threshold 미만으로 잘리면 모두 없앱니다.
        threshold = self.preprocess_threhsold
        for key in self.content_keys:
            if key not in self.basic_keys:
                result = []
                for text in processed_output[key].split("\n"):
                    if len(text) >= threshold:
                        result.append(text)
                processed_output[key] = "\n".join(result)
        
        # 2000자 단위로 모두 자릅니다.
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=0  
            )
        
        for key in self.content_keys:
            documents = text_splitter.create_documents([processed_output[key]])
            for doc in documents:
                doc.metadata = {"Title": self.title, "Key": key} 
            processed_output[key] = documents

        return processed_output

    def _basic_summarize(self, basic_summarize_template, processed_output):
        # 기본 요약
        essay = ""
        for key in self.content_keys:
            if key not in self.references:
                for doc in processed_output[key]:
                    essay += doc.page_content + "\n\n"
        basic_summarize_message = basic_summarize_template.format(essay=essay)
        response = CitationLinker._message_to_openai(message=basic_summarize_message, model=self.model)
        return response.choices[0].message.content
    
    def _reference_preprocess(self, reference_extraction_template, processed_output):
        reference_extraction_message = reference_extraction_template.format(references=processed_output['References'])
        flag = True
        while flag:
            response = CitationLinker._message_to_openai(message=reference_extraction_message, model=self.model)
            text = response.choices[0].message.content
            text = re.sub("```json","",text)
            text = re.sub("```","",text)
            json_data = json.loads(text)
            flag = False
            
        reference_dict = {}
        for key, dict_data in json_data.items():
            dict_data['Counter'] = 0
            dict_data['Context'] = []
            reference_dict[key] = dict_data
        processed_output['References'] = deepcopy(reference_dict)
        return processed_output, reference_dict
    
    def _reference_counting(self, reference_count_template_dict, processed_output, reference_dict):
        for index in range(len(reference_count_template_dict)):
            result = []
            for key in self.content_keys:
                if key not in self.basic_keys + self.references:
                    for essay in processed_output[key]:
                        reference_count_message = reference_count_template_dict[str(index)].format(references=reference_dict, essay=essay, condition=self.reference_condition)
                        response = CitationLinker._message_to_openai(reference_count_message, model=self.model)
                        try:
                            text = response.choices[0].message.content
                            text = re.sub("```json","",text)
                            text = re.sub("```","",text)
                            text_data = json.loads(text)
                            result.append(text_data)
                        except: 
                            text_data = None
                        # items['References'] = text_data

            for data in result:
                for key, value_dict in data.items():
                    try:
                        processed_output["References"][key]['Counter'] += value_dict['Counter']
                        processed_output["References"][key]['Context'].extend(value_dict['Context'])
                    except:
                        pass
        return processed_output
    
    def _download_reference(self, processed_output, nums = 5):
        # counter 순으로 정렬을 한다
        # title을 던진다
        # 개수가 맞으면 멈춘다
        related_reference = processed_output['References']
        related_reference = dict(sorted(related_reference.items(), key=lambda item: item[1]["Counter"], reverse=True))
        processed_output['References'] = related_reference
        ordered_titles = list(processed_output['References'].items())
        downloads_lst = []

        for index, valud_dict in ordered_titles:
            title = valud_dict['Title']
            try :
                paper_info = self._fetch_arxiv_paper(title)
                if paper_info is None:
                    paper_info = self._fetch_arxiv_paper(title, 150)
            except Exception as e:
                print(index,"번째 논문 예외 발생: ", e)
                try :
                    paper_info = self._fetch_arxiv_paper(title, None)
                except:
                    paper_info = None

            if paper_info is not None:
                pdf_url = paper_info['pdf_url']
                abstract = paper_info['abstract']
                processed_output['References'][index]['abstract'] = abstract
                processed_output['References'][index]['pdf_url'] = pdf_url
                save_path = self.essay_dir / (index+ "-" + paper_info['title']+".pdf")
                self._download_arxiv_pdf(pdf_url, save_path)
                print(index,"번째 논문 다운로드 완료")

                downloads_lst.append((index, valud_dict))
            
            else:
                pdf_url = None
                abstract = None
                processed_output['References'][index]['abstract'] = abstract
                processed_output['References'][index]['pdf_url'] = pdf_url
                print(index,"번째 논문 다운로드 실패")
                print(f"    {processed_output['References'][index]['Title']}")
            
            if len(downloads_lst) == nums:
                break
            
        # 논문 다운로드 후, 질문 축소
        # 이렇게 하는 이유는 동일 질문을 여러개 갖고 있을 수 있기 때문입니다.
        
        related_reference = dict(downloads_lst)
        total_related_reference = processed_output['References']
        return related_reference, total_related_reference, processed_output
    
    def _reduce_questions(self, question_reduction_template, related_reference):
        for index in related_reference.keys():
            query_list = related_reference[index]['Context']
            user_message = question_reduction_template.format(text_list=query_list)
            response = CitationLinker._message_to_openai(user_message, model=self.model)
            related_reference[index]['Questions'] = response.choices[0].message.content
        return related_reference
    
    def _find_connection_from_reference(self, reference_qna_template, related_reference):

        for index in tqdm(related_reference.keys(), desc="인용 논문과의 관련 지점 정리..."):
            title = related_reference[index]['Title']
            questions = related_reference[index]['Questions']
        
            for path in self.essay_dir.rglob("*.pdf"):
                if path.name.split("-")[0] == index:
                    break
        
            loader = UnstructuredPDFLoader(path)
            documents = loader.load()
            essay = documents[0].page_content
            essay = "\n".join([text for text in essay.split("\n") if len(text) >= self.preprocess_threhsold])
            reference_qna_message = reference_qna_template.format(essay = essay, questions=questions, title=title)
            response = CitationLinker._message_to_openai(reference_qna_message, model=self.model)
            summary = response.choices[0].message.content
            related_reference[index]['Summary'] = summary
            
        return related_reference
    
    def forward(self):

        # 논문 id를 받아서 논문을 다운 받습니다.
        # 추후에 논문 pdf를 drag and drop 방식으로 바꿀 수도 있습니다
        self._search_and_download_essay(
            arxiv_id=self.target_id,
        )

        # 텍스트 전처리
        processed_output = self._preprocess(
            save_path=self.essay_dir / f"0-{self.title}.pdf"
        )
        print("step 1: ", "\n",
            processed_output)

        # 기본 요약
        response=self._basic_summarize(
            basic_summarize_template=basic_summarize_template,
            processed_output=processed_output
        )
        print("step 2: ", "\n",
            response, "\n")

        # 기본 요약된 정보 저장
        with open(self.result_dir/"basic_summary.json", 'w', encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=4)

        # 참고문헌 목록화
        processed_output, reference_dict = self._reference_preprocess(
            reference_extraction_template=reference_extraction_template,
            processed_output=processed_output
        )
        print("step 3: ", "\n", 
            processed_output, "\n",
            reference_dict, "\n" 
        )
        # 인용횟수 counting
        # reference_count_template_dict / processed_output / reference_dict
        processed_output = self._reference_counting(
            reference_count_template_dict=reference_count_template_dict, 
            processed_output=processed_output, 
            reference_dict=reference_dict
            )
        print("step 4: ", "\n",
            processed_output, "\n")
        # reference 논문 다운 받아오기
        # 전략 처음엔 30개 중에 다운을 받습니다.
        # 그 다음 150개를 받습니다. 
        # 그 다음 default 값으로 받습니다.
        # 그럼에도 없으면 None으로 채워넣습니다. 
        related_reference, total_related_reference, processed_output = self._download_reference(processed_output=processed_output)
        print("step 5: ", "\n",
            related_reference, "\n", 
            total_related_reference, "\n",
            processed_output, "\n",
        )

        with open(self.result_dir/"reference_count.json", 'w', encoding="utf-8") as f:
            json.dump(total_related_reference, f, ensure_ascii=False, indent=4)

        # # 논문 다운로드 후, 질문 축소
        # # 이렇게 하는 이유는 동일 질문을 여러개 갖고 있을 수 있기 때문입니다.
        related_reference = self._reduce_questions(
            question_reduction_template=question_reduction_template,
            related_reference=related_reference
        )
        print("step 6: ", "\n",
            related_reference, "\n", 
        )
        # reference와의 접점을 찾기 위한 요약
        # 인용 논문과 원래 논문의 접점을 찾고 정리합니다.
        # 원래 논문이 어떻게 연구를 발전시키는지까지 정리합니다. 
        related_reference=self._find_connection_from_reference(
            reference_qna_template=reference_qna_template,
            # research_progress_template=research_progress_template,
            # processed_output=processed_output,
            related_reference=related_reference
        )
        print("step 7: ", "\n",
            related_reference, "\n", 
        )
        with open(self.result_dir/"reference_qna.json", 'w', encoding="utf-8") as f:
            json.dump(related_reference, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    with open("archive/config.json",'r') as f:
        config = json.load(f)

    citation_linker = CitationLinker(config)
    citation_linker.forward()