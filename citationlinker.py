import json
import arxiv
import requests
from pathlib import Path
import os
import re
from dotenv import load_dotenv
from copy import deepcopy

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
        self.model = config['model']

        self.essay_dir = Path(config['essay_dir'])
        self.result_dir = Path(config['result_dir'])
        CitationLinker.create_directory_if_not_exists(self.essay_dir)
        CitationLinker.create_directory_if_not_exists(self.result_dir)

        self.title = None
        self.authors = None
        self.submitted = None
        self.abstract = None
        self.pdf_url = None

        self.basic_keys = ["Title", "Authors", "Submitted" ,"Abstract"]
        self.content_config = config['content_keys']
        self.content_keys = [value_dict["name"] for _, value_dict in self.content_config.items()]
    
    @staticmethod
    def create_directory_if_not_exists(directory_path):
        if not os.path.exists(directory_path):
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
    
    @staticmethod
    def _message_to_openai(message, model):
        response = client.chat.completions.create(
            model=model,
            store=True,
            messages=[{"role": "user", "content": message}],
            temperature=0
        )
        return response

    def forward(self):
        result = {
            "basic_summary" : None,
        }

        # 논문 id를 받아서 논문을 다운 받습니다.
        self._search_arxiv_pdf(arxiv_id=self.target_id)
        title = self.title
        save_path = self.essay_dir / f"0-{title[:15]}.pdf"
        self._download_arxiv_pdf(
            pdf_url=self.pdf_url, 
            save_path=save_path
        )

        # 텍스트 전처리
        processed_output = self._preprocess(
            save_path=save_path
        )

        # 기본 요약
        basic_summarize_message = basic_summarize_template.format(essay="\n\n".join([processed_output[key] for key in self.content_keys]))
        response = CitationLinker._message_to_openai(message=basic_summarize_message)
        result['basic_summary'] = response.choices[0].message.content,

        # 참고문헌 목록화화
        reference_extraction_message = reference_extraction_template.format(reference=processed_output['References'])
        flag = True
        while flag:
            response = CitationLinker._message_to_openai(message=reference_extraction_message)
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

        # 인용횟수 counting
        n = 5
        for index in range(n):
            result = []
            for key in self.content_keys:
                if key not in self.basic_keys:
                    for essay in processed_output[key]:
                        reference_count_message = reference_count_template_dict[str(index)].format(references=reference_dict, essay=essay)
                        response = CitationLinker._message_to_openai(reference_count_message)
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
                    processed_output["References"][key]['Counter'] += value_dict['Counter']
                    processed_output["References"][key]['Context'].extend(value_dict['Context'])
        print(processed_output)

if __name__ == "__main__":

    with open("config.json",'r') as f:
        config = json.load(f)

    citation_linker = CitationLinker(config)
    citation_linker.forward()


