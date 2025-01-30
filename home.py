import streamlit as st

st.set_page_config(
    page_title='ChatBot',
    page_icon = '※'
)

st.title("Sparta ChatBot Service")

st.markdown(
    '''
    Welcome to ChatBot Portfolio!

    Here are the apps I made:

    - [x] [ImageQAGPT](/ImageGPT)
    - [ ] [CitationLinkerGPT](/CitationGPT)

    Detailed Information:
    1) ImageGPT: A service that allows users to upload multiple images and perform Q&A about them using the GPT API.
    2) CitationLinkerGPT: A service that generates detailed summaries based on cited papers by taking basic configuration inputs and enables Q&A using the GPT API.
    '''
)
