from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import os
from dotenv import load_dotenv
import base64
from PIL import Image
from langsmith import Client

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="ImageGPT", page_icon="📃")

st.title("📃 ImageGPT")

if "input_tokens" not in st.session_state:
    st.session_state["input_tokens"] = 0
if "output_tokens" not in st.session_state:
    st.session_state["output_tokens"] = 0
if "cost_per_1000_input" not in st.session_state:
    st.session_state["cost_per_1000_input"] = 0.000075  # 기본값: GPT-4o-mini input 토큰 가격
if "cost_per_1000_output" not in st.session_state:
    st.session_state["cost_per_1000_output"] = 0.000600  # 기본값: GPT-4o-mini output 토큰 가격

# ✅ 이미지 저장 및 base64 변환
@st.cache_data(show_spinner="Processing Images...")
def process_images(files):
    image_base64_list = []
    os.makedirs("./.cache/files", exist_ok=True)

    for i, file in enumerate(files, start=1):
        file_path = f"./.cache/files/{file.name}"
        image = Image.open(file)
        image.save(file_path, format="JPEG")

        with open(file_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            image_base64_list.append({"index": i, "base64": image_base64, "name": file.name})

    return image_base64_list

def save_message(message, role, input_tokens=0, output_tokens=0):
    st.session_state['messages'].append({
        'message': message, 
        'role': role, 
        'input_tokens': input_tokens, 
        'output_tokens': output_tokens
    })

def send_message(message, role, input_tokens=0, output_tokens=0, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if input_tokens or output_tokens:
            st.markdown(f"🔹 **Tokens Used:** {input_tokens} in, {output_tokens} out")
    if save:
        save_message(message, role, input_tokens, output_tokens)

def paint_history():
    for message in st.session_state['messages']:
        send_message(message['message'], message['role'], message['input_tokens'], message['output_tokens'], save=False)

def build_messages(context, message, encoded_images):
    system_message = SystemMessage(
        content=[
            {"type": "text", "text": "You are an AI assistant. Answer the user's question based on the conversation history and provided images."},
            {"type": "text", "text": context}
        ]
    )

    human_message = HumanMessage(
        content=[
            {"type": "text", "text": message}
        ] + [
            {"type": "image_url", "image_url": {"url": image_url}} for image_url in encoded_images
        ]
    )
    return [system_message, human_message]

def update_token_display(token_display):
    """ 🔄 사이드바 토큰 사용량 및 비용 정보 업데이트 """
    total_cost = (
        (st.session_state["input_tokens"] / 1000 * st.session_state["cost_per_1000_input"]) +
        (st.session_state["output_tokens"] / 1000 * st.session_state["cost_per_1000_output"])
    )
    
    with token_display.container():
        st.sidebar.subheader("📊 **Token Usage Summary**")
        st.sidebar.markdown(f"- 🔵 **Total Input Tokens:** `{st.session_state['input_tokens']}`")
        st.sidebar.markdown(f"- 🟢 **Total Output Tokens:** `{st.session_state['output_tokens']}`")
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 **Estimated Cost**")
        st.sidebar.markdown(f"🔹 **Input Cost:** `${st.session_state['input_tokens'] / 1000 * st.session_state['cost_per_1000_input']:.6f}`")
        st.sidebar.markdown(f"🔹 **Output Cost:** `${st.session_state['output_tokens'] / 1000 * st.session_state['cost_per_1000_output']:.6f}`")
        st.sidebar.markdown(f"### **💵 Total Cost:** `${total_cost:.6f}`")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# ✅ 사이드바 (이미지 업로드 및 토큰 정보 표시)
with st.sidebar:
    st.subheader("📂 **Upload Images**")
    files = st.file_uploader(
        label='Upload images (.jpg, .jpeg, .png)', 
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    images = process_images(files) if files else []

    token_display = st.sidebar.empty()  # 🔹 동적으로 업데이트할 공간 생성

    st.sidebar.subheader("⚙️ **Settings**")
    cost_per_1000_input = st.sidebar.number_input(
        "💰 Cost per 1000 Input Tokens ($)", 
        min_value=0.0, 
        value=st.session_state["cost_per_1000_input"], 
        step=0.000001,
        format="%.6f" 
    )
    st.session_state["cost_per_1000_input"] = cost_per_1000_input  

    cost_per_1000_output = st.sidebar.number_input(
        "💰 Cost per 1000 Output Tokens ($)", 
        min_value=0.0, 
        value=st.session_state["cost_per_1000_output"], 
        step=0.000001,
        format="%.6f" 
    )
    st.session_state["cost_per_1000_output"] = cost_per_1000_output  

    # update_token_display(token_display)  

    if images:
        st.sidebar.subheader("🖼 **Uploaded Images**")
        for img in images:
            st.sidebar.image(f"data:image/jpeg;base64,{img['base64']}", caption=f"📸 Image {img['index']}: {img['name']}")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if images:
    send_message('준비가 완료되었습니다! 이미지에 대해 질문하세요.', 'AI', save=False)
    paint_history()
    message = st.chat_input('Ask anything about your file...')

    if message:
        send_message(message, 'HUMAN')
        context = "\n".join([f"{msg['role']}: {msg['message']}" for msg in st.session_state['messages']])
        formatted_images = [f"data:image/jpeg;base64,{img['base64']}" for img in images]
        formatted_messages = build_messages(context, message, formatted_images)
        response = llm.invoke(formatted_messages)

        input_tokens_used = response.usage_metadata["input_tokens"] 
        output_tokens_used = response.usage_metadata["output_tokens"] 

        st.session_state["input_tokens"] += input_tokens_used
        st.session_state["output_tokens"] += output_tokens_used

        send_message(response.content, 'AI', input_tokens_used, output_tokens_used)
        update_token_display(token_display)

else:
    st.markdown('''
    ## 👋 Welcome to ImageGPT!
    
    - Upload an image via the sidebar.
    - Ask questions about the uploaded image.
    - View token usage and cost in the sidebar.
    ''')

