from langchain.chains import LLMChain
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

os.environ["AWS_PROFILE"] = "facundo"

# bedrock client

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

modelID = "anthropic.claude-3-haiku-20240307-v1:0"


llm = BedrockChat(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"temperature": 0.9}
)


def my_chatbot(language, user_input):
    prompt = PromptTemplate(
        input_variables=["language", "user_input"],
        template="Your name is HoloBot (Don't mention this unless asked). You are answering to Facundo Fierro, born 05-mar-1987 who lives in Copenhagen, Denmark. You are in {language}.\n\n{user_input}. Answer without saying according to the information provided or similar sentences."
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response = bedrock_chain({'language': language, 'user_input': user_input})
    return response


# print(my_chatbot("english", "who is buddha?"))


st.title("HoloBot")

language = st.sidebar.selectbox("Language", ["english", "spanish"])
# placeholder_text = "EnMessage HoloBot..."
# user_input = st.text_input("Label", placeholder=placeholder_text)

if language:
    user_input = st.sidebar.text_area(
        label="Message HoloBot",

        max_chars=100)

if user_input:
    response = my_chatbot(language, user_input)
    st.write(response['text'])
# python -m streamlit run main.py
