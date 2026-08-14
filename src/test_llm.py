import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
)

response = llm.invoke(
    "What is Profit and Loss? Cost Price is the amount paid to purchase an article. Answer using only this information."
)

print(response.content)