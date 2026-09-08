import os
from dotenv import load_dotenv
from openai import OpenAI
# from env import OPENAI_API_KEY

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LLMservice:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o-mini"
        self.SYSTEM_PROMPT = """You are an Intelligent Assistant that answers questions based on the provided context.
        
        Rules:

        1. Use only provided context to answer questions. Do not make up information.

        2. If the answer is not present in the context, respond with 'I count not find the answer in the provided documents'

        3. Answer clearly and professinally. Avoid using filler words like 'umm' or 'ahh'.

        4. Keep the answer concise and to the point. Avoid unnecessary elaboration unless user explicitly requests it.

        """

    def ask_question(self, question: str, context: str):
        prompt = f"{self.SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def call_with_tools(self, messages, tools):
        """
        Call LLM with tool definitions and handle the response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        return response.choices[0].message


