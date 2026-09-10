import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from logging_config import get_logger
# from env import OPENAI_API_KEY

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger = get_logger(__name__)

class LLMservice:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o-mini"
        logger.info("LLM service initialized: model=%s api_key_configured=%s", self.model, bool(OPENAI_API_KEY))
        self.SYSTEM_PROMPT = """You are an Intelligent Assistant that answers questions based on the provided context.
        
        Rules:

        1. Use only provided context to answer questions. Do not make up information.

        2. If the answer is not present in the context, respond with 'I count not find the answer in the provided documents'

        3. Answer clearly and professinally. Avoid using filler words like 'umm' or 'ahh'.

        4. Keep the answer concise and to the point. Avoid unnecessary elaboration unless user explicitly requests it.

        """

    def extract_spo(self, text: str):
        """
        Extracts Subject-Predicate-Object (SPO) triplets from the given text.
        Returns a list of triplets: [(subject, predicate, object), ...]
        """
        prompt = f"""Extract all key factual relationships from the following text as Subject-Predicate-Object (SPO) triplets.
        Format each triplet as a JSON list: [["Subject", "Predicate", "Object"], ...]
        
        Text: {text}
        
        Return ONLY the JSON list. No preamble or explanation.
        """
        logger.info("SPO extraction started: input_characters=%d", len(text))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are an expert knowledge graph engineer. Extract structured SPO triplets from text."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            triplets = json.loads(response.choices[0].message.content.strip())
            logger.info("SPO extraction completed: triplets=%d", len(triplets))
            return triplets
        except Exception:
            logger.exception("SPO response parsing failed")
            return []

    def extract_entities(self, text: str):
        """
        Extracts key entities from the query to be used as seeds for graph search.
        """
        prompt = f"""Extract the main entities (people, places, organizations, events) from the following query.
        Return ONLY a JSON list of strings: ["Entity1", "Entity2", ...]
        
        Query: {text}
        """
        logger.info("Entity extraction started: query_characters=%d", len(text))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are an entity extractor. Return only a JSON list of strings."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            entities = json.loads(response.choices[0].message.content.strip())
            logger.info("Entity extraction completed: entities=%d", len(entities))
            return entities
        except Exception:
            logger.exception("Entity response parsing failed")
            return []

    def ask_question(self, question: str, context: str):
        logger.info("Answer generation started: question_characters=%d context_characters=%d", len(question), len(context))
        prompt = f"{self.SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        logger.info("Answer generation completed: answer_characters=%d", len(answer))
        return answer

    def call_with_tools(self, messages, tools):
        """
        Call LLM with tool definitions and handle the response.
        """
        logger.info("LLM tool call started: tools=%d messages=%d", len(tools), len(messages))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        message = response.choices[0].message
        logger.info("LLM tool call completed: tool_calls=%d", len(message.tool_calls or []))
        return message
