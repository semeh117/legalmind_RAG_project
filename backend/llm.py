import os
from groq import Groq
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(question: str ,chunks : list[dict]) -> str:

   """Build a prompt for the LLM using the retrieved chunks and the user's question."""
   
   context = ""
   for result in zip(chunks["documents"][0],chunks["metadatas"][0]):
      txt, metadata = result
      context += f" [Page {metadata['page_number']}]:\n{txt}\n\n"

   return f"""You are a legal consultant. Answer the user's question based ONLY on the document excerpts below.



DOCUMENT EXCERPTS:
{context}

RULES:
- Be clear, simple and well organized
- Maximum 5 lines
- Always mention the page number where you found the answer
- If the answer is not in the document, say "I cannot find this in the provided document"
- If the question is unclear, ask the user for more information

QUESTION: {question}
"""     
def get_answer(question: str, chunks : list[dict]) -> str:
   """Get an answer from the LLM based on the question and the retrieved chunks."""
   prompt = build_prompt(question, chunks)
   response = client.chat.completions.create(
      model="openai/gpt-oss-120b",
      messages=[
         {"role": "user", "content": prompt}
      ],
      max_tokens=500
   )
   return response.choices[0].message.content
