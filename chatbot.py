from typing import List, Any

import torch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Field
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

class SimpleStringLLM(LLM):
    pipeline: Any = Field(exclude=True)

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        return self.pipeline(prompt)

    def generate(self, prompts: List[str], **kwargs) -> LLMResult:
        generations = [[Generation(text=self._call(prompt))] for prompt in prompts]
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "simple-string-llm"

class StrOutputPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.task = getattr(pipeline, "task", None)

    def __call__(self, prompt, **kwargs):
        result = self.pipeline(prompt, **kwargs)
        if isinstance(result, list) and "generated_text" in result[0]:
            text = result[0]["generated_text"]
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)

        # Stop if model keeps generating follow-ups
        if "Question:" in text:
            text = text.split("Question:")[0].strip()

        return text.strip()

# === Load vector DB ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./rag_db", embedding_function=embedding_model)
retriever = vectordb.as_retriever()


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

pipe = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    return_full_text=False,
    clean_up_tokenization_spaces=True
)

llm = SimpleStringLLM(pipeline=StrOutputPipeline(pipe))


# === Prompt template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a friendly assistant. Use the information below to answer questions in a warm, conversational tone.

Context:
{context}

Question:
{question}

Answer:"""
)

# === QA chain ===
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# === Static input test ===
static_question = "Tell me about your routine"
# Run RAG
raw_answer = qa.invoke(static_question)["result"]
answer = raw_answer.strip()

print("\nQuestion:", static_question)
print("Answer:", answer)