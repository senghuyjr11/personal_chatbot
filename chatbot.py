from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import torch
from langchain_huggingface import HuggingFaceEmbeddings

# === Quantization Config ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# === Model Loading ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# === Pipeline Setup ===
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,       # Limit response length
    temperature=0.2,          # Lower temp = less verbose
    top_p=0.8,                # Top-p filtering for focus
    repetition_penalty=1.1,   # Penalize repetition
    do_sample=False           # Disable sampling for deterministic output
)

# === Vector DB Setup ===
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}
)
vectordb = Chroma(
    persist_directory="./rag_db",
    embedding_function=embedding_model
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# === Concise Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the following question briefly using only the context provided.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
)

# === QA Chain ===
qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=pipe),
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False
)

# === Test Questions ===
questions = [
    "What is your name and where are you from?",
]

for q in questions:
    response = qa.invoke(q)["result"].strip()

    # Extract only the part after the last 'Answer:'
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    print(f"\nQuestion: {q}")
    print(f"Answer: {response}")
