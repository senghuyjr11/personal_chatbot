from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import torch
import transformers
import warnings

# === Local Mistral model path ===
model_path = "./Mistral-7B"

# === Quantization Config (4-bit) ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# === Load Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

model = AutoModelForCausalLM.from_pretrained(
    "./Mistral-7B",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True
)

# === Pipeline Setup ===
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    temperature=0.1,
    top_p=0.5,
    repetition_penalty=1.2,
    do_sample=False
)

# === Embedding for Retrieval ===
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}
)

vectordb = Chroma(
    persist_directory="./rag_db",
    embedding_function=embedding_model
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# === Strict Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use only the context below to answer.
If the answer is not in the context, respond with: "I don't know".

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
    "What is your name?",
    "What is my cat name?"
]

for q in questions:
    response = qa.invoke(q)["result"].strip()

    # Extract only the part after the last 'Answer:'
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    print(f"\nQuestion: {q}")
    print(f"Answer: {response}")
