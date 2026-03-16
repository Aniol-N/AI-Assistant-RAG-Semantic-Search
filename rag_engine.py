import json
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

model_name_llm = "PleIAs/Pleias-RAG350M"
fallback_model_name_llm = "PleIAs/Pleias-RAG-350M"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name_llm)
    model_llm = AutoModelForCausalLM.from_pretrained(model_name_llm)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(fallback_model_name_llm)
    model_llm = AutoModelForCausalLM.from_pretrained(fallback_model_name_llm)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_name_emb = "MongoDB/mdbr-leaf-ir"
fallback_model_name_emb = "sentence-transformers/all-mpnet-base-v2"

try:
    model_emb = SentenceTransformer(model_name_emb)
except Exception:
    model_emb = SentenceTransformer(fallback_model_name_emb)

with open("documents.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

if isinstance(raw_data, dict):
    documents = list(raw_data.values())
else:
    documents = [
        item["texto"] if isinstance(item, dict) and "texto" in item else str(item)
        for item in raw_data
    ]

print(f"Loaded {len(documents)} documents.")

document_embeddings = model_emb.encode(documents, convert_to_numpy=True)


def get_documents(query, top_k=2, umbral=0.4):
    if not query or not query.strip():
        return []

    query_embedding = model_emb.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    retrieved_documents = []
    for i in sorted_indices:
        if similarities[i] < umbral:
            continue
        retrieved_documents.append(documents[i])
        if len(retrieved_documents) >= top_k:
            break

    return retrieved_documents


def get_answer(query, retrieved_documents):
    docs = " ".join(retrieved_documents)
    prompt = (
        "Answer the question based only on the context provided\n"
        f"Context: {docs}\n"
        f"Question: {query}\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if output_text.startswith(prompt):
        return output_text[len(prompt) :].strip()
    return output_text.split("Answer:", 1)[-1].strip()


def ask(query, top_k=2, umbral=0.4):
    docs = get_documents(query, top_k, umbral)
    return get_answer(query, docs)


def main():
    user_input = input("Enter your question (or 'exit' to quit): ")
    answer = ask(user_input)
    print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
