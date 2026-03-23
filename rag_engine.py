import json
import re
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

model_name_emb = "MongoDB/mdbr-leaf-ir"
fallback_model_name_emb = "sentence-transformers/all-mpnet-base-v2"
model_name_llm = "PleIAs/Pleias-RAG-350M"
fallback_model_name_llm = "PleIAs/Pleias-RAG350M"

try:
    model_emb = SentenceTransformer(model_name_emb)
except Exception:
    model_emb = SentenceTransformer(fallback_model_name_emb)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name_llm)
    model_llm = AutoModelForCausalLM.from_pretrained(model_name_llm)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(fallback_model_name_llm)
    model_llm = AutoModelForCausalLM.from_pretrained(fallback_model_name_llm)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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


def get_documents(query, top_k=2, umbral=0.45):
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


def _split_sentences(text):
    # Keep a simple sentence split to return verbatim snippets from context.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _tokenize(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _extractive_answer(query, retrieved_documents, max_words=35):
    if not retrieved_documents:
        return "I don't have enough information in the retrieved documents."

    query_tokens = _tokenize(query)
    best_sentence = ""
    best_score = -1.0

    for doc in retrieved_documents:
        for sentence in _split_sentences(doc):
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue

            overlap = len(query_tokens & sentence_tokens)
            coverage = overlap / max(len(query_tokens), 1)

            # Prefer concise extractive answers when scores tie.
            brevity_bonus = 1.0 / (1.0 + len(sentence.split()))
            score = coverage + 0.15 * brevity_bonus

            if score > best_score:
                best_score = score
                best_sentence = sentence

    if not best_sentence:
        best_sentence = retrieved_documents[0].strip()

    words = best_sentence.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(".,;:") + "..."
    return best_sentence


def _is_answer_supported_by_context(answer, documents, min_ratio=0.35):
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return False
    context_tokens = _tokenize(" ".join(documents))
    overlap = len(answer_tokens & context_tokens)
    ratio = overlap / max(len(answer_tokens), 1)
    return ratio >= min_ratio


def _llm_generate_answer(query, retrieved_documents, max_new_tokens=45):
    context = " ".join(retrieved_documents)
    prompt = (
        "Answer the question based only on the context provided\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0][input_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def recuperar_documentos(consulta, top_k=2, umbral=0.45):
    return get_documents(consulta, top_k=top_k, umbral=umbral)


def generar_respuesta(consulta, documentos_recuperados):
    if not documentos_recuperados:
        return "I don't have enough information in the retrieved documents."

    llm_answer = _llm_generate_answer(consulta, documentos_recuperados, max_new_tokens=45)
    if llm_answer and _is_answer_supported_by_context(llm_answer, documentos_recuperados):
        words = llm_answer.split()
        if len(words) > 35:
            return " ".join(words[:35]).rstrip(".,;:") + "..."
        return llm_answer

    # If the generated text drifts from context, return a strict extractive fallback.
    return _extractive_answer(consulta, documentos_recuperados, max_words=35)


def preguntar(consulta, top_k=2, umbral=0.45):
    docs = recuperar_documentos(consulta, top_k=top_k, umbral=umbral)
    return generar_respuesta(consulta, docs)


def get_answer(query, retrieved_documents):
    return generar_respuesta(query, retrieved_documents)


def ask(query, top_k=2, umbral=0.45):
    return preguntar(query, top_k=top_k, umbral=umbral)


def main():
    user_input = input("Enter your question (or 'exit' to quit): ")
    answer = ask(user_input)
    print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
