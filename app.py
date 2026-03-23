import gradio as gr

import rag_engine


def _get_retrieve_fn():
	if hasattr(rag_engine, "recuperar_documentos"):
		return rag_engine.recuperar_documentos
	return rag_engine.get_documents


def _get_generate_fn():
	if hasattr(rag_engine, "generar_respuesta"):
		return rag_engine.generar_respuesta
	return rag_engine.get_answer


def ask(query, top_k, umbral):
	query = (query or "").strip()
	if not query:
		return "Please enter a question.", "No documents retrieved."

	retrieve_docs = _get_retrieve_fn()
	generate_answer = _get_generate_fn()

	docs = retrieve_docs(query, top_k=int(top_k), umbral=float(umbral))
	answer = generate_answer(query, docs)

	docs_formatted = "\n\n---\n\n".join(docs) if docs else "No documents retrieved."
	return answer, docs_formatted


with gr.Blocks(title="RAG Semantic Search Assistant") as demo:
	gr.Markdown("# RAG Semantic Search Assistant")
	gr.Markdown(
		"Ask a question in English and the assistant will answer using only "
		"the retrieved context from the knowledge base."
	)

	query_input = gr.Textbox(
		label="Question",
		placeholder="Example: Where is the hospital located?",
		lines=2,
	)
	top_k_input = gr.Slider(
		minimum=1,
		maximum=4,
		value=2,
		step=1,
		label="Top K documents",
	)
	umbral_input = gr.Slider(
		minimum=0.0,
		maximum=1.0,
		value=0.45,
		step=0.05,
		label="Similarity threshold (umbral)",
	)

	submit_button = gr.Button("Enviar")

	answer_output = gr.Textbox(label="Answer", lines=3, max_lines=3)
	docs_output = gr.Textbox(label="Retrieved documents", lines=1, max_lines=5)

	submit_button.click(
		fn=ask,
		inputs=[query_input, top_k_input, umbral_input],
		outputs=[answer_output, docs_output],
		api_name="ask",
	)


if __name__ == "__main__":
	demo.launch()
