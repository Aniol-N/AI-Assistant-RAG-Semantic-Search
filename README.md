# AI Assistant RAG Semantic Search

Asistente de preguntas y respuestas basado en RAG (Retrieval-Augmented Generation) que responde usando únicamente contexto recuperado de una base documental.

## Demo en producción
- Hugging Face Space: https://huggingface.co/spaces/anioln/AI-Assistant-RAG-Semantic-Search

## Problema que resuelve
Este proyecto permite consultar documentos y obtener respuestas con trazabilidad al contexto recuperado, reduciendo alucinaciones frente a un LLM puro.

## Cómo funciona (pipeline RAG)
1. Embedding de documentos y consulta.
2. Cálculo de similitud coseno.
3. Recuperación Top-k con umbral de similitud.
4. Generación de respuesta con LLM.
5. Verificación de soporte en contexto y fallback extractivo.

## Features
- Búsqueda semántica configurable (`top_k`, `umbral`).
- Respuestas controladas por contexto.
- Fallback extractivo cuando la generación no está soportada.
- Interfaz desplegada en Hugging Face.

## Stack técnico
- Python
- sentence-transformers
- transformers
- scikit-learn
- numpy
- torch
- Gradio/HF Spaces (despliegue)

## Estructura del proyecto
- `app.py`: interfaz y/o endpoints de ejecución.
- `rag_engine.py`: lógica principal de retrieval + generación.
- `documents.json`: base documental de ejemplo.
- `tests/`: pruebas automáticas.
- `requirements.txt`: dependencias.

## Instalación local
```bash
git clone https://github.com/TU_USUARIO/AI-Assistant-RAG-Semantic-Search.git
cd AI-Assistant-RAG-Semantic-Search
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución
```bash
python app.py
```

## Preview

### Vista rápida
- Prueba en vivo: https://huggingface.co/spaces/anioln/AI-Assistant-RAG-Semantic-Search
- Flujo visible en la UI: pregunta -> respuesta -> documentos recuperados.

### Capturas (UI)

![chrome_Ilxk5mb3Ei](https://github.com/user-attachments/assets/768c8122-9e14-46b0-bc1a-af938e525d89)
![chrome_pISXj1mUCx](https://github.com/user-attachments/assets/0372f459-d77c-4323-8743-781fa0f063e7)

## Dataset de ejemplo
El contenido base usado para estas pruebas está en `documents.json` e incluye:
- Datos de contacto del hospital.
- Horario de atención.
- Correo oficial.
- Servicios principales.
- Ubicación del hospital.
