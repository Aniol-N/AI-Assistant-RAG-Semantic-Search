# Una vez hemos hecho el deploy en Hugging Face, podemos probar la API de Gradio
# usando el cliente de Gradio (https://github.com/gradio-app/gradio-client)

from gradio_client import Client

client = Client("manuell01/practica-chat")
result = client.predict(
	query="Where is the hospital located!!",
	top_k=5,
	umbral=0.55,
	api_name="/ask"
)
print(result[0])