# Student Early Stress Detection API

## Cómo arrancar
- **Windows:** Haz doble clic en `iniciar_api.bat` (requiere Python 3.10+ instalado).
- **Mac / Linux:** Ejecuta `bash iniciar_api.sh`.

Una vez iniciada:
- **URL base para la web:** `http://127.0.0.1:8000`
- **Probar endpoints visualmente:** `http://127.0.0.1:8000/docs`

## Endpoints para conectar la web
- `GET /health` -> Verifica si el modelo y el contrato están cargados.
- `GET /contract` -> Devuelve preguntas, rangos y textos del formulario.
- `GET /test-samples?limit=10` -> Devuelve muestras de `.test` ya procesadas.
- `POST /predict` -> Recibe `{ "features": { ... } }` y devuelve la predicción y factores de peso.