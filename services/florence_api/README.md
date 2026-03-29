# florence_api

## Endpoints [see main.py](./main.py)

- `GET /health` -> `{"status":"ok"}`.
- `GET /ready` -> `{"status":"ready"}` если модель загружена, иначе `503`.
- `POST /detect` -> JSON с детекциями:
  - Request: `multipart/form-data`, поле `file` (изображение).
  - Response: [see contracts.py](./contracts.py).
  - Ошибки: `400` (не изображение / пустой файл / decode), `500` (ошибка инференса).

`florence_api` не выполняет геометрическую нормализацию bbox. Нормализация и дополнительная постобработка выполняются в `predict_api`.

## Environment

- `PRELOAD_MODELS` (default: `true`) - загружать модель на startup.

## Run locally

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8001
```

Проверка:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/ready
```

## Example

```bash
curl -X POST "http://localhost:8001/detect" \
  -F "file=@/path/to/image.jpg"
```

Пример ответа:

```json
{
  "detections": [
    {
      "label": "coin",
      "bbox": [123.4, 56.7, 220.1, 160.9],
      "score": 0.94
    }
  ]
}
```
