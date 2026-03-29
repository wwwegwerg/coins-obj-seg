# predict_api

## Endpoints [see main.py](./main.py)

- `GET /health` -> `{"status":"ok"}`.
- `GET /ready` -> проверяет доступность downstream-сервисов (`/health` у Florence и SAM), иначе `503`.
- `POST /predict` -> JSON с объектами и инстансами:
  - Request: `multipart/form-data`, поле `file` (изображение).
  - Response: [see contracts.py](./contracts.py).
  - Ошибки: `400` (невалидный файл), `500` (ошибка оркестрации/вызова downstream).

## Environment

- `FLORENCE_API_URL` (required), например `http://florence-api:8000`
- `SAM_API_URL` (required), например `http://sam-api:8000`
- `PREDICT_HTTP_TIMEOUT_SECONDS` (default: `300`)
- `PREDICT_READINESS_TIMEOUT_SECONDS` (default: `3`)

## Run locally

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Accept: application/json" \
  -F "file=@/path/to/image.jpg"
```

Пример ответа:

```json
{
  "objects": ["coin"],
  "instances": [
    {
      "label": "coin",
      "mask_score": 0.93,
      "bbox": [123.4, 56.7, 220.1, 160.9],
      "bbox_mask_iou": 0.88,
      "png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```
