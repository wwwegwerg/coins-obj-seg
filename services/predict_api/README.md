# predict_api

## Endpoints [see app/main.py](./app/main.py)

- `GET /health` -> `{"status":"ok"}`.
- `GET /ready` -> проверяет доступность downstream-сервисов (`/health` у Florence и SAM), иначе `503`.
- `POST /predict` -> JSON с объектами и инстансами:
  - Request: `multipart/form-data`, поле `file` (изображение).
  - Response: [see app/contracts.py](./app/contracts.py).
  - Ошибки: `400` (невалидный файл), `500` (ошибка оркестрации/вызова downstream).

## Configuration

- Базовые значения лежат в [app/constants.py](./app/constants.py).
- При необходимости можно переопределить `FLORENCE_API_URL`, `SAM_API_URL`, `PREDICT_HTTP_TIMEOUT_SECONDS`, `PREDICT_READINESS_TIMEOUT_SECONDS` обычными переменными окружения.

## Run locally

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
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
