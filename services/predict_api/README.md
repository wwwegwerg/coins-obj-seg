# predict_api

Оркестратор пайплайна: вызывает `florence_api` для детекции, затем `sam_api` для сегментации и возвращает агрегированный результат.

## Endpoints

- `GET /health` -> `{"status":"ok"}`.
- `GET /ready` -> проверяет доступность downstream-сервисов (`/health` у Florence и SAM), иначе `503`.
- `POST /predict` -> JSON с объектами и инстансами:
  - Request: `multipart/form-data`, поле `file` (изображение).
  - Response:
    - `objects`: уникальные label.
    - `instances`: список объектов с bbox, площадью маски, `bbox_mask_iou` и `png_base64`.
  - Ошибки: `400` (невалидный файл), `500` (ошибка оркестрации/вызова downstream).

## Environment

Пример: `.env.example`

- `FLORENCE_API_URL` (required), например `http://florence-api:8000`
- `SAM_API_URL` (required), например `http://sam-api:8000`
- `PREDICT_HTTP_TIMEOUT_SECONDS` (default: `300`)
- `PREDICT_READINESS_TIMEOUT_SECONDS` (default: `3`)

Без `FLORENCE_API_URL` или `SAM_API_URL` сервис не стартует.

## Run locally

Запустите Florence и SAM отдельно, затем:

```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

`predict_api` автоматически подгружает `.env` из директории сервиса при старте.
Убедитесь, что запуск выполняется из `services/predict_api` и в `.env` заданы `FLORENCE_API_URL` и `SAM_API_URL`.

Проверка:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Accept: application/json" \
  -F "file=@/path/to/image.jpg;type=image/jpeg"
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
      "mask_area": 15234,
      "bbox_mask_iou": 0.88,
      "png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```
