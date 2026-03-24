# florence_api

Сервис детекции объектов на базе Florence-2. Принимает изображение и возвращает список объектов с bbox.

## Endpoints

- `GET /health` -> `{"status":"ok"}`.
- `GET /ready` -> `{"status":"ready"}` если модель загружена, иначе `503`.
- `POST /detect` -> JSON с детекциями:
  - Request: `multipart/form-data`, поле `file` (изображение).
  - Response: `{"detections":[{"label":"...", "bbox":[x1,y1,x2,y2]}]}`.
  - Ошибки: `400` (не изображение / пустой файл / decode), `500` (ошибка инференса).

## Environment

Пример: `.env.example`

- `FLORENCE_MODEL_ID` (default: `microsoft/Florence-2-base-ft`)
- `FLORENCE_MODEL_DIR` (default в коде: `models/florence-2-base-ft`, в compose: `/models/florence-2-base-ft`)
- `PRELOAD_MODELS` (default: `true`) - загружать модель на startup.

## Run locally

```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

`florence_api` автоматически подгружает `.env` из директории сервиса при старте.
Убедитесь, что запуск выполняется из `services/florence_api`.

Проверка:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Example

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Accept: application/json" \
  -F "file=@/path/to/image.jpg;type=image/jpeg"
```

Пример ответа:

```json
{
  "detections": [
    {
      "label": "coin",
      "bbox": [123.4, 56.7, 220.1, 160.9]
    }
  ]
}
```
