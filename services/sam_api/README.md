# sam_api

Сервис сегментации на базе SAM. Принимает изображение и список bbox, возвращает ZIP с PNG-масками и метаданными.

## Endpoints

- `GET /health` -> `{"status":"ok"}`.
- `GET /ready` -> `{"status":"ready"}` если модель загружена, иначе `503`.
- `POST /segment` -> `application/zip`.
  - Request: `multipart/form-data`:
    - `file`: изображение.
    - `bboxes`: JSON-строка со списком bbox, например `[[10,20,120,140],[50,60,200,220]]`.
  - Response: ZIP:
    - `metadata.json`
    - `mask_000.png`, `mask_001.png`, ...
  - Ошибки: `400` (валидация файла/`bboxes`), `500` (ошибка инференса).

## Формат metadata.json

```json
{
  "instances": [
    {
      "detection_index": 0,
      "mask_filename": "mask_000.png",
      "mask_score": 0.93,
      "mask_area": 15234
    }
  ]
}
```

## Environment

Пример: `.env.example`

- `SAM_MODEL_ID` (default: `facebook/sam2.1-hiera-tiny`)
- `SAM_MODEL_DIR` (default в коде: `models/sam2.1-hiera-tiny`, в compose: `/models/sam2.1-hiera-tiny`)
- `PRELOAD_MODELS` (default: `true`) - загружать модель на startup.

## Run locally

```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

`sam_api` автоматически подгружает `.env` из директории сервиса при старте.
Убедитесь, что запуск выполняется из `services/sam_api`.

Проверка:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Example

```bash
curl -X POST "http://localhost:8000/segment" \
  -H "Accept: application/zip" \
  -F "file=@/path/to/image.jpg;type=image/jpeg" \
  -F 'bboxes=[[10,20,120,140],[50,60,200,220]]' \
  --output sam_result.zip
```
