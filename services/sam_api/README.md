# sam_api

## Endpoints [see main.py](./main.py)

- `GET /health` -> `{"status":"ok"}`.
- `GET /ready` -> `{"status":"ready"}` если модель загружена, иначе `503`.
- `POST /segment` -> `application/zip`.
  - Request: `multipart/form-data`:
    - `file`: изображение.
    - `bboxes`: JSON-строка со списком bbox, например `[[10,20,120,140],[50,60,200,220]]`.
  - Response: ZIP:
    - `metadata.json`
    - `mask_000.png`, `mask_001.png`, ...
    - PNG-маски полноразмерные (размер исходного изображения), grayscale (`L`), значения `0/255`.
  - Ошибки: `400` (валидация файла/`bboxes`), `500` (ошибка инференса).

## Формат metadata.json

[See contracts.py](./contracts.py).

## Environment

- `PRELOAD_MODELS` (default: `true`) - загружать модель на startup.

## Run locally

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8002
```

Проверка:

```bash
curl http://localhost:8002/health
curl http://localhost:8002/ready
```

## Example

```bash
curl -X POST "http://localhost:8002/segment" \
  -H "Accept: application/zip" \
  -F "file=@/path/to/image.jpg" \
  -F 'bboxes=[[10,20,120,140],[50,60,200,220]]' \
  --output sam_result.zip
```
