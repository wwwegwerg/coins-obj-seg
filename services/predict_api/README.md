# predict_api

Оркестратор и центр постобработки пайплайна: вызывает `florence_api` (детекция) и `sam_api` (raw маски), а затем выполняет всю вторичную обработку и формирует финальный ответ.

## Endpoints

- `GET /health` -> `{"status":"ok"}`.
- `GET /ready` -> проверяет доступность downstream-сервисов (`/health` у Florence и SAM), иначе `503`.
- `POST /predict` -> JSON с объектами и инстансами:
  - Request: `multipart/form-data`, поле `file` (изображение).
  - Response:
    - `objects`: уникальные label.
    - `instances`: список объектов с bbox, `bbox_mask_iou` и `png_base64`.
  - Ошибки: `400` (невалидный файл), `500` (ошибка оркестрации/вызова downstream).

В `predict_api` выполняются:

- постобработка детекций Florence до SAM:
  - confidence threshold;
  - box sanity checks;
- нормализация bbox;
- вырезка объекта по маске (PNG с прозрачным фоном);
- вычисление `bbox_mask_iou`;
- mask-level NMS (дедуп почти одинаковых масок по IoU масок);
- сборка финального ответа.

Параметры постобработки Florence захардкожены в `service.py`:

- `CONFIDENCE_THRESHOLD = 0.0`
- `MIN_BOX_WIDTH_PX = 4.0`
- `MIN_BOX_HEIGHT_PX = 4.0`
- `MIN_BOX_AREA_RATIO = 0.0001`
- `MAX_BOX_AREA_RATIO = 0.95`
- `MIN_BOX_ASPECT_RATIO = 0.2`
- `MAX_BOX_ASPECT_RATIO = 5.0`
- `MIN_CUTOUT_WIDTH_PX = 32`
- `MIN_CUTOUT_HEIGHT_PX = 32`
- `MIN_MASK_SCORE = 0.7`
- `MIN_BBOX_MASK_IOU = 0.3`
- `MASK_NMS_IOU_THRESHOLD = 0.88`

Финальная фильтрация инстансов после SAM выполняется через `AND`:

- `mask_score >= MIN_MASK_SCORE`
- `bbox_mask_iou >= MIN_BBOX_MASK_IOU`

После этого применяется `mask-level NMS`: если IoU двух бинарных масок >= `MASK_NMS_IOU_THRESHOLD`,
остается только более приоритетный кандидат.

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
      "bbox_mask_iou": 0.88,
      "png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```
