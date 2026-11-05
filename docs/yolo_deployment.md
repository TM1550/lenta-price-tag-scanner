# YOLO: инференс, ONNX и MLflow

Текущий инференс выбирает новый YOLO checkpoint по имени видео через
`data/video_oof_20261104/model_by_video.json`. Для известных видео проверяются SHA256
и отсутствие видео в обучающем списке. Для нового видео используется fold 0;
это deployment-default, а не обещание OOF для произвольного входа. Явный параметр
`yolo_weights=...` имеет приоритет. После нового обучения передайте его `last.pt`.
Выбор моделей и SHA сохраняются в `detectors.json` и `metrics.json`.

EfficientNet включён в режиме `efficient_mode=rank`: геометрическая оценка кропа
умножается на `1 - weight + weight * probability`, по умолчанию weight=0.5.
Низкая вероятность сама по себе не удаляет кроп. `filter` включает строгий порог,
`off` отключает классификатор. Старый OOF запуск использовал `off`: его 4.1610%
не является измерением качества нового режима.
[Новый полный OOF-прогон с классификатором](efficient_video_oof_results.md) получил
152/3701 = 4.1070%. Для него использованы собственные held-out веса EfficientNet
каждого видео; общий inference-checkpoint в этой оценке не применялся.

## Подготовка к production

```bash
poetry install --with data,train,export,serve
poetry run price-scanner export-yolo
poetry run price-scanner serve-yolo-package
```

Если установленный entrypoint отличается, эквивалентная команда:
`python -m lenta_price_tag_scanner.commands export-yolo`.
Обе команды подтягивают DVC-веса и пример кадра. Экспорт копирует исходные веса в
новую выходную папку и не меняет DVC-источник. Повторный запуск требует новой `output`.
ONNX проверяется checker и сравнением детекций с PyTorch на реальном кадре.
Вход ONNX: RGB NCHW float32 /255 после letterbox. Выход — xywh и вероятности классов;
NMS и обратное преобразование координат выполняет Ultralytics. Для ONNX-инференса
можно передать `yolo_weights=data/yolo_deployment_v1/onnx/model.onnx`.

## MLflow Serving YOLO

Готовые проверенные ONNX/MLflow-пакеты доступны через
`dvc pull -r minio data/yolo_deployment_v1.dvc` после
[настройки S3-ключей и HTTPS-доступа](dvc_handoff.md).
ONNX: 8 совпавших детекций, максимальная абсолютная разница 0.00024414.
MLflow HTTP: статус 200, результат совпадает с локальной загрузкой PyFunc.

```bash
poetry run mlflow models serve -m data/yolo_deployment_v1/mlflow --env-manager local --host 127.0.0.1 --port 8082
curl -H "Content-Type: application/json" --data-binary @data/yolo_deployment_v1/mlflow/request.json http://127.0.0.1:8082/invocations
```

Вход: `dataframe_split` с колонкой `image_base64`; выход: `detections_json`
(рамки xyxy в пикселях исходного изображения, confidence, class_id) и число рамок.
Сервис детектора работает на CPU и не заменяет полный видеопайплайн с OCR.
MLflow PyFunc — стандартный способ упаковки пользовательской модели; ONNX —
поддерживаемый Ultralytics формат. Источники:
[Ultralytics export](https://docs.ultralytics.com/modes/export/),
[MLflow PythonModel](https://mlflow.org/docs/latest/ml/model/python_model/).

## Логирование экспериментов

Обучение YOLO уже логирует folds, параметры, метрики и графики через MLflow в
`train_oof.py`. Сокомандник может позже выполнить новый тест и записать результаты
с актуальным Git commit, конфигурацией и SHA256 весов. Это будет новый тестовый run.
Исторические серверные runs имели `git_commit=uncommitted`; не следует задним числом
выдавать текущий commit за версию того запуска. Их исходники подтверждает SHA256-опись.
