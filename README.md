# Распознавание ценников на видео торгового зала

Автор: **Цыренов Мэргэн**. Python-пакет: `lenta_price_tag_scanner`.

## Постановка задачи

Проект решает задачу извлечения структурированных данных с ценников по видео
торгового зала. Система обнаруживает ценники, объединяет наблюдения во времени,
выбирает читаемые изображения, распознаёт текст и формирует CSV-таблицу.
Постановка основана на задаче хакатона Lenta Tech.

## Формат входных и выходных данных

Входные данные — каталог видео в формате MP4, включая записи в разрешении 4K.
Ориентация кадров и шаг обработки задаются конфигурацией.

Результат — `submission.csv` в кодировке UTF-8 BOM с 29 столбцами:
18 основных и 11 дополнительных QR-полей. Временная отметка `frame_timestamp`
измеряется в миллисекундах, координаты bbox — в пикселях исходного кадра.
Пустое значение обозначает отсутствие достоверно извлечённого поля.
Одна строка соответствует сегменту трека; один физический ценник может быть
представлен несколькими сегментами.

Пример видео, изображение ценника и веса моделей представлены DVC-артефактом
`data/inference_assets.dvc`. Описание схемы и выходных артефактов:
[видеопайплайн](docs/video_to_csv.md).

## Метрики

Основная метрика — доля правильно заполненных полей относительно числа
эталонных полей. Предсказания сопоставляются с эталонными CSV автоматически
по видео, временным отметкам и координатам ценников. Пропущенные соответствия
учитываются в знаменателе.

| Компонент                                | Выборка            | Результат                                         |
| ---------------------------------------- | ------------------ | ------------------------------------------------- |
| Полный пайплайн + EfficientNet rank, OOF | 5 видео, 3701 поле | Micro: **152/3701 = 4.1070%**, macro: **4.3018%** |
| Пайплайн без EfficientNet, OOF           | 5 видео, 3701 поле | Micro: **154/3701 = 4.1610%**, macro: **4.3896%** |
| EfficientNet, OOF                        | 2008 изображений   | Precision: 0.8645, recall: 0.5286, F1: 0.6560     |
| Qwen3-VL-4B, OCR                         | 294 транскрипции   | CER: 0.5869                                       |

Новый результат полного пайплайна получен с PaddleOCR + Qwen3-VL и EfficientNet
в режиме ранжирования. Сохранённая оценка без классификатора приведена отдельно.
[Результаты по видео и протокол](docs/efficient_video_oof_results.md).
Метрика правильности полей отличается от
заполненности CSV и официальной метрики хакатона.
Подробные результаты по видео приведены в [OOF-отчёте](docs/video_oof_results.md).
Графики и журналы экспериментов находятся в [plots](plots).

## Валидация

Для YOLO и EfficientNet используется пятикратная кросс-валидация с разделением
по видео. Все изображения одного видео относятся к одному внешнему fold.
Каждая модель обучается независимо; checkpoint сохраняется после фиксированного
числа эпох. Для YOLO используется `last.pt`. Внешний fold не применяется для
выбора лучшего checkpoint.

Основной YOLO-датасет содержит 391 кадр и 2593 bbox. Все пять моделей обучены
по 100 эпох. Исходный YOLO-набор с четырьмя группами видео доступен как отдельная
версия данных. OCR-набор используется для сравнения вариантов распознавания;
его результаты интерпретируются как оценка на development-выборке.

Схема разбиения и команды обучения:
[YOLO OOF](docs/yolo_roboflow_oof.md), [OOF-обучение](docs/oof_training.md).

## Данные

| Набор                             | Состав                                                   | DVC-указатель                                |
| --------------------------------- | -------------------------------------------------------- | -------------------------------------------- |
| Видео и эталонная разметка        | 5 видео, CSV, OCR-данные и веса PaddleOCR                | `data/runtime_assets.dvc`                    |
| YOLO SAM3/Roboflow                | 391 кадр, 2593 bbox, 5 folds                             | `data/yolo_roboflow_5fold_v1.dvc`            |
| Исходный YOLO                     | 251 изображение, 2118 bbox, 4 folds                      | `data/yolo_original_4fold.dvc`               |
| EfficientNet                      | 2008 изображений: 350 положительных и 1658 отрицательных | `data/efficient_user_20261102.dvc`           |
| Итоговые YOLO-модели и OOF-отчёты | 5 checkpoint и результаты видеооценки                    | `data/video_oof_20261104.dvc`                |
| EfficientNet и новый OOF          | 5 классификаторов, CSV и отчёты с режимом rank           | `data/video_oof_efficient_rank_20261105.dvc` |
| Артефакты инференса               | YOLO, EfficientNet, каталог и короткое видео             | `data/inference_assets.dvc`                  |
| Псевдоразметка                    | Детекции старой YOLO и метаданные отбора                 | `data/pseudo_teacher_v1.dvc`                 |
| Экспорт YOLO                      | ONNX и MLflow-пакет с проверкой инференса                | `data/yolo_deployment_v1.dvc`                |

OCR-набор содержит 294 изображения и 107 групп наблюдений ценников.
Источник исходных материалов — [набор организаторов](https://drive.google.com/drive/folders/1_UbQ7x4MK9fZjA-9DqJY_A8nedsg9KMy).
Локальный товарный каталог содержит поля `fullname` и `code` и используется
для уточнения наименования по однозначному совпадению кода.

Датасеты, веса и результаты предсказаний версионируются через DVC. В Git
хранятся код, конфигурации, документация, графики и DVC-указатели.
Основное хранилище данных — MinIO; [состав артефактов и настройка доступа](docs/dvc_handoff.md).

## Моделирование

### Бейзлайн

[YOLO](https://docs.ultralytics.com/) → классификация читаемости
[EfficientNet/timm](https://huggingface.co/docs/timm/) → [EasyOCR](https://www.jaided.ai/easyocr/).
Обучение EfficientNet реализовано через [Lightning](https://lightning.ai/docs/pytorch/stable/),
обучение YOLO — через Ultralytics.

### Основная модель

YOLO → BoT-SORT → фильтры качества → выбор кропов →
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) +
[Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) →
декодирование QR/EAN → сопоставление с каталогом → CSV.

Qwen3-VL используется как vision-language model для дополнительного распознавания
текста. Ревизия весов зафиксирована в конфигурации. Каталог применяется при
однозначном совпадении кода; нечёткие совпадения сохраняются как предложения.

CPU-профиль использует PaddleOCR и EfficientNet в режиме `efficient_mode=rank`.
Классификатор участвует в ранжировании кропов без порогового отбрасывания.
VLM включается отдельным параметром. Для режима `rank` выполнен полный OOF-прогон
с весами обоих компонентов, соответствующими отложенному видео: 4.1070%
правильных полей. Улучшения относительно сохранённого запуска без классификатора нет.

Для известных видео YOLO-checkpoint выбирается по `model_by_video.json`.
Проверяются SHA256 весов и отсутствие отложенного видео в обучающей группе.
Для остальных видео используется заданный в конфигурации checkpoint по умолчанию.
[Выбор моделей и deployment](docs/yolo_deployment.md).

## Setup

Основная платформа — Linux x86-64, Python **3.12**. Для GPU-режима используется
CUDA **12.8**. Версии Torch 2.11.0+cu128 и torchvision 0.26.0+cu128 закреплены
в Poetry lock через официальный wheel index. CPU-режим поддерживается;
инструкции для Windows приведены в [описании CPU-запуска](docs/laptop_run.md).

Установка полного окружения из корня репозитория:

```bash
python -m pip install poetry==2.2.1
poetry env use python3.12
poetry install --with data,train,ocr,vlm,paddle,export,serve
poetry run pre-commit install
```

Установка зависимостей CPU-инференса без групп обучения и VLM:

```bash
poetry install --with data,ocr,paddle
```

Команды обучения и инференса разделены. Lightning и MLflow входят в отдельные
группы зависимостей и не импортируются основным видеоинференсом.
YAML-конфигурации включены в состав Python-пакета.

### Доступ к DVC

Основной remote `minio` — закрытый S3-бакет `lenta-dvc` с публичным HTTPS endpoint
**`https://31.31.207.108`**. Для скачивания нужны S3-ключи MinIO с правом чтения;
SSH-доступ и туннель не требуются. TLS-сертификат проверяется клиентом.
[Состав артефактов и настройка доступа](docs/dvc_handoff.md).

В корне репозитория укажите выданные S3-ключи только в локальном конфиге:

```bash
poetry run dvc remote modify --local minio access_key_id '<ACCESS_KEY>'
poetry run dvc remote modify --local minio secret_access_key '<SECRET_KEY>'
# Наборы, необходимые команде download:
poetry run price-scanner download
# Все девять версионируемых артефактов:
poetry run dvc pull -r minio
```

`poetry install --with data` устанавливает зависимости; загрузка артефактов запускается
отдельно. Для передачи данных без ML-зависимостей достаточно отдельного окружения с
`python -m pip install "dvc[s3]==3.67.1"`.

Учётные данные сохраняются в `.dvc/config.local`, который игнорируется Git.
`train`, `infer`, `export` и `serve-package` по умолчанию вызывают DVC pull.
Для уже подготовленных внешних файлов можно явно указать `pull_dvc=false`.

Функция `download_data()` и CLI для публичных исходников:

```bash
poetry run price-scanner download source=public output=data/raw_public
```

Она не подменяет ручную разметку или обученные веса скачанными оригиналами.

## Train

Адрес MLflow задаётся параметром `mlflow_uri`; значение по умолчанию —
**http://127.0.0.1:8080**. Запуск локального сервера:

```bash
mkdir -p data/control_runs
poetry run mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///data/control_runs/mlflow.db --artifacts-destination data/control_runs/mlflow_artifacts
```

Команды обучения выполняются в отдельном терминале. Полное обучение:

```bash
poetry run price-scanner train output=runs/efficient_oof
poetry run price-scanner train-yolo output=runs/yolo_roboflow_5fold_full
```

Обучение на исходном YOLO-наборе:

```bash
poetry run price-scanner train oof=yolo dataset=data/yolo_original_4fold output=runs/yolo_oof
```

Контрольные CPU-профили для проверки обучения и логирования:

```bash
poetry run python -m lenta_price_tag_scanner.train_oof --config-name smoke_efficient
poetry run python -m lenta_price_tag_scanner.train_oof --config-name smoke_yolo
```

Контрольный профиль использует один fold, две эпохи и по восемь изображений
обучающей и валидационной выборок. MLflow сохраняет параметры, фактическое число
эпох, loss, итоговые метрики, Git commit, dirty-флаг, конфигурации и графики.
Для каждого fold сохраняются checkpoint, предсказания и история обучения.

Параметры обучения находятся в `configs/oof` и `configs/train_oof.yaml`.
Доступные GPU задаются переменной `CUDA_VISIBLE_DEVICES`.
Правила подготовки версий данных описаны в
[подготовке датасетов](docs/dataset_preparation.md), результаты контроля — в
[протоколе MLflow](docs/training_control.md).

## Production preparation

Экспорт и упаковка выполняются для обученного checkpoint. Примеры команд:

```bash
poetry run price-scanner export output=artifacts/efficient_onnx
poetry run price-scanner serve-package output=artifacts/efficient_mlflow
poetry run price-scanner export-yolo output=artifacts/yolo_onnx
poetry run price-scanner serve-yolo-package output=artifacts/yolo_mlflow
```

Для других весов задаётся параметр `checkpoint=/absolute/path/to/model.pt`.
Каталог `output` должен быть новым.

EfficientNet ONNX принимает RGB-изображения с ImageNet-нормализацией в формате
float32 NCHW, размером 320 × 320 и динамическим batch. Выход — logits;
вероятность вычисляется функцией sigmoid. Экспорт проверяется ONNX checker
и сравнением с PyTorch для batch 1 и 2.

YOLO ONNX принимает RGB float32 NCHW после letterbox и нормализации в [0, 1].
Постпроцессинг включает NMS и восстановление координат. Экспорт проверяется
сравнением детекций с PyTorch на изображении из DVC-набора.

Поставка включает модель ONNX или MLflow bundle, конфигурацию, описание
препроцессинга, отчёт проверки, Poetry lock и DVC-указатели.
Qwen3-VL используется в формате Transformers.

Экспорт EfficientNet в TensorRT при установленном `trtexec`:

```bash
poetry run price-scanner export format=tensorrt output=artifacts/efficient_trt
```

## Infer

### Видео → CSV

Запуск после обучения с явным указанием checkpoint:

```bash
poetry run price-scanner infer videos=/absolute/path/to/new_videos yolo_weights=runs/yolo_roboflow_5fold_full/fold_0/training/weights/last.pt efficient_weights=runs/efficient_oof/fold_0/model.pt output=data/new_video_result
```

При отсутствии явного `yolo_weights` применяется карта моделей для известных
видео либо checkpoint по умолчанию для остальных. Применение фиксированного
checkpoint к новому видео является инференсом, а не OOF-оценкой.

Короткий пример на CPU:

```bash
poetry run price-scanner infer detector_device=cpu max_frames=7 top_k=1 output=data/infer_example
```

CPU-профиль с видео и локальными PaddleOCR-весами из DVC:

```bash
poetry run price-scanner infer-cpu output=data/cpu_video_result
```

Режим с Qwen3-VL на GPU:

```bash
poetry run price-scanner infer videos=/absolute/path/to/videos detector_device=cuda top_k=2 efficient_mode=rank vlm_fallback=true vlm_limit=1000 vlm_name_override=true output=data/video_result
```

Первоначальная загрузка OCR/VLM-весов требует доступа к источнику моделей.
После загрузки вычисления выполняются локально. Для каждого запуска требуется
новый каталог результатов. Дополнительные отчёты создаются командами:

```bash
poetry run python -m scripts.review_video_csv data/video_result
poetry run python -m lenta_price_tag_scanner.field_extraction data/video_result/submission.csv --output=data/video_result/coverage_recomputed.json
```

### MLflow Serving

HTTP-сервисы выполняют классификацию изображения ценника или детекцию YOLO.
Полный видеопайплайн запускается отдельной batch-командой.

Сервис EfficientNet:

```bash
poetry run mlflow models serve -m artifacts/efficient_mlflow --env-manager local --host 127.0.0.1 --port 18081
```

Проверочный запрос из отдельного терминала:

```bash
poetry run python -m scripts.serve_smoke
```

Сервис YOLO и пример запроса:

```bash
poetry run mlflow models serve -m artifacts/yolo_mlflow --env-manager local --host 127.0.0.1 --port 18082
```

```bash
curl -H "Content-Type: application/json" --data-binary @artifacts/yolo_mlflow/request.json http://127.0.0.1:18082/invocations
```

Оба сервиса принимают POST `/invocations` в формате MLflow `dataframe_split`
с колонкой `image_base64`. EfficientNet возвращает вероятность читаемости
`probability`, YOLO — `detections_json` с координатами и confidence и число
детекций `num_detections`. Готовые JSON-примеры создаются при упаковке моделей.

## Проверки и документация

```bash
poetry run pre-commit run --all-files
poetry run python -m unittest discover -s tests
```

CI выполняет проверку стиля, unit-тесты и проверку конфигураций.
Контроль обучения, ONNX и HTTP-инференса описан в соответствующих протоколах.

- [Навигация по документации](docs/README.md).
- [Проектное предложение](docs/proposal.pdf).
- [Соответствие требованиям](docs/compliance.md).
- [Управление данными и артефактами](docs/dvc_handoff.md).
- [YOLO ONNX и MLflow Serving](docs/yolo_deployment.md).
