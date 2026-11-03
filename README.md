# Распознавание ценников на видео торгового зала

Автор: **Цыренов Мэргэн**. Пакет: `lenta_price_tag_scanner`.

## Постановка задачи

Робот снимает полки магазина на камеру 4K. Система находит ценники, выбирает читаемые
кадры, извлекает текст и формирует таблицу для цифрового мониторинга торгового зала.
Это самостоятельное оформление нашего решения хакатона Lenta Tech; открытые решения
других команд использованы как источники архитектурных идей, а не как готовый проект.

## Формат входных и выходных данных

Вход: папка MP4-видео; ориентация кадров задаётся конфигурацией. Выход: `submission.csv`
в UTF-8 BOM с 29 столбцами: 18 основных из исходной заявки/схемы и 11 дополнительных
QR-полей. `frame_timestamp` — миллисекунды, bbox — пиксели исходного кадра.
Пустая строка означает неизвестное значение. Одна строка соответствует сегменту трека;
гарантированная дедупликация физических ценников пока не реализована.

Реальный короткий пример видео, кроп и необходимые веса входят в DVC-артефакт
`data/inference_assets.dvc`. Спецификация выхода и результаты: [video_to_csv](docs/video_to_csv.md).

## Метрики

Цель исходной заявки: **доля правильно найденных параметров товара > 0.04** против
результата команды 0.02 на хакатоне. Эта цель пока **не подтверждена**: нужен scorer
соответствия товаров и эталонных полей. Её нельзя заменять заполненностью CSV.

Проверенные результаты:

- EfficientNet: pooled OOF precision 0.8645, recall 0.5286, F1 0.6560 на 2008 кропах.
- Qwen3-VL-4B: CER 0.5869 на 294 транскрипциях; сравнение восьми вариантов не улучшило CER.
- Полный видеопайплайн: 430 строк, 916/9890 заполненных содержательных ячеек (9.26%).
  Без VLM на тех же кропах — 7.38%. Это покрытие, **не точность**.

Графики и текстовые логи: [plots](plots). Ошибки OCR и возможные дубли требуют проверки.

## Валидация

EfficientNet: 5 folds, целое видео удерживается во внешнем fold. YOLO: **4 folds**,
поскольку исходная детекторная разметка содержит только четыре видео; это согласованное
отклонение от первоначальной заявки. Случайное разделение соседних кадров не используется.
Модель каждого fold обучается заново, checkpoint берётся после фиксированного числа эпох.
Неизменённые псевдометки не объявляются ручным эталоном. Для подбора гиперпараметров
нужна внутренняя валидация; внешний fold не используется для выбора лучшего checkpoint.
OCR-набор использован для сравнения решений и не является нетронутым test.

## Данные

Пять исходных видео, датасет YOLO (251 изображение, 2118 bbox), снимок EfficientNet
(2008 кропов: 350/1658), OCR-набор (294 изображения, 107 track-групп).
Из ручной правки исключены два кропа, присутствовавшие одновременно в обоих классах.
Ваш редактируемый `data/oof_v2` не является целью новых DVC pull: согласованные данные
обучения хранятся отдельно, чтобы не перезаписывать ручную работу.

Источник оригиналов: [папка организаторов](https://drive.google.com/drive/folders/1_UbQ7x4MK9fZjA-9DqJY_A8nedsg9KMy).
Публичность ссылки и доступ проверяющего нужно подтвердить перед сдачей.
Каталог содержит только `fullname` и `code`; цен, скидок и дат в нём нет.
Данные и веса не хранятся в Git; в Git находятся только DVC-указатели.

## Моделирование

### Бейзлайн

[YOLO](https://docs.ultralytics.com/) → классификация читаемости
[EfficientNet/timm](https://huggingface.co/docs/timm/) → [EasyOCR](https://www.jaided.ai/easyocr/).
Для обучения EfficientNet используется [Lightning](https://lightning.ai/docs/pytorch/stable/),
для YOLO — тренировочный framework Ultralytics.

### Основная модель

YOLO → BoT-SORT → фильтры размера/контраста/резкости → лучшие кропы сегмента →
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) +
[Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) → QR/EAN → локальный каталог → CSV.
Qwen является VLM, а не отдельным от VLM этапом. Ревизия Qwen зафиксирована в конфигурации.
Каталог применяется только при однозначном совпадении кода; нечёткие варианты остаются
предложениями. Основная модель отличается от идеи «восстановить большинство полей из БД»:
имеющийся каталог не позволяет этого сделать.

EfficientNet подключается параметром `efficient_weights`, но по умолчанию выключен:
низкая OOF-полнота делает его строгую фильтрацию нежелательной.

## Внедрение

Полный видеопайплайн реализован как локальная batch-команда. Для обученного
EfficientNet реализованы ONNX с проверкой численной эквивалентности и HTTP-сервис
MLflow Serving. Это сервис классификации кропов, **не сервис всего видеопайплайна**.
TensorRT CLI предусмотрен, но сборка engine требует `trtexec`; Triton не развёрнут.

## Setup

Проверяемая платформа: Linux x86-64, Python **3.12**, NVIDIA/CUDA **12.8** для GPU.
Torch 2.11.0+cu128 и torchvision 0.26.0+cu128 закреплены через официальный wheel index.
CPU-инференс также поддерживается; полноценная установка этого lock на macOS не заявляется.

После клонирования перейдите в корень репозитория:

```bash
python -m pip install poetry==2.2.1
poetry env use python3.12
poetry install --with data,train,ocr,vlm,paddle,export,serve
poetry run pre-commit install
poetry run pre-commit run --all-files
poetry run python -m unittest discover -s tests
```

Для только CPU-инференса без обучения и VLM:

```bash
poetry install --with data,ocr,paddle
```

Первая загрузка OCR/VLM-весов требует сети; после загрузки изображения обрабатываются
локально. Обучение и инференс разнесены, Lightning/MLflow не импортируются командой
инференса и входят в отдельные группы зависимостей. YAML-конфигурации включены в wheel.

### Доступ к DVC

Подготовлен локальный архив **`coursework_artifacts.tar.gz`** с DVC object store.
В рабочей папке автора он находится в `../server_results`; преподавателю нужен отдельный
доступный URL этого архива или общий DVC remote. **Без передачи хранилища одно клонирование
GitHub не восстановит подготовленные данные и веса.**

После получения распакуйте архив в удобное место и укажите абсолютный путь к `dvc-store`:

```bash
tar -xzf coursework_artifacts.tar.gz
poetry run dvc remote modify --local artifacts url /absolute/path/to/dvc-store
poetry run price-scanner download
```

Авторизация/локальные пути — только в `.dvc/config.local`, который игнорируется Git.
`train`, `infer`, `export` и `serve-package` по умолчанию вызывают настоящий DVC pull
и завершаются ошибкой при недоступном хранилище. Для уже подготовленных внешних файлов
можно явно указать `pull_dvc=false`.

Функция `download_data()` и CLI для публичных исходников:

```bash
poetry run price-scanner download source=public output=data/raw_public
```

Она не подменяет ручную разметку или обученные веса скачанными оригиналами.

## Train

По умолчанию MLflow ожидается на **http://127.0.0.1:8080**. Для своего тестового сервера:

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db
```

Проверка сквозного запуска на реальных данных (не оценка качества):

```bash
poetry run price-scanner train smoke=true 'folds=[0]' oof.accelerator=cpu oof.workers=0 output=runs/smoke
```

Полные запуски:

```bash
poetry run price-scanner train output=runs/efficient_oof
poetry run price-scanner train oof=yolo dataset=data/yolo_original_4fold output=runs/yolo_oof
```

Параметры — в `configs/oof`, `configs/train_oof.yaml`; `CUDA_VISIBLE_DEVICES=0` выбирает
разрешённую GPU. MLflow записывает loss, метрики, гиперпараметры, git commit и dirty-флаг.
Итоговый run содержит графики precision/recall/F1/accuracy/loss. Локально сохраняются
checkpoint каждого fold, OOF-предсказания, история и конфигурация.

Подготовка/изменение датасетов: [dataset_preparation](docs/dataset_preparation.md),
[efficient_review](docs/efficient_review.md). После новой разметки создавайте новый
согласованный manifest/snapshot; перенос JPEG между папками сам по себе manifest не обновляет.

## Production preparation

Экспортируется **реально обученный checkpoint**, без молчаливой подстановки стандартной модели:

```bash
poetry run price-scanner export output=artifacts/efficient_onnx
poetry run price-scanner serve-package output=artifacts/efficient_mlflow
```

ONNX: RGB/ImageNet normalization, float32 NCHW, размер 320, динамический batch;
выход — logits, вероятность получается sigmoid. Экспорт проверяет ONNX checker и
совпадение с PyTorch на batch 1 и 2. Пример входа версионируется DVC.
Поставка: ONNX/MLflow bundle, описание препроцессинга, конфигурация, отчёт проверки,
Poetry lock и DVC-артефакты. VLM сохраняется в Transformers-формате, не ONNX.

Если установлен TensorRT `trtexec`:

```bash
poetry run price-scanner export format=tensorrt output=artifacts/efficient_trt
```

Отсутствие TensorRT выдаёт явную ошибку, а не ложное сообщение об успешном engine.

## Infer

Новая папка результата обязательна, существующие результаты не перезаписываются.
Короткий DVC-пример, CPU:

```bash
poetry run price-scanner infer detector_device=cpu max_frames=7 top_k=1 output=data/infer_example
```

Полное видео с Qwen, после задания каталога исходных MP4:

```bash
poetry run price-scanner infer videos=/absolute/path/to/videos detector_device=cuda top_k=2 vlm_fallback=true vlm_limit=1000 vlm_name_override=true output=data/video_result
poetry run python -m scripts.review_video_csv data/video_result
poetry run python -m lenta_price_tag_scanner.field_extraction data/video_result/submission.csv --output=data/video_result/coverage_recomputed.json
```

Ограничения заполненности, правила объединения кадров и описание артефактов:
[полный пайплайн](docs/video_to_csv.md).

HTTP-сервис классификатора кропа:

```bash
poetry run mlflow models serve -m artifacts/efficient_mlflow --env-manager local --host 127.0.0.1 --port 18081
# В другом терминале:
poetry run python -m scripts.serve_smoke
```

POST `/invocations`: MLflow `dataframe_split` с одной колонкой `image_base64`.
Ответ: `predictions`, содержащий `probability` для каждого кропа. Готовый JSON-запрос
создаётся при `serve-package`; можно отправить его через curl. Сервис не читает
произвольные пути клиента на сервере.

## Проверка перед сдачей

[Навигация по документации](docs/README.md) · [DVC и резервная копия](docs/dvc_handoff.md).

[Матрица требований и оставшиеся действия](docs/compliance.md). CI проверяет хуки,
unit-тесты и конфигурации, но не подменяет полное обучение на доступном DVC-наборе.
Исторические отчёты описывают состояние на момент экспериментов; актуальные команды — здесь.
