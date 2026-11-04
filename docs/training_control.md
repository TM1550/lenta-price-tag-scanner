# Контроль обучения и MLflow

Это короткая проверка инфраструктуры на реальных DVC-данных: один fold, по восемь
train/validation изображений и две эпохи для YOLO и EfficientNet. Веса создаются
заново, CPU, без Qwen. Полученные метрики не заменяют полную OOF-оценку 4.1610%.

## Запуск

Установите зависимости `poetry install --with data,train`. Выполняйте команды из
корня Git checkout с чистым `git status`: все исходники и конфиги должны быть
закоммичены до запуска. Данные и результаты находятся под игнорируемым `data/`.

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///data/control_runs/mlflow.db --artifacts-destination data/control_runs/mlflow_artifacts
```

В другом терминале:

```bash
poetry run python -m lenta_price_tag_scanner.train_oof --config-name smoke_efficient
poetry run python -m lenta_price_tag_scanner.train_oof --config-name smoke_yolo
```

Профили наследуют основные конфиги и используют встроенный DVC pull. Если данные
уже восстановлены локально, допустим явный `pull_dvc=false`. Для повторного запуска
выберите новые `output=...`, так как старые результаты не перезаписываются.

## Что записывается

- Git commit и dirty-флаг, fold, seed, SHA256 manifest, исходные гиперпараметры,
  фактические две эпохи, размеры проверочной выборки и отметка `smoke=true`.
- EfficientNet: train loss по эпохам, финальные loss/precision/recall/F1/accuracy.
- YOLO: box/cls/DFL train loss по эпохам и финальные метрики детектора.
- YAML-конфигурация, предсказания, графики истории обучения в fold-run;
  графики итоговых метрик в отдельном summary-run. Всего четыре завершённых runs.

Внешний fold оценивается после фиксированного числа эпох. Выбор лучшего checkpoint
по нему не выполняется. YOLO использует `last.pt`. Встроенная MLflow-интеграция
Ultralytics отключается в callback данного запуска, чтобы логирование проекта
оставалось в собственном активном run.

## Подтверждение

Результаты выполненной проверки и минимум три графика сохраняются в
`plots/training_control_20261105/`. Полная база MLflow, артефакты и проверочные веса
хранятся локально в `data/control_runs_20261105/`, вне Git. Для просмотра этого запуска:

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///data/control_runs_20261105/mlflow.db --artifacts-destination data/control_runs_20261105/mlflow_artifacts
```

Старые серверные runs с `git_commit=uncommitted` остаются историческими. Их код
подтверждён SHA256-описью. Новый контрольный запуск демонстрирует корректное
версионирование и логирование обучения, не приписывает старые результаты новому коду.
