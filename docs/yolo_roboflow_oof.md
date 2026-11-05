# YOLO: новый очищенный набор, 5 folds по видео

Датасет `data/yolo_roboflow_5fold_v1` собран из очищенного экспорта SAM3/Roboflow.
Он содержит **391 кадр и 2593 bbox**, один класс `price_tag`. Два кадра с пустой,
не подтверждённой как фон разметкой исключены; их список — `excluded.json`.
Все положительные кадры и все сохранённые bbox вошли в набор. Семь кадров,
не вошедших в исходный экспорт Roboflow, автоматически не восстанавливаются.

| Fold | Видео validation | Train кадров | Val кадров | Val bbox |
| ---- | ---------------- | -----------: | ---------: | -------: |
| 0    | 25_12-20.mp4     |          308 |         83 |      466 |
| 1    | 25_2-10.mp4      |          330 |         61 |      613 |
| 2    | 26_12-20.mp4     |          218 |        173 |      917 |
| 3    | 43_15.mp4        |          361 |         30 |      334 |
| 4    | 49_5.mp4         |          347 |         44 |      263 |

Каждый кадр находится в validation ровно один раз. Видео и точные дубликаты
изображений между train и val не пересекаются. Идентификатор видео не доказывает
независимость полок: если несколько видео показывают те же ценники, нужна более
крупная группа съёмки и меньше folds. Результаты измеряют качество относительно
подготовленной SAM3-разметки; геометрическая очистка не заменяет ручной проверки.

## Структура и воспроизведение

```text
data/yolo_roboflow_5fold_v1/
  yolo/images/<video>/...jpg
  yolo/labels/<video>/...txt
  yolo/manifest.json
  yolo/folds.json
  folds/fold_0/{data.yaml,train.txt,val.txt}
  ... fold_1 ... fold_4
  excluded.json
  validation.json
  ultralytics_list_check.json
```

Изображения и метки хранятся в одном экземпляре. В списках folds используются
пути относительно файлов списков; YAML разрешается относительно собственного
каталога. В manifest пути относительны корню набора. При обучении `train_oof`
создаёт абсолютные списки и `data.yaml` внутри каталога конкретного запуска.

Все команды ниже выполняются **из корня репозитория**.

В уже настроенном окружении ноутбука:

```powershell
$py = '../.venv-pseudolabel/Scripts/python.exe'
$env:DVC_NO_ANALYTICS = '1'
$env:DVC_SITE_CACHE_DIR = "$PWD/../server_results/dvc-site-cache"
$env:YOLO_CONFIG_DIR = "$PWD/data/laptop_runtime/ultralytics"
& $py -m dvc pull -r minio data/yolo_roboflow_5fold_v1.dvc
& $py -m lenta_price_tag_scanner.commands train-yolo --show-config
```

Согласованный DVC-снимок содержит 809 файлов и доступен в MinIO.
Перед загрузкой откройте SSH-туннель и задайте S3-ключи по
[инструкции DVC](dvc_handoff.md). GitHub содержит указатель, сами данные передаются через DVC.

Сборка заново из очищенного набора, с новым выходным каталогом:

```powershell
& $py -m lenta_price_tag_scanner.yolo_folds output=data/yolo_roboflow_5fold_rebuild
```

Повторная сборка не перезаписывает существующие данные. После изменения меток
нужно повторить очистку и сборку, обновить DVC-снимок и использовать новое имя
эксперимента. Хеши защищают от незаметного изменения исходных меток.

## Окружение и MLflow

На GPU-компьютере установите проект с группами обучения и DVC:

```powershell
poetry install --with train,data
```

Lock-файл рассчитан на CUDA-сборку PyTorch. На текущем ноутбуке используется
проверенное отдельное CPU-окружение `.venv-pseudolabel`; MLflow в нём пока не
установлен. Для выполнения команд обучения в этом окружении добавьте:

```powershell
& $py -m pip install 'mlflow==3.17.2'
```

Запустите MLflow в отдельном терминале из корня репозитория (тем же Python):

```powershell
& $py -m mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///data/mlflow_yolo.db
```

При использовании Poetry замените `& $py` в командах на `poetry run python`.
Сервис и обучение должны использовать один доступный `mlflow_uri`.

## Запуск

Проверка одной эпохи на CPU: до 8 train и 8 val кадров, только fold 0.
Это диагностический запуск, его метрики не подходят для отчёта о качестве.

```powershell
& $py -m lenta_price_tag_scanner.commands train-yolo smoke=true 'folds=[0]' oof.device=cpu oof.workers=0 oof.batch=2 oof.size=320 oof.amp=false output=runs/yolo_roboflow_smoke_cpu
```

Перед полным GPU-запуском проверьте занятость карты:

```powershell
nvidia-smi
```

Полные 5 folds последовательно на доступной CUDA GPU 0:

```powershell
& $py -m lenta_price_tag_scanner.commands train-yolo oof.device=0 output=runs/yolo_roboflow_5fold_full
```

На текущем ноутбуке CUDA отсутствует. Для полного CPU-запуска:

```powershell
& $py -m lenta_price_tag_scanner.commands train-yolo oof.device=cpu oof.workers=0 oof.batch=2 oof.amp=false output=runs/yolo_roboflow_5fold_cpu
```

100 эпох × 5 folds на CPU потребуют существенно больше времени. Отдельный fold
запускается параметром `'folds=[2]'`; для каждого отдельного запуска используйте
свой `output`, например `output=runs/yolo_roboflow_fold_2`. Для уменьшения GPU-памяти
снижайте `oof.batch`, сохраняя одинаковые остальные настройки всех folds.

Основной профиль `configs/train_yolo_roboflow.yaml` наследует `train_oof.yaml`
и использует `configs/oof/yolo_roboflow.yaml`: YOLOv8n из архитектуры YAML,
100 эпох, imgsz 1280, batch 8, AdamW, lr 0.001, seed 42 + номер fold.
Это начальные настройки, качество этого нового обучения ещё не измерено.
Каждый fold создаёт новую модель с нуля. Старые веса детектора не используются.
Остановка по outer validation отключена, оценивается последняя модель `last.pt`.
Ultralytics может выполнить встроенную валидацию на последней эпохе даже при
`val=false`; для итоговых OOF-прогнозов всё равно явно загружается `last.pt`.
Выбирать эпохи или гиперпараметры по этим пяти outer folds нельзя; для этого
нужна отдельная внутренняя валидация.

## Результаты будущего обучения и выполненные проверки

В `runs/<имя>/fold_N/` будут `training/weights/last.pt`, история обучения,
`evaluation/` с метриками детектора и `oof_predictions.json` только для
отложенного видео. Прогнозы: `[x1, y1, x2, y2, confidence, class]` в пикселях
исходного изображения, минимальный confidence для сохранения — 0.001.
В корне запуска появятся `metrics.json`, полный `config.yaml`, графики по folds.
MLflow сохраняет параметры, хеш манифеста, commit/dirty, потери и итоговые метрики.
Это метрики детекции; правильность OCR/CSV оценивается отдельным пайплайном.

Выполнены проверки всех изображений и меток по SHA256, покрытия validation,
изоляции видео и отсутствия точных дубликатов. Все 10 списков train/val прочитаны
установленным Ultralytics, пути меток разрешаются корректно. Проверена композиция
Hydra-конфига и автоматические тесты сборки, переносимости и обнаружения изменений.
Обучение и запуск MLflow на этом этапе не выполнялись.
