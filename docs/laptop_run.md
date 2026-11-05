# Запуск на ноутбуке без сервера

Проверено 03.11.2026 на Windows, AMD integrated graphics, 16 ГБ RAM.
Рабочий режим этого ноутбука — YOLO + трекинг + PaddleOCR CPU + QR/EAN + каталог → CSV.
Qwen в этот CPU-прогон не входил: NVIDIA/CUDA нет, а текущая реализация
Qwen4B CPU использует float32 и не подходит для доступных 16 ГБ RAM.
Это ограничение аппаратуры/реализации, не зависимость от удалённого сервера.
По решению пользователя загрузка Qwen отложена. Полных весов Qwen в DVC пока
нет, поэтому утверждение «все модели доступны офлайн» относится только к
проверенному CPU-пайплайну. Незавершённая локальная загрузка Qwen удалена
04.11.2026 при [очистке служебных дублей](workspace_cleanup.md).
Она не содержала готовых частей модели и не использовалась CPU-пайплайном.

## Изменение конфигурации 04.11.2026

05.11 после переноса эвристик в Hydra повторён короткий CPU smoke: первые 9 кадров
43_15 → 11 строк CSV за 41.20 с; Git commit `57eabac`, dirty=false. Это проверка
исполнения с прежними порогами, без переобучения основных моделей и без GPU.

Новая конфигурация проверена на первых 120 кадрах 43_15: 23 строки, 29 колонок,
254.37 секунды; EfficientNet оценил 415 кропов, выбран YOLO fold 3. Coverage
48/529 = 9.0737% — проверка выполнения, не OOF-точность.

Текущий default использует новые пять YOLO по `model_by_video.json` и EfficientNet
в режиме ранжирования. Нужен также `data/video_oof_20261104.dvc`.
Приведённый ниже старый результат не является измерением качества новой конфигурации.
Подробности и команды: [YOLO deployment](yolo_deployment.md).

## Исторический проверенный результат

Полное видео `43_15.mp4`: 45 строк, 29 столбцов, 434.51 секунды.
Заполнено 95/1035 содержательных полей (9.1787%). Это coverage, не точность.
Результаты: `data/laptop_cpu_verified/submission.csv`, `review.html`, кропы и детали.
Входы взяты из отдельной no-SCM папки, восстановленной через DVC.
В тестовом процессе исходящие socket-соединения были заблокированы;
три попытки соединений отклонены, пайплайн успешно завершился без сетевого ввода.

## Данные и модели

- `data/inference_assets.dvc`: исходная YOLO, обученный EfficientNet, каталог, короткое видео.
- `data/runtime_assets.dvc`: пять полных видео, их эталонные CSV, OCR-разметка,
  локальные веса PP-OCRv5 mobile detection и eslav recognition.
- `data/efficient_user_20261102.dvc`, `data/yolo_original_4fold.dvc`,
  `data/pseudo_teacher_v1.dvc`: обучающие снимки.
- `data/video_oof_20261104.dvc`: пять обученных YOLO-folds и результаты полной OOF-оценки.
- `data/yolo_deployment_v1.dvc`: артефакты экспорта YOLO.

Основной DVC remote — `minio`. Для первичной загрузки нужен доступ к MinIO через
SSH-туннель; [настройка подключения](dvc_handoff.md). После получения артефактов
CPU-инференс выполняется локально. `laptop-temp` сохранён как резервное хранилище автора.

## Команды PowerShell

Из корня `Design_ML_system`:

```powershell
$python = '../.venv-pseudolabel/Scripts/python.exe'
$env:DVC_NO_ANALYTICS = '1'
$env:DVC_SITE_CACHE_DIR = "$PWD/../server_results/dvc-site-cache"
$env:YOLO_CONFIG_DIR = "$PWD/data/laptop_runtime/ultralytics"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = 'True'

& $python -m dvc pull -r minio data/inference_assets.dvc data/runtime_assets.dvc data/video_oof_20261104.dvc
& $python -m lenta_price_tag_scanner.commands infer-cpu output=data/my_laptop_run
& $python -m scripts.review_video_csv data/my_laptop_run
```

Выбирайте новую выходную папку на каждый запуск. По умолчанию профиль проверяет
полное `43_15.mp4`; для всех видео добавьте `pattern=*.mp4`.
Основной `infer` тоже использует локальные Paddle-веса после DVC pull.

Окружение `.venv-pseudolabel` установлено отдельно от серверного Poetry-окружения:
CPU Torch 2.15.0+cpu, PaddlePaddle 3.2.2, PaddleOCR 3.3.2, PaddleX 3.3.13.
Точные установленные версии сохранены в `../server_results/laptop_cpu_environment.txt`.
Не выдаём этот прогон за проверку установки из CUDA Poetry lock на Windows.
Для новой машины сначала нужны зависимости и доступ к DVC-хранилищу.
