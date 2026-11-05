# Хранение данных в DVC и MinIO

Основное хранилище проекта — закрытый MinIO на VPS `31.31.207.108`.
В Git хранятся DVC-указатели и конфигурация remote; данные и веса передаются через S3 API.
Remote `minio` выбран по умолчанию:

```ini
['remote "minio"']
    url = s3://lenta-dvc
    endpointurl = http://127.0.0.1:19000
    region = us-east-1
    use_ssl = false
```

HTTP используется только внутри SSH-туннеля. На VPS API и консоль привязаны к
`127.0.0.1:9000` и `127.0.0.1:9001`; соединение между ноутбуком и VPS шифруется SSH.

## Состав артефактов

| DVC-указатель | Файлов | МиБ | Содержимое |
| --- | ---: | ---: | --- |
| `data/efficient_user_20261102.dvc` | 2011 | 38.8 | Кропы читаемости, метки и manifest для EfficientNet |
| `data/inference_assets.dvc` | 6 | 88.7 | YOLO, EfficientNet, каталог и короткий пример видео |
| `data/pseudo_teacher_v1.dvc` | 159 | 167.3 | Псевдоразметка старой YOLO и метаданные отбора |
| `data/runtime_assets.dvc` | 636 | 496.8 | Полные видео, эталонные CSV, OCR-набор и локальные PaddleOCR-веса |
| `data/video_oof_20261104.dvc` | 1361 | 157.2 | Пять YOLO-моделей, итоговые CSV, отчёты, галереи и provenance OOF |
| `data/yolo_deployment_v1.dvc` | 51 | 28.8 | Артефакты экспорта YOLO для развёртывания |
| `data/yolo_original_4fold.dvc` | 508 | 254.3 | Исходная разметка YOLO и 4 folds по видео |
| `data/yolo_roboflow_5fold_v1.dvc` | 809 | 394.3 | Очищенный экспорт Roboflow/SAM3 и 5 folds по видео |

Суммарный логический объём — 1 704 991 484 байта (около 1,70 ГБ).
Общие объекты хранятся в DVC однократно. Полные веса Qwen в эти артефакты не входят;
VLM загружается отдельно по зафиксированной в конфигурации ревизии.

## Подключение

Для DVC нужны два независимых доступа: SSH к VPS и S3-ключи MinIO.
Администратор предоставляет SSH-пользователя и разрешает публичный ключ участника;
каждый участник использует собственный приватный ключ. Запись в `~/.ssh/config`:

```sshconfig
Host regru-vps
    HostName 31.31.207.108
    User <SSH_USER>
    Port 22
    IdentityFile ~/.ssh/<PRIVATE_KEY_FILE>
    IdentitiesOnly yes
```

Откройте туннель в отдельном терминале Windows Git Bash, PowerShell или Linux/WSL:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:19000:127.0.0.1:9000 -L 127.0.0.1:19001:127.0.0.1:9001 regru-vps
```

Терминал с туннелем должен оставаться открытым. Запускайте DVC в той же среде,
где открыт туннель: например, обе команды в WSL или обе на Windows.
API доступен по `http://127.0.0.1:19000`, консоль — по `http://127.0.0.1:19001`.
Проверка API: `curl --fail http://127.0.0.1:19000/minio/health/live`.

## Настройка DVC

В проектном Poetry-окружении используется группа `data`:

```bash
poetry install --with data
```

Установка зависимостей не скачивает датасеты. Если нужны только передача и
версионирование данных, используйте отдельное Python-окружение с DVC:

```bash
python -m pip install "dvc[s3]==3.67.1"
```

Это окружение не требует установки PyTorch/CUDA. Ниже команды приведены для
самостоятельного `dvc`; в Poetry-окружении используйте `poetry run dvc`.

Из корня репозитория задайте полученные S3-ключи:

```bash
dvc remote modify --local minio access_key_id '<ACCESS_KEY>'
dvc remote modify --local minio secret_access_key '<SECRET_KEY>'
dvc config --local core.remote minio
```

Параметры `--local` сохраняются в `.dvc/config.local`, который исключён из Git.
На MinIO подготовлены отдельные аккаунты: два с правами чтения/записи и один
только для чтения бакета `lenta-dvc`. Доступ читателя предназначен для проверки
и скачивания артефактов. Администраторские ключи для работы DVC не требуются.

## Загрузка и обновление данных

Все восемь артефактов:

```bash
dvc pull -r minio
```

Только новый обучающий датасет YOLO и EfficientNet:

```bash
dvc pull -r minio data/yolo_roboflow_5fold_v1.dvc data/efficient_user_20261102.dvc
```

Минимальные артефакты текущего CPU-инференса:

```bash
dvc pull -r minio data/inference_assets.dvc data/runtime_assets.dvc data/video_oof_20261104.dvc
```

Обновление отдельного датасета выполняется владельцем доступа на запись:

```bash
dvc add data/yolo_roboflow_5fold_v1
dvc push -r minio data/yolo_roboflow_5fold_v1.dvc
git add data/yolo_roboflow_5fold_v1.dvc
git commit -m "Update versioned YOLO dataset"
git push
```

Сначала завершается передача объектов в MinIO, затем публикуется DVC-указатель в Git.
`git push` сам по себе не передаёт файлы датасета. Рабочие папки ручной разметки
используются как источник подготовки снимков; цели DVC перечислены в таблице выше.

## Развёртывание и резервная копия

MinIO размещён в `/opt/lenta-minio`, постоянные объекты — в подкаталоге `data`.
Лицензия подключена к контейнеру только для чтения. Docker включён в автозапуск,
контейнер использует `restart: unless-stopped`. Команды администратора VPS:

```bash
cd /opt/lenta-minio
docker compose -f compose.yaml -f compose.licensed.yaml ps
docker compose -f compose.yaml -f compose.licensed.yaml up -d
```

Лицензия, `.env`, закрытые ключи и конфиги MinIO остаются вне Git.
Локальный object store автора `laptop-temp` сохранён как резервная копия;
MinIO выбран основным remote. Удаление локальных данных не является частью миграции.
Локальный резервный архив проекта сохранён на ноутбуке и исключён из целей DVC;
для воспроизведения используются восемь артефактов из таблицы выше.
