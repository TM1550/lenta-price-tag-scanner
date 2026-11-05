# Хранение данных в DVC и MinIO

Основное хранилище проекта — закрытый MinIO на VPS `31.31.207.108`.
В Git хранятся DVC-указатели и конфигурация remote; данные и веса передаются через S3 API.
Remote `minio` выбран по умолчанию:

```ini
['remote "minio"']
    url = s3://lenta-dvc
    endpointurl = https://31.31.207.108
    region = us-east-1
    use_ssl = true
    ssl_verify = true
```

DVC обращается к S3 API по HTTPS и проверяет TLS-сертификат сервера.
Nginx публикует API бакета `lenta-dvc` на порту 443; бакет остаётся закрытым.
MinIO API `127.0.0.1:9000` и консоль `127.0.0.1:9001` доступны локально на VPS.
Административные и внутренние маршруты MinIO не публикуются через HTTPS proxy.

## Состав артефактов

| DVC-указатель                                | Файлов |   МиБ | Содержимое                                                        |
| -------------------------------------------- | -----: | ----: | ----------------------------------------------------------------- |
| `data/efficient_user_20261102.dvc`           |   2011 |  38.8 | Кропы читаемости, метки и manifest для EfficientNet               |
| `data/inference_assets.dvc`                  |      6 |  88.7 | YOLO, EfficientNet, каталог и короткий пример видео               |
| `data/runtime_assets.dvc`                    |    636 | 496.8 | Полные видео, эталонные CSV, OCR-набор и локальные PaddleOCR-веса |
| `data/video_oof_20261104.dvc`                |   1361 | 157.2 | Пять YOLO-моделей, итоговые CSV, отчёты, галереи и provenance OOF |
| `data/yolo_deployment_v1.dvc`                |     51 |  28.8 | Артефакты экспорта YOLO для развёртывания                         |
| `data/yolo_roboflow_5fold_v1.dvc`            |    809 | 394.3 | Очищенный экспорт Roboflow/SAM3 и 5 folds по видео                |
| `data/video_oof_efficient_rank_20261105.dvc` |   1295 | 139.7 | Пять EfficientNet-folds, полный OOF с rank, CSV и отчёты          |

Суммарный логический объём — 1 409 412 221 байт (около 1,41 ГБ).
Общие объекты хранятся в DVC однократно. Полные веса Qwen в эти артефакты не входят;
VLM загружается отдельно по зафиксированной в конфигурации ревизии.

## Подключение

Пользователь получает `access_key_id` и `secret_access_key` MinIO.
Для скачивания достаточно отдельного аккаунта только с правами чтения бакета.
Для проверки проекта используется учётная запись MinIO с правом чтения.

```bash
curl --fail https://31.31.207.108/minio/health/live
```

Ответ HTTP 200 означает, что сервис доступен. Доступ к объектам дополнительно
проверяется S3-ключами. TLS использует доверенный сертификат Let’s Encrypt для IP;
сертификат автоматически продлевается. Проверку TLS отключать не требуется.

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

Все семь артефактов:

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
docker compose --env-file .env --env-file https.env -f compose.yaml -f compose.licensed.yaml -f compose.https.yaml ps
docker compose --env-file .env --env-file https.env -f compose.yaml -f compose.licensed.yaml -f compose.https.yaml up -d
```

Лицензия, `.env`, закрытые ключи и конфиги MinIO остаются вне Git.
Локальный object store автора `laptop-temp` сохранён как резервная копия;
MinIO выбран основным remote. Удаление локальных данных не является частью миграции.
Локальный резервный архив проекта сохранён на ноутбуке и исключён из целей DVC;
для воспроизведения используются семь артефактов из таблицы выше.

## HTTPS и обслуживание сертификата

Конфигурация Nginx находится в `/opt/lenta-minio/https/nginx/default.conf`,
сертификат и ключ — в закрытом каталоге `/opt/lenta-minio/tls/letsencrypt`.
`lenta-minio-certbot.timer` проверяет необходимость продления дважды в сутки.
`renew_tls.sh` запускает закреплённый образ Certbot и после успешного завершения
проверяет и перезагружает Nginx. Порт 80 обслуживает ACME HTTP-01 challenge;
остальные HTTP-запросы перенаправляются на HTTPS.

```bash
systemctl status lenta-minio-certbot.timer
systemctl start lenta-minio-certbot.service
# Проверка продления через staging без замены рабочего сертификата:
bash /opt/lenta-minio/https/renew_tls.sh --dry-run
```

При ошибке подключения проверьте, что `.dvc/config.local` не
перекрывает общий `endpointurl` значением `http://127.0.0.1:19000`.
При необходимости задайте актуальный адрес локально:

```bash
dvc remote modify --local minio endpointurl https://31.31.207.108
dvc remote modify --local minio use_ssl true
dvc remote modify --local minio ssl_verify true
```

При использовании HTTP-прокси, который не пропускает HTTPS к IP-адресам,
добавьте `31.31.207.108` в `NO_PROXY` для процесса DVC. Храните ключи MinIO в локальной конфигурации DVC.
