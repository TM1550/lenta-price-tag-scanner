# Хранение данных в DVC и MinIO

Для воспроизведения проекта используется публичный remote `release`
(`s3://lenta-dvc-release`, HTTPS endpoint `https://31.31.207.108`).
MinIO разрешает анонимные чтение и просмотр объектов этого бакета.
Рабочее хранилище сопровождающих проекта доступно отдельно как remote `minio`.

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

## Скачивание

Из корня клонированного репозитория:

```bash
poetry install --with data
poetry run dvc pull
```

Для получения данных без установки ML-зависимостей используйте отдельное
Python-окружение и команду `python -m pip install "dvc[s3]==3.67.1"`.
Затем выполните `dvc pull -r release`. Указать только обучающие наборы можно так:

```bash
dvc pull -r release data/yolo_roboflow_5fold_v1.dvc data/efficient_user_20261102.dvc
```

В общем конфиге включены проверка TLS и `allow_anonymous_login`.
При наличии локального переопределения remote укажите `-r release` явно.

## Обновление артефактов

Сопровождающий сохраняет новую версию командой `dvc add` и передаёт её
в рабочий бакет командой `dvc push -r minio` с персональными ключами.
Перед публикацией новых указателей в GitHub необходимо перенести объекты,
на которые они ссылаются, в бакет выпуска с административным доступом
и проверить скачивание через `release` из чистого кэша.

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
Для публичного скачивания выбран remote `release`. Удаление локальных данных не является частью миграции.
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
