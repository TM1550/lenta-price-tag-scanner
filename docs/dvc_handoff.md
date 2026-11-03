# Временное DVC-хранилище и перенос с сервера

03.11.2026 контрольное восстановление всех пяти указателей в новую no-SCM
папку прошло: 2687 DVC-объектов получено, статус данных — up to date.

Локальный remote `laptop-temp`:
`C:/Users/tsire/Downloads/MLOPS_codex/server_results/dvc-store`.
Он выбран по умолчанию через `.dvc/config.local`, который не должен попадать
в Git. Ранее существовавшие настройки сохранены; резервная копия приватного
конфига находится в игнорируемом `.dvc/tmp/config.local.before_laptop_20261103`.
Не публикуйте её: там могут быть старые учётные данные.

Это временное хранилище на том же ноутбуке, а не облачная резервная копия.
GitHub содержит только указатели `.dvc`; `git push` не отправляет сами данные.

Из корня репозитория, PowerShell:

```powershell
$env:DVC_NO_ANALYTICS = '1'
$env:DVC_SITE_CACHE_DIR = 'C:/Users/tsire/Downloads/MLOPS_codex/server_results/dvc-site-cache'
$dvc = '../.venv-pseudolabel/Scripts/dvc.exe'
& $dvc pull -r laptop-temp data/efficient_user_20261102.dvc data/yolo_original_4fold.dvc data/inference_assets.dvc data/pseudo_teacher_v1.dvc data/server_backup_20261103.tar.gz.dvc
```

При переходе на remote сокомандника не удаляйте локальное хранилище, пока
не выполнены push и восстановление из новой чистой копии:

```powershell
# Сначала настроить remote team с адресом сокомандника.
& $dvc remote add team '<адрес хранилища>'
# Пароли/токены задавать только локально, по инструкции для типа хранилища.
& $dvc push -r team data/efficient_user_20261102.dvc data/yolo_original_4fold.dvc data/inference_assets.dvc data/pseudo_teacher_v1.dvc data/server_backup_20261103.tar.gz.dvc
& $dvc config --local core.remote team
```

Для публичного default сначала установить `dvc config core.remote team`, затем
убрать перекрытие `dvc config --local --unset core.remote`. Команда `push -r team`
явно задаёт назначение независимо от текущего default.

Финальный архив сервера хранится в `../server_results/server_project_final_20261103.tar.gz`.
Внутри TRANSFER_INVENTORY.json: SHA256 и размер каждого включённого файла,
исключённые каталоги и ссылки. Исключены основные окружения, скачиваемые кеши, временные
файлы, дубли delivery, Git-история, приватные DVC-конфиги, `.env`, ключи и файлы
учётных данных. Они не нужны для сохранения обученных весов и результатов.
Обученные модели, исходники, данные, outputs, логи и MLflow-результаты включены.
Архив не предназначен для публикации как общедоступный материал: права на
распространение исходных видео/каталога остаются ответственностью владельца.

Подтверждение полного переноса: `../server_results/final_transfer_verification.json`.
Подтверждение восстановления DVC: `../server_results/dvc_restore_verified.txt`.
Пока оба файла не созданы успешными проверками, удалять серверную папку рано.
Серверные данные в ходе настройки не удаляются.

Локальная Git-история очищена; [протокол и порядок публикации](git_history_cleanup.md).
Ранее раскрытый пароль всё ещё нужно заменить: переписывание истории не отзывает его.
