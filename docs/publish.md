# Отправка актуальных изменений на GitHub

Основная ветка проекта — `main`. Данные и модели передаются в MinIO,
изменения кода, документации и DVC-указателей публикуются в GitHub.
Для новых коммитов используется обычная отправка без переписывания истории.

```bash
git status --short
git switch main
git fetch origin
git log --oneline --left-right HEAD...origin/main
git merge --ff-only origin/main
# При обновлении данных сначала передать объекты в MinIO:
dvc push -r minio
git push origin main
```

Если fast-forward невозможен, сначала объедините изменения сокомандника и
разрешите конфликты, затем повторите проверки. Команды старой очистки истории
с force/lease для обычной публикации больше не нужны.

Перед отправкой: `pre-commit run --all-files`, unit-тесты и проверка индекса Git.
В Git отправляются код, документация, графики и DVC-указатели. Данные, веса,
MLflow DB и `.dvc/config.local` остаются вне Git. Доступ к артефактам передаётся
через [DVC remote](dvc_handoff.md).
