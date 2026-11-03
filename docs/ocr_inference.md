# Инференс кропов для OCR и VLM

Из корня репозитория:

```powershell
poetry run python -m lenta_price_tag_scanner.ocr_crops
```

Локальные пути по умолчанию используют существующие ../data и ../lenta-price-tag-scanner/base_pipeline. На сервере передавать videos=../data yolo_weights=../models/price_tag.pt efficient_weights=../models/efficient.pth output=../outputs/ocr_run device=cuda. Изменения задаются в configs/ocr_crops.yaml или Hydra overrides. Выходной каталог должен быть новым.

Результат: ocr_inputs.jsonl — выбранные входы для текстовой модели, image — путь к JPEG относительно каталога результата, filename — видео, frame_timestamp — миллисекунды, bbox_xyxy — координаты в исходном кадре. Изображения сохраняются в исходном разрешении кропа, без ресайза для OCR. EfficientNet использует уменьшенную копию только для классификации. text и product_id остаются null: этот шаг ничего не распознаёт и не отправляет наружу.

Поддерживаются legacy efficient.pth (BGR, 320×320, 1-sigmoid) и новые model.pt из train_oof (RGB, размер из checkpoint, sigmoid класса 1). Новый формат определяется по state_dict и проверяется по metadata. В yolo_weights можно передать last.pt обученного детектора. Для отключения классификатора — efficient_weights=null; тогда в OCR попадают все кропы после геометрического отбора. Для короткой проверки — pattern=25_12-20.mp4 max_frames=150.

Порог отбора high=0.6 соответствует старому pipeline и не является откалиброванным значением. Для downstream использовать только ocr_inputs.jsonl; полный manifest.jsonl и остальные категории сохранены для диагностики фильтра. Пропущенные EfficientNet ценники могут снижать recall OCR.

Точного повторения пикселей внутри видео не сохраняем, но трекинга нет: несколько кропов могут относиться к одному ценнику. Это входы для OCR/VLM, а не финальный CSV с уникальными ценниками. Нельзя считать их количество количеством товаров.
