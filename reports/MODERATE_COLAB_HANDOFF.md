# Moderate v3: handoff в Colab

Локальный архив:

`semester4_coursework/dataset_local/ru_dataset/05_colab_package/colab_lipreading_moderate_v3_prep.zip`

Размер около 120 MB. Внутри находятся 818 raw word clips, speaker split
`train/val/test`, `vocab.txt`, `README.md` и три уже существующих инструмента:
`tools/create_landmark_mouth_crops.py`, `tools/create_ml_splits.py` и
`tools/package_ml_dataset.py`.

## Шаги в Colab

1. Загрузить архив в Google Drive, например в
   `MyDrive/lipreading_sem4/colab_lipreading_moderate_v3_prep.zip`.
2. Создать новый Colab notebook и включить GPU. Для MediaPipe GPU не обязателен,
   но GPU пригодится позже для обучения.
3. Выполнить установку зависимостей:

```python
%pip -q install mediapipe opencv-python-headless
```

4. Выполнить подготовку архива:

```python
from pathlib import Path
import shutil
import subprocess
import zipfile
from google.colab import drive

drive.mount('/content/drive')

archive = Path('/content/drive/MyDrive/lipreading_sem4/colab_lipreading_moderate_v3_prep.zip')
extract_dir = Path('/content/lipreading_moderate_v3')
if not extract_dir.exists():
    with zipfile.ZipFile(archive) as z:
        z.extractall(extract_dir)

dataset_root = extract_dir / 'ru_dataset'
split_dir = dataset_root / '03_splits/ml_splits_moderate_v3/speaker_top10'
tool = extract_dir / 'tools/create_landmark_mouth_crops.py'
out_dir = dataset_root / '02_model_inputs/mouth_crops_landmark_moderate_v3'
manifest = dataset_root / '04_quality_reports/landmark_moderate_v3_labels.csv'
failed = dataset_root / '04_quality_reports/landmark_moderate_v3_failed.csv'
contact = dataset_root / '04_quality_reports/landmark_moderate_v3_contact_sheet.jpg'

command = [
    'python', str(tool),
    '--dataset-root', str(dataset_root),
    '--split-csv', str(split_dir / 'train.csv'),
    '--split-csv', str(split_dir / 'val.csv'),
    '--split-csv', str(split_dir / 'test.csv'),
    '--out-dir', str(out_dir),
    '--manifest', str(manifest),
    '--failed-manifest', str(failed),
    '--contact-sheet', str(contact),
    '--size', '96',
    '--min-detection-ratio', '0.70',
]
subprocess.run(command, check=True)
```

## Ожидаемый результат

После выполнения нужно проверить:

```python
import pandas as pd

landmark_df = pd.read_csv(manifest)
failed_df = pd.read_csv(failed) if failed.exists() else pd.DataFrame()
print('success:', len(landmark_df))
print('failed:', len(failed_df))
print(landmark_df['landmark_status'].value_counts())
print(landmark_df['landmark_detection_ratio'].astype(float).describe())
```

Для продолжения желательно получить минимум 90% успешных crop-ов, все 10 слов
и отсутствие систематического провала одного спикера. `failed.csv` сохраняется
для анализа, но не означает автоматическое удаление исходного клипа.

5. Построить новый ML split уже по manifest landmark crops:

```python
split_out = dataset_root / '03_splits/ml_splits_landmark_moderate_v3'
subprocess.run([
    'python', str(extract_dir / 'tools/create_ml_splits.py'),
    '--labels', str(manifest),
    '--out-dir', str(split_out),
    '--top-words', '10',
    '--seed', '42',
    '--train-speakers', 'spk16,spk17',
    '--val-speakers', 'spk18',
    '--test-speakers', 'spk19',
    '--only-speaker-top',
], check=True)
```

6. Упаковать mouth crops для обучения:

```python
mouth_archive = Path('/content/colab_lipreading_moderate_v3_landmark.zip')
subprocess.run([
    'python', str(extract_dir / 'tools/package_ml_dataset.py'),
    '--dataset-root', str(dataset_root),
    '--split-dir', str(split_out / 'speaker_top10'),
    '--output', str(mouth_archive),
    '--archive-root', 'ru_dataset',
], check=True)
shutil.copy2(
    mouth_archive,
    Path('/content/drive/MyDrive/lipreading_sem4') / mouth_archive.name,
)
```

После этого запускается тот же controlled MS-TCN/BiGRU protocol на новом
архиве. Нельзя сравнивать moderate и padded по разным split или разному числу
эпох.
