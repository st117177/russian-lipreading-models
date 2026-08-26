# Статус работы для 4 семестра

## Что было в прошлой курсовой

В 3 семестре был собран небольшой демонстрационный датасет для чтения русских слов по губам. Основной пайплайн:

```text
video + transcript
-> audio + cleaned text
-> WebMAUS forced alignment
-> TextGrid
-> words_frames.txt
-> нарезка word-level clips
-> labels.csv / vocab.txt
-> validation
```

Основные скрипты лежат в `lip-read-coursework-materials/scripts`.

## Что мы сделали для 4 семестра

Мы расширили работу от демо-датасета к ML-эксперименту:

- скачали и подготовили дополнительные YouTube-видео;
- разбили длинные видео на chunks для WebMAUS;
- получили/использовали `TextGrid` и `words_frames.txt`;
- нарезали word-level клипы;
- исправили слишком короткую нарезку через padded-версию клипов;
- сделали quality check;
- сделали mouth crops;
- сделали speaker-based train/val/test split;
- подготовили архив для Google Colab;
- в Colab собрали первый PyTorch baseline.

## Актуальная версия датасета

Использовать нужно padded-версию:

- `ru_dataset/01_intermediate_clips/selected_clips_padded/`
- `ru_dataset/01_intermediate_clips/selected_labels_padded.csv`
- `ru_dataset/01_intermediate_clips/selected_labels_padded_clean.csv`
- `ru_dataset/02_model_inputs/mouth_crops_padded/`
- `ru_dataset/02_model_inputs/mouth_crops_padded_labels.csv`
- `ru_dataset/03_splits/ml_splits_mouth_crops_padded_clean_manual_no_spk09_10/speaker_top10/`

Старые папки без `_padded` оставлены только для сравнения: там клипы были нарезаны слишком коротко.

## Размер датасета

Сравнение с демо-датасетом 3 семестра:

- было в `lip-read-demo-dataset/labels.csv`: 182 клипа, 8 слов, 5 спикеров;
- стало в `ru_dataset/selected_labels_padded_clean.csv`: 1222 clean-клипа, 14 слов, 8 спикеров;
- общий рост clean-датасета: `182 -> 1222`, то есть примерно в 6.7 раза.

Padded full-frame clips:

- всего: 1280
- после quality filtering: 1222
- минимальная длительность после исправления: около 0.79 сек
- средняя длительность: около 1.07 сек

Mouth crops:

- 1222 клипа
- битых crop-видео после проверки: 0

Отдельный split без плохих `spk09` и `spk10`:

- train: 700 клипов, speakers `spk03`, `spk04`, `spk05`
- val: 105 клипов, speaker `spk06`
- test: 150 клипов, speakers `spk07`, `spk08`
- всего для baseline без `spk09`/`spk10`: 955 клипов, 10 слов, 6 нормальных спикеров;
- рост рабочего ML-набора относительно демо-датасета: `182 -> 955`, то есть примерно в 5.2 раза.

Целевой план на следующий этап:

- не скачивать видео хаотично, а добирать данные под баланс;
- заменить или исключить плохих `spk09` и `spk10`;
- довести рабочий датасет примерно до 1500-2500 clean-клипов;
- оставить 10-12 слов для первой устойчивой модели;
- собрать 8-10 нормальных talking-head спикеров;
- стремиться хотя бы к 80-100 clean-клипам на каждое слово.

Слова, которые особенно нужно добирать/балансировать:

- `сегодня`
- `делать`
- `нужно`
- `человек`
- `сейчас`

## Проблемы, которые мы нашли

1. Первая batch-нарезка была слишком короткой: часть слов визуально не успевала произноситься.
   Исправление: сделали `selected_clips_padded` с большим запасом до и после слова.

2. `spk09` и `spk10` оказались плохими кандидатами для честного lip-reading датасета:
   в них есть вставки экранов, телефон, кадры без нормального лица и live/монтаж.
   Исправление: сделали отдельный split `*_no_spk09_10`.

3. В первом PyTorch baseline модель стала предсказывать самый частый класс `есть`.
   Это показало проблему дисбаланса классов.
   Исправление/эксперимент: попробовали `class-weighted CrossEntropyLoss`.

## PyTorch baseline

В Colab был собран baseline:

```text
CSV
-> load_video через OpenCV
-> tensor [3, 24, 96, 96]
-> PyTorch Dataset
-> DataLoader
-> Simple3DCNN
-> CrossEntropyLoss
-> train/validation loop
```

Модель:

```text
Conv3d -> ReLU -> MaxPool3d
Conv3d -> ReLU -> MaxPool3d
Conv3d -> ReLU -> MaxPool3d
AdaptiveAvgPool3d
Flatten
Linear(64, num_classes)
```

Результат первого baseline:

- обычный loss дал validation accuracy около 30%, но это оказалось misleading;
- модель почти всегда предсказывала самый частый класс `есть`;
- weighted loss сделал оценку честнее, но качество осталось около случайного уровня для 10 классов.

Вывод: пайплайн обучения работает, но датасет и модель нужно улучшать.

## Что еще нужно сделать для курсовой

Минимальный план:

1. Привести Colab notebook в чистый вид и сохранить в репозиторий.
2. Обновить README, чтобы там были актуальные padded-файлы и Colab-инструкция.
3. Добрать/заменить плохих спикеров `spk09`, `spk10`.
4. Сделать balanced split или balanced sampler.
5. Обучить baseline заново и построить графики loss/accuracy.
6. Добавить confusion matrix и per-class accuracy.
7. Написать отчет: датасет, пайплайн, модель, эксперименты, проблемы, выводы.

Расширенный план:

1. Попробовать более сильную temporal model: CNN+GRU/LSTM или 3D-CNN+Transformer.
2. Увеличить датасет до более ровного количества клипов на слово и спикера.
3. Сравнить full-frame clips и mouth crops.
4. Добавить ручной quality review для спорных клипов.

## Где лежат отчеты и артефакты

- старый PDF-отчет: `Tarasova_lip_read_rus_dataset_report (3) (1).pdf`
- описание пайплайна: `README_PIPELINE.md`
- описание текущего датасета: `semester4_coursework/dataset_local/ru_dataset/README.md`
- quality report: `ru_dataset/04_quality_reports/quality_padded/quality_report.md`
- картинки для проверки качества:
  - `ru_dataset/04_quality_reports/quality_padded/padded_timing_compare.jpg`
  - `ru_dataset/04_quality_reports/quality_padded/spk09_spk10_contact_sheet.jpg`
  - `ru_dataset/04_quality_reports/quality_padded/flagged_contact_sheet.jpg`
- архив для Colab: `ru_dataset/05_colab_package/colab_lipreading_dataset.zip`

Полного нового отчета для 4 семестра пока нет. Этот файл является кратким статус-репортом.
