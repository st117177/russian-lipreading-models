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
- weighted loss и balanced sampler сделали оценку честнее, но качество осталось около случайного уровня для 10 классов.

Вывод: пайплайн обучения работает, но датасет и модель нужно улучшать.

Следующий добавленный baseline:

```text
mouth crop video
-> RGB или grayscale frames
-> frame-level CNN
-> BiGRU по последовательности кадров
-> Linear classifier
-> word class
```

Зачем он нужен:

- `Simple3DCNN` проверяет технический пайплайн, но плохо моделирует порядок движений губ;
- `FrameCNN+BiGRU` отдельно извлекает признаки кадра и затем анализирует временную последовательность;
- grayscale-режим проверяется как оптимизация: меньше входных данных и меньше зависимости от цвета/освещения.

Промежуточные результаты на speaker-based split:

- `FrameCNN+BiGRU RGB`: best validation accuracy около `0.295`;
- `FrameCNN+BiGRU grayscale`: best validation accuracy около `0.314`;
- grayscale немного лучше RGB, но обе модели всё ещё в основном предсказывают частое слово, поэтому проблему нельзя считать решённой.

После этого в notebook добавлен random-split sanity check для `FrameCNN+BiGRU grayscale`.
Это не финальная честная метрика, а диагностический эксперимент:

- если random split заметно лучше speaker-based split, значит главная проблема в обобщении на новых спикеров;
- если random split тоже слабый, нужно проверять качество разметки, alignment и mouth crops.

Результат random-split sanity check:

- `FrameCNN+BiGRU grayscale random split`: best validation accuracy около `0.25`;
- random split не стал лучше speaker-based split, значит проблема не только в unseen speakers.

После этого добавлен model-based clip audit:

```text
trained grayscale CNN+BiGRU
-> per-clip loss and prediction
-> high-loss CSV files
-> wrong-prediction CSV files
-> contact sheets for quick visual review
```

Идея: не чистить руками весь датасет, а сначала посмотреть 30-50 самых подозрительных клипов.

Следующий модельный шаг:

```text
grayscale mouth ROI
-> ResNet18 frame encoder
-> BiGRU temporal encoder
-> Linear classifier
-> word class
```

Это LRW-style baseline: он ближе к архитектурам из word-level lip-reading проектов,
где используется более сильный frame encoder и отдельный temporal backend.
Сейчас он реализован через `torchvision.models.resnet18` без внешних LRW-pretrained
checkpoint. Подключение готовых LRW-весов можно оставить как отдельный расширенный этап.

Промежуточный результат `ResNet18+BiGRU`:

- параметров: около `11.7M`;
- `ResNet18+BiGRU grayscale speaker split`: best validation accuracy около `0.314`;
- результат примерно совпал с маленькой `FrameCNN+BiGRU grayscale`;
- модель всё ещё в основном предсказывает самый частый класс.

Вывод: простое увеличение архитектуры не решило задачу. Следующий шаг должен быть
не просто “модель побольше”, а более аккуратная стратегия обучения: pretrained/frozen
frame encoder, augmentation, class balancing для ResNet-модели и/или дальнейшая
проверка качества данных.

После этого в notebook добавлен диагностический блок `Tiny Overfit Test`.
Он нужен не для финального качества, а для проверки самого training pipeline:

```text
train subset из 16-32 клипов
-> FrameCNN+BiGRU grayscale
-> много эпох только на этих клипах
-> ожидаем train accuracy около 90-100%
```

Как интерпретировать результат:

- если модель не может запомнить 16-32 клипа, значит надо сначала чинить
  `clip_path -> word -> class_id`, загрузку видео, target ids, loss, learning rate
  или training loop;
- если модель легко запоминает маленький subset, значит базовый pipeline работает,
  а низкая validation accuracy на полном split связана скорее с размером, качеством,
  балансом данных или обобщением на новых спикеров;
- если tiny overfit успешен, только после этого имеет смысл продолжать сравнение
  `FrameCNN+BiGRU`, `ResNet18+BiGRU`, pretrained/frozen encoder и augmentation.

Первый запуск этого теста в Colab дал:

- 16 клипов: best memorization accuracy `0.875` (14 из 16);
- 32 клипа: best memorization accuracy `0.65625` (21 из 32);
- loss уменьшался, то есть forward/backward/optimizer работают, но формальный
  порог `0.90` не был достигнут.

После этого tiny-overfit блок был сделан более строгим и воспроизводимым:

- фиксируются random seeds Python, NumPy и PyTorch;
- 16/32 видео один раз декодируются и кэшируются в RAM;
- `DataLoader` работает с `num_workers=0`;
- отдельная tiny-модель использует `GroupNorm` вместо нестабильного на маленьком
  batch `BatchNorm`;
- dropout, augmentation и weight decay отключены;
- обучение идет до 150 эпох с early stopping при 100%;
- проверяются пути, соответствие `word -> class_id -> word`, конфликтующие
  дубликаты тензоров, нулевое движение и нулевая дисперсия пикселей;
- сохраняются history CSV, errors CSV и первые/средние/последние кадры ошибок.

Новые критерии перехода: минимум `0.95` на 16 клипах и минимум `0.90` на
32 клипах. До прохождения обоих критериев тяжелые model experiments запускать
не нужно.

Стабильный tiny-overfit был запущен в Colab и успешно прошел оба gate:

- 16 клипов: `1.00` memorization accuracy на эпохе 43;
- 32 клипа: `1.00` memorization accuracy на эпохе 129;
- label round trip: корректен;
- conflicting duplicate tensors: 0;
- zero-motion clips: 0;
- zero-variance clips: 0;
- итог: `PIPELINE PASS`.

Это подтверждает, что video loading, labels, forward pass, loss, backward pass
и optimizer способны обучать модель. Низкое качество полного baseline теперь
нужно исследовать как проблему обобщения, архитектуры, дисбаланса и объема
данных, а не как фундаментальную ошибку training loop.

Для следующего этапа добавлен отдельный notebook
`notebooks/lipreading_framecnn_bigru_controlled_colab.ipynb`. Он обучает
`FrameCNN+BiGRU grayscale` на 700 train-клипах и проверяет на 105 клипах нового
спикера `spk06`: максимум 30 эпох, AdamW, early stopping с patience 7, выбор
checkpoint по validation macro-F1. Test split в этом эксперименте не используется.

Контролируемый `FrameCNN+BiGRU grayscale` был запущен в Colab. Результат:

- early stopping после 8 эпох;
- лучшая эпоха: 1;
- validation accuracy: `0.2952`;
- validation macro-F1: `0.0736`;
- validation balanced accuracy: `0.1106`;
- majority-class baseline (`есть`): `0.3048`;
- модель предсказала только 3 класса из 10;
- `есть` было предсказано для 89 из 105 validation-клипов;
- test split не использовался.

Вывод: компактная модель может запоминать маленький subset, но почти не учится
обобщать на полном speaker-based split и в основном схлопывается в частый класс.

Следующий контролируемый эксперимент вынесен в notebook
`notebooks/lipreading_frozen_resnet18_bigru_colab.ipynb`:

```text
grayscale frames
-> repeat to RGB + ImageNet normalization
-> frozen ImageNet-pretrained ResNet18
-> cached 24 x 512 frame features
-> trainable BiGRU classifier
```

Frozen features извлекаются один раз и сохраняются на Google Drive. Это делает
эксперимент быстрее и проверяет, помогают ли готовые visual features без обучения
11.7M параметров ResNet18 на маленьком датасете. Test split не используется.

Эксперимент `Frozen ImageNet ResNet18 features + BiGRU` был запущен:

- feature cache: train `[700, 24, 512]`, validation `[105, 24, 512]`;
- лучшая эпоха: 6;
- validation accuracy: `0.2857`;
- validation macro-F1: `0.0935`;
- validation balanced accuracy: `0.1594`;
- предсказано 5 классов из 10;
- train accuracy дошла примерно до `0.50`, но validation не росла;
- test split не использовался.

По сравнению с FrameCNN macro-F1 вырос с `0.0736` до `0.0935`, balanced accuracy
с `0.1106` до `0.1594`, число предсказываемых классов с 3 до 5. Однако accuracy
осталась ниже majority baseline, поэтому улучшение недостаточно для размораживания
большого ResNet-блока.

Следующий дешевый diagnostic experiment:
`notebooks/lipreading_frozen_resnet18_bigru_balanced_colab.ipynb`. Он повторно
использует готовые features и меняет только train sampling: каждое непустое
сочетание `(speaker, word)` получает одинаковую суммарную вероятность. Это
проверяет влияние доминирования `spk04` и частых слов до расширения датасета.

Эксперимент с balanced sampler был запущен. Лучший checkpoint получен на
14-й эпохе:

- validation accuracy: `0.2381`;
- validation macro-F1: `0.1669`;
- validation balanced accuracy: `0.1768`;
- предсказано 9 классов из 10;
- train accuracy дошла до `0.8414`, после чего early stopping завершил обучение;
- test split не использовался.

Балансировка решила именно проблему collapse: по сравнению с обычным
`FrameCNN+BiGRU` число предсказываемых классов выросло с 3 до 9, а macro-F1
с `0.0736` до `0.1669`. Обычная accuracy снизилась, потому что модель перестала
почти всегда выбирать частое слово `есть`. При этом большой разрыв между train
и validation показывает переобучение на трех train-спикерах. Oversampling не
может создать разнообразие движений губ, ракурсов и внешности новых людей.

Текущий вывод: прежде чем размораживать ResNet18 или добавлять более тяжелую
архитектуру, нужно собрать dataset v2 с новыми спикерами. Подробный план:
`data/DATASET_V2_PLAN.md`.

После этого добавлен эксперимент `Frozen ImageNet ResNet18+BiGRU`:

```text
ImageNet-pretrained ResNet18 frame encoder
-> encoder frozen
-> train only BiGRU temporal encoder + classifier
```

Зачем: проверить, помогут ли готовые общие визуальные признаки на маленьком
датасете, где обучение ResNet18 с нуля оказалось неэффективным. Это ещё не
lip-reading pretrained модель, но более контролируемый шаг перед интеграцией
LRW-pretrained checkpoint.

## Что еще нужно сделать для курсовой

Минимальный план:

1. Считать диагностику pipeline завершенной: tiny overfit 16/32 прошел.
2. Считать первое сравнение моделей на dataset v1 завершенным: проверены
   `FrameCNN+BiGRU`, frozen `ResNet18+BiGRU` и balanced sampler.
3. Зафиксировать dataset v1 и не менять его файлы или split.
4. Для dataset v2 добавить 2-4 новых хороших спикера. `spk04` больше не
   увеличивать.
5. Сначала добавить новых людей в train и сохранить старый `spk06` как
   validation, чтобы результат v2 можно было честно сравнить с v1.
6. Повторить frozen `ResNet18+BiGRU` с balanced sampler на v2 по тому же
   протоколу.
7. Только после v2 решить, нужен ли LRW-style `3D frontend + ResNet18 + TCN`
   или размораживание последнего блока ResNet18.
8. Для двух лучших вариантов выполнить по три запуска с разными seed, выбрать
   модель по validation macro-F1 и один раз оценить ее на test.
9. Написать отчет: датасет, pipeline, диагностика, модели, эксперименты и выводы.

Расширенный план:

1. Если `ResNet18+BiGRU` тоже слабая, сначала проверить датасет/разметку, затем попробовать CNN+TCN.
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
