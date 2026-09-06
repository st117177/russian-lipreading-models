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
- стало в `ru_dataset/01_intermediate_clips/selected_labels_padded_clean.csv`: 1222 clean-клипа, 14 слов, 8 спикеров;
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

## Последний validation-only эксперимент

Для pretrained LRW MS-TCN проверили temporal masking: в половине train-примеров
занулялся случайный блок из четырех соседних кадров. На каждом из трех held-out
спикеров (`spk03`, `spk17`, `spk19`) запускали три seed. Обычный pretrained
MS-TCN получил mean validation macro-F1 `0.5832`, masked-вариант — `0.5391`.
Парная разница составила `-0.0441 +/- 0.0271`, masked-вариант выиграл `0/9`
сравнений. Поэтому основной моделью остается MS-TCN без temporal masking.
Эксперимент не использовал test split.
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

## Dataset v2: выполненное расширение

Dataset v1 зафиксирован без изменений: 955 клипов, 10 слов и 6 спикеров.
Для dataset v2 локально обработаны 5 новых исходных видео общей длительностью
около 147 минут от двух новых анонимных спикеров. Сведения об авторах, URL и
исходных идентификаторах видео хранятся только локально и в репозиторий не
добавляются.

Выполнены этапы:

1. Видео и автосубтитры загружены пакетно.
2. Видео разбиты на 32 пятиминутных чанка.
3. Для всех 32 чанков подготовлены WAV 16 kHz и очищенный текст.
4. Через BAS WebMAUS автоматически получены 32 TextGrid.
5. Из TextGrid получены `words_frames.txt` с учетом FPS каждого видео.
6. Нарезано 437 word-level клипов с временным padding вокруг слова.
7. Automatic quality check оставил 408 клипов с найденным лицом.
8. Созданы 408 mouth ROI; по contact sheets исключены 8 ложных кропов.
9. Итоговое расширение: 400 чистых новых mouth-клипов.

Новый speaker-based split:

- train: 1100 клипов, 5 спикеров (`spk03`, `spk04`, `spk05`, `spk11`, `spk12`);
- validation: 105 клипов, прежний `spk06`;
- test: 150 клипов, прежние `spk07`, `spk08`;
- всего: 1355 клипов, 10 слов, 8 спикеров.

Таким образом, между v1 и v2 изменился только train: добавлены 400 клипов и
два новых спикера. Validation и test сохранены без изменений, поэтому следующий
эксперимент сможет измерить влияние разнообразия обучающих спикеров.

Для воспроизводимого запуска подготовлены:

- локальный архив `colab_lipreading_dataset_v2.zip`, содержащий только 1355
  клипов из split и файлы разметки;
- отдельный notebook
  `notebooks/lipreading_frozen_resnet18_bigru_balanced_v2_colab.ipynb`;
- отдельная папка результатов на Google Drive, поэтому результаты v1 не
  перезаписываются.

Notebook повторяет на v2 тот же протокол, что дал лучший macro-F1 на v1:
frozen ImageNet ResNet18, BiGRU и speaker-word balanced sampler. Признаки v2
извлекаются заново, потому что старый cache содержит только 700 train-клипов v1.
Validation остается прежним (`spk06`), test не используется.

Эксперимент на dataset v2 выполнен с seed 42:

- лучшая эпоха: 14;
- train accuracy на лучшей эпохе: около `0.67`;
- validation accuracy: `0.1524`;
- validation macro-F1: `0.1301`;
- validation balanced accuracy: `0.1655`;
- модель предсказала все 10 классов;
- test split не использовался.

Для сравнения, тот же balanced baseline на v1 с seed 42 дал accuracy `0.2381`,
macro-F1 `0.1669`, balanced accuracy `0.1768` и 9 предсказываемых классов.
Следовательно, в одном запуске добавление 400 клипов и двух train-спикеров не
улучшило качество на `spk06`: macro-F1 снизился на `0.0368`. При этом исчез
полный class collapse, так как v2 предсказывает все классы. Это полезный
отрицательный результат, но одного seed и 105 validation-клипов недостаточно
для окончательного вывода.

Для проверки устойчивости добавлен короткий notebook
`notebooks/lipreading_v1_v2_balanced_multiseed_colab.ipynb`. Он использует уже
готовые feature caches, повторяет одинаковый balanced BiGRU protocol для v1 и
v2 на seed `42`, `123`, `2026`, считает среднее, стандартное отклонение и
парную разницу `v2 - v1`. ResNet18 и видео повторно не запускаются, test не
используется.

Первый диагностический запуск multi-seed notebook остановился до обучения и
обнаружил, что v1 и v2 назначили одним и тем же словам разные числовые class ID
из-за разного порядка `vocab.txt`. Сами validation-слова, спикер и frozen
features совпадают (`max feature difference = 0`). Notebook исправлен: для
обоих датасетов targets теперь заново вычисляются как
`word_to_idx[word]` по единому словарю v1. Это не меняет предыдущие отдельные
результаты v1/v2, где каждый cache обучался и оценивался со своим согласованным
словарем, но делает совместное сравнение корректным.

Исправленный multi-seed эксперимент успешно выполнил все шесть запусков.
Результаты `Frozen ResNet18 features + balanced BiGRU`:

- v1 mean validation accuracy: `0.1778 ± 0.0574`;
- v2 mean validation accuracy: `0.1587 ± 0.0440`;
- v1 mean macro-F1: `0.1311 ± 0.0312`;
- v2 mean macro-F1: `0.1416 ± 0.0458`;
- v1 mean balanced accuracy: `0.1471 ± 0.0258`;
- v2 mean balanced accuracy: `0.1715 ± 0.0104`;
- средняя парная разница macro-F1 `v2 - v1`: `+0.0106`;
- v2 выиграл по macro-F1 на двух seed из трех;
- среднее число предсказываемых классов выросло с `8.67` до `9.33`;
- test split не использовался.

Вывод: добавление 400 клипов и двух train-спикеров дало небольшой положительный
тренд по macro-F1 и более стабильную balanced accuracy, но обычная accuracy не
выросла, а абсолютное качество остается низким. При трех seed это полезный
контролируемый результат, но не сильное статистическое доказательство. Дальше
нужно улучшать модель, а не продолжать хаотично собирать видео.

Следующий controlled experiment:
`notebooks/lipreading_v2_tcn_backend_multiseed_colab.ipynb`. Он сохраняет
dataset v2, frozen ResNet18 features, balanced sampler и три seed, но заменяет
BiGRU на компактный residual TCN. Это изолирует влияние temporal backend.
Эксперимент не называется полной LRW-style моделью, потому что пока не включает
lip-reading-pretrained 3D frontend и visual encoder.

TCN experiment выполнен на seed `42`, `123`, `2026`:

- mean validation accuracy: `0.1206 +/- 0.0291`;
- mean validation macro-F1: `0.1182 +/- 0.0287`;
- mean validation balanced accuracy: `0.1742 +/- 0.0312`;
- среднее число предсказываемых классов: `8.33` из 10;
- TCN выиграл у BiGRU по macro-F1 на одном seed из трех;
- mean macro-F1 оказался ниже BiGRU на `0.0234`;
- test split не использовался.

Вывод: TCN немного сохранил balanced accuracy, но снизил macro-F1, accuracy и
class coverage. Простая замена temporal backend не устраняет основную проблему.
Следующая проверяемая гипотеза относится к visual frontend: ImageNet-признаки
не специализированы для артикуляции губ.

Для этой проверки подготовлен notebook
`notebooks/lipreading_lrw_pretrained_frontend_bigru_multiseed_colab.ipynb`.
Он использует официальный LRW-pretrained `3D Conv + ResNet18` как замороженный
visual encoder, извлекает новый feature cache и обучает тот же balanced BiGRU
на трех seed. Поэтому относительно предыдущего лучшего протокола меняется
только источник визуальных признаков. Test split не загружается.

LRW-pretrained experiment успешно выполнен. Все ключи официального checkpoint
совпали с архитектурой, train/validation feature caches имеют ожидаемые формы
`[1100, 24, 512]` и `[105, 24, 512]`. Результаты по трем seed:

- mean validation accuracy: `0.2857 +/- 0.0530`;
- mean validation macro-F1: `0.2736 +/- 0.0290`;
- mean validation balanced accuracy: `0.3157 +/- 0.0039`;
- среднее число предсказываемых классов: `9.67` из 10;
- выигрыш mean macro-F1 относительно ImageNet frontend: `+0.1320`;
- LRW frontend выиграл у ImageNet frontend на всех трех seed;
- test split не использовался.

Отдельные macro-F1: seed 42 — `0.2411`, seed 123 — `0.2827`, seed 2026 —
`0.2971`. Результат устойчивее предыдущих экспериментов: стандартное отклонение
balanced accuracy равно только `0.0039`. Главный модельный вывод курсовой:
предобучение на задаче чтения по губам дает намного больший эффект, чем замена
temporal backend или добавление ImageNet-признаков.

На этом этапе архитектура была выбрана до просмотра исходного test:

```text
24 grayscale mouth frames
-> frozen LRW-pretrained 3D Conv + ResNet18
-> sequence [24, 512]
-> trainable BiGRU
-> classifier for 10 Russian words
```

Для единственного финального этапа подготовлен test-only notebook
`notebooks/lipreading_lrw_pretrained_final_test_colab.ipynb`. Он не обучает
модель и не меняет гиперпараметры: загружает три уже выбранных по validation
checkpoint, извлекает признаки 150 test-клипов спикеров `spk07` и `spk08` и
считает mean/std test metrics. После этого test нельзя использовать для нового
подбора архитектуры.

Историческое test-only оценивание старого crop выполнено без ошибок и без обучения:

- seed 42: accuracy `0.1600`, macro-F1 `0.1331`, balanced accuracy `0.1612`;
- seed 123: accuracy `0.2133`, macro-F1 `0.1742`, balanced accuracy `0.2087`;
- seed 2026: accuracy `0.1733`, macro-F1 `0.1515`, balanced accuracy `0.1713`;
- mean test accuracy: `0.1822 +/- 0.0278`;
- mean test macro-F1: `0.1530 +/- 0.0206`;
- mean test balanced accuracy: `0.1804 +/- 0.0250`;
- среднее число предсказываемых классов: `9.67` из 10.

Accuracy majority baseline на test равна `28/150 = 0.1867` для слова `чтобы`.
Средняя accuracy модели близка к ней, но majority classifier имеет balanced
accuracy `0.1` и macro-F1 около `0.0315`, потому что не распознает остальные
девять классов. Финальная модель существенно лучше по обеим class-balanced
метрикам и использует почти весь словарь.

Test macro-F1 ниже validation macro-F1 (`0.1530` против `0.2736`). Это
подтверждает, что основным ограничением остается перенос на незнакомых спикеров.
Слово `потом` не распознано ни одним checkpoint, но его test support равен
только трем клипам; `время` также представлено лишь семью клипами. Этот результат
фиксируется как ограничение, а не используется для нового подбора модели.
Краткая финальная таблица вынесена в `reports/FINAL_MODEL_RESULTS.md`.

## Landmark v3: исправление preprocessing

После анализа ошибок выяснилось, что старый `create_mouth_crops.py` находил лицо
на нескольких кадрах и применял один неподвижный lower-face crop ко всему
клипу. При наклонах и движении головы положение рта заметно менялось, а входные
данные не совпадали с выровненными mouth ROI, на которых обучался LRW frontend.

Добавлен `scripts/create_landmark_mouth_crops.py`:

- MediaPipe Face Mesh на каждом кадре;
- центр ROI по landmarks губ;
- выравнивание по линии глаз;
- временное сглаживание центра, размера и угла;
- интерполяция единичных пропусков landmarks;
- порог не менее 70% кадров с успешной детекцией;
- отдельные manifest, failed CSV и old/new contact sheet.

Старые model inputs не перезаписывались. В landmark v3 вошли 1085 из 1100
train-клипов и все 105 validation-клипов. Test намеренно не включался.

На неизменных LRW-pretrained frontend, balanced BiGRU, seeds и validation split
получены результаты:

- seed 42: accuracy `0.5524`, macro-F1 `0.5513`, balanced accuracy `0.5859`;
- seed 123: accuracy `0.5333`, macro-F1 `0.5244`, balanced accuracy `0.5663`;
- seed 2026: accuracy `0.5524`, macro-F1 `0.5320`, balanced accuracy `0.5838`;
- mean accuracy: `0.5460 +/- 0.0110`;
- mean macro-F1: `0.5359 +/- 0.0139`;
- mean balanced accuracy: `0.5787 +/- 0.0108`;
- модель предсказывает все 10 классов на каждом seed.

Относительно LRW-модели со старым crop mean macro-F1 вырос на `0.2622`, а
landmark v3 выиграл на всех трех paired seed. Значит, прежний crop был одним из
главных ограничений. Конфигурация `landmark v3 + frozen LRW frontend + balanced
BiGRU` фиксируется до нового test.

Старые `spk07` и `spk08` уже были просмотрены и не считаются независимым test
для новой версии. Для финальной оценки нужны минимум два новых спикера; после
их подготовки выполняется один inference-запуск без дальнейшего выбора модели.

## Что еще нужно сделать для курсовой

Минимальный план:

1. Считать диагностику pipeline завершенной: tiny overfit 16/32 прошел.
2. Считать первое сравнение моделей на dataset v1 завершенным: проверены
   `FrameCNN+BiGRU`, frozen `ResNet18+BiGRU` и balanced sampler.
3. Зафиксировать dataset v1 и не менять его файлы или split.
4. Считать dataset v2 подготовленным: добавлены два новых спикера и 400 чистых
   mouth-клипов; `spk04` не увеличивался.
5. Сохранить старый `spk06` как validation и старые `spk07`/`spk08` как test.
6. Считать single-seed повтор frozen `ResNet18+BiGRU` с balanced sampler на v2
   завершенным: улучшение относительно v1 не получено.
7. Считать multi-seed сравнение v1/v2 завершенным: v2 показывает небольшой
   положительный тренд по macro-F1 и balanced accuracy.
8. Считать controlled TCN backend experiment завершенным: улучшения macro-F1
   относительно BiGRU не получено.
9. Считать LRW-pretrained experiment завершенным: модель выиграла у ImageNet
   frontend на всех трех seed.
10. Считать landmark v3 experiment завершенным: macro-F1 вырос с `0.2736` до
    `0.5359`, выигрыш получен на всех трех seed.
11. Зафиксировать `landmark v3 + frozen LRW frontend + balanced BiGRU` и больше
    не подбирать параметры по `spk06`, `spk07` или `spk08`.
12. Считать новый независимый test подготовленным: 508 клипов, 10 слов,
    3 ранее не использованных спикера, одинаковые 50-минутные source ranges.
13. Финальный test-only запуск завершен на трех заранее выбранных checkpoint:
    `0.4370 +/- 0.0104` accuracy, `0.3919 +/- 0.0158` macro-F1 и
    `0.4012 +/- 0.0179` balanced accuracy; все модели предсказывают 10/10 классов.
14. Перенести зафиксированные результаты в текст курсовой: датасет, pipeline,
    диагностика, модели, эксперименты, ограничения и выводы.
15. Считать validation-only regularization experiment завершенным без
    улучшения: лучший новый вариант дал `0.4737` mean macro-F1 против `0.5359`
    у frozen LRW + BiGRU baseline; final test не использовался.
16. Для дальнейшего улучшения сначала добавить новых train-спикеров. Частичную
    разморозку последнего LRW ResNet-блока проверять как отдельный controlled
    experiment и выбирать только по validation.

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
- архив dataset v1 для Colab: `ru_dataset/05_colab_package/colab_lipreading_dataset.zip`
- архив dataset v2 для Colab: `ru_dataset/05_colab_package/colab_lipreading_dataset_v2.zip`
- локальный архив нового final test:
  `ru_dataset/05_colab_package/colab_lipreading_final_test_landmark_v3.zip`
- test-only notebook:
  `notebooks/lipreading_lrw_landmark_v3_untouched_final_test_colab.ipynb`

Полного нового отчета для 4 семестра пока нет. Этот файл является кратким статус-репортом.

## Расширение Train: Dataset V3

После финальной оценки предыдущей версии выполнено отдельное расширение только
train-части. Добавлены четыре новых анонимизированных спикера; сведения об
источниках хранятся только в локальном приватном manifest и не публикуются.

Результат preprocessing:

- 200 минут исходной речи, по 50 минут на спикера;
- 40 пятиминутных чанков и 40 успешных WebMAUS TextGrid;
- 817 padded word-level клипов;
- автоматический quality check оставил 816 клипов;
- landmark preprocessing создал 816 из 816 mouth crops размером `96x96`;
- новый train: 1901 клип, 10 слов, 9 спикеров;
- прежний validation: 105 клипов `spk06`;
- максимальная доля одного train-спикера снизилась до `24.8%`;
- test в train/validation ZIP не включён.

Для контролируемого сравнения создан notebook
`notebooks/lipreading_lrw_landmark_dataset_v3_multiseed_colab.ipynb`. Он
использует зафиксированную архитектуру LRW-pretrained frontend + balanced BiGRU,
те же seeds и validation macro-F1.

Эксперимент выполнен. Dataset v3 получил:

- validation accuracy: `0.5302 +/- 0.0361`;
- validation macro-F1: `0.5091 +/- 0.0214`;
- validation balanced accuracy: `0.5492 +/- 0.0411`;
- все 10 классов предсказывались во всех трех запусках.

Dataset v2 при том же протоколе имел macro-F1 `0.5359 +/- 0.0139`. Dataset v3
проиграл на всех трех seed, средняя разница составила `-0.0268`. Это не означает,
что новые данные бесполезны вообще: результат относится к одному фиксированному
validation-спикеру. Однако простое добавление всех четырех новых спикеров не дало
устойчивого улучшения, поэтому test повторно не открывался.

Следующий шаг - `notebooks/lipreading_dataset_v3_speaker_ablation_colab.ipynb`.
Он использует уже сохраненные LRW feature caches и сравнивает dataset v2 с
вариантами, где каждый новый спикер добавлен отдельно. Так можно отделить эффект
количества данных от качества и доменного несоответствия конкретного спикера.

Численные результаты сохранены в `reports/results/`. Test split в эксперименте
dataset v3 не использовался.

## Dataset v3: ablation по новым спикерам

На T4 полностью выполнены контроль `base_v2` и четыре варианта с добавлением
одного нового спикера. Контроль воспроизвел прежний результат: validation
macro-F1 `0.5359 +/- 0.0139`. Средние результаты одиночных добавлений:

- `spk16`: `0.5337 +/- 0.0440`, разница `-0.0022`, выигрыш на 2 из 3 seed;
- `spk17`: `0.4827 +/- 0.0174`, разница `-0.0532`;
- `spk18`: `0.4630 +/- 0.0767`, разница `-0.0729`;
- `spk19`: `0.5268 +/- 0.0139`, разница `-0.0091`.

Ни один новый спикер не улучшил среднее качество. `spk16` близок к нейтральному,
но результат нестабилен. Повтор `all_v3` в этом запуске оборвался после первого
seed, поэтому он не включен в ablation CSV; полный all-speaker результат уже был
получен отдельным экспериментом выше.

Автоматический quality check нашел только один клип без лица, но анализ разметки
показал сильную разреженность групп `спикер x слово`: например, у `spk16` есть
115 клипов слова «есть» и только один клип «человек», а у `spk18` только два
клипа «потом». Текущий inverse-frequency sampler многократно повторяет такие
единичные примеры. Следующий эксперимент должен сравнить `uniform`, балансировку
по словам, текущую балансировку по `спикер x слово` и ее сглаженный вариант.

GPU-результаты сохранены в `reports/results/dataset_v3_single_speaker_gpu_*.csv`.
Test split не использовался.

Для следующего шага создан
`notebooks/lipreading_dataset_v3_sampler_ablation_colab.ipynb`. Он сравнивает
четыре стратегии на одних и тех же features и seeds: обычный shuffle,
балансировку по слову, `1 / count(спикер, слово)` и сглаженный вес
`1 / sqrt(count(спикер, слово))`. Основной test в notebook не загружается.

## Dataset v3: ablation стратегии sampling

Эксперимент выполнен на T4 для трех фиксированных seed. Архитектура, LRW feature
cache, validation-спикер и training protocol не менялись; менялся только способ
формирования train batches. Получены средние validation macro-F1:

- `speaker_word_inverse`: `0.5091 +/- 0.0214`;
- `speaker_word_sqrt`: `0.5201 +/- 0.0562`;
- `word_inverse`: `0.5487 +/- 0.0821`;
- `uniform_shuffle`: `0.5510 +/- 0.0081`.

Обычный shuffle улучшил macro-F1 относительно текущего sampler на `+0.0420` и
выиграл на всех трех paired seed. Он также оказался самым стабильным по
стандартному отклонению. Это подтверждает гипотезу, что полная балансировка по
редким группам `спикер x слово` слишком часто повторяла единичные клипы и
ухудшала обобщение.

Для следующего dataset v3 запуска выбран `uniform_shuffle`. Результат относится
только к validation; final test не загружался и не использовался. Полные числа
сохранены в `reports/results/dataset_v3_sampler_ablation_*.csv`.

## Следующий эксперимент: partial fine-tuning LRW encoder

Подготовлен
`notebooks/lipreading_lrw_layer4_finetune_dataset_v3_colab.ipynb`. Он сначала
воспроизводит frozen LRW + BiGRU baseline с `uniform_shuffle` и seed 42 на
сохраненных признаках. Затем размораживается только последний блок
`ResNet18.layer4`; LRW frontend и более ранние блоки остаются frozen, а
BatchNorm не обновляет статистики на маленьких batch.

Это single-seed decision gate. Partial fine-tuning сохраняется как основное
направление только при улучшении validation macro-F1 минимум на `0.02`.
При успехе тот же протокол повторяется на seed 123 и 2026. При неуспехе frozen
encoder остается основной моделью, а следующим отдельным экспериментом будет
легкая spatial/temporal augmentation. Test split в этом notebook не читается.

## Dataset v3: результат partial fine-tuning

Эксперимент выполнен на T4 с seed 42. Frozen baseline был воспроизведен точно:
validation macro-F1 `0.5454`. После разморозки `ResNet18.layer4` лучший результат
получен на эпохе 3: macro-F1 `0.5545`, accuracy `0.5810`. Прирост составил только
`+0.0092`, что ниже заранее установленного порога `+0.02`, поэтому decision gate
дал `NO PASS`.

Train macro-F1 при этом вырос до `0.9937`, а validation после лучшей эпохи начала
ухудшаться. Это признак быстрого переобучения 8.39 млн параметров `layer4` на
небольшом train. Последний блок снова зафиксирован; трех-seed fine-tuning и test
не запускались.

## Dataset v3: frozen LRW + horizontal flip

Следующим изолированным экспериментом стала пространственная аугментация без
изменения архитектуры. Для каждого train-клипа извлечены дополнительные frozen
LRW-признаки после горизонтального отражения mouth ROI. Original и hflip features
объединены только в train; validation-спикер не изменялся, test не загружался.

Результаты по трем seed относительно соответствующего `uniform_shuffle` baseline:

- seed 42: `0.5750`, парный прирост `+0.0296`;
- seed 123: `0.5694`, парный прирост `+0.0091`;
- seed 2026: `0.5880`, парный прирост `+0.0406`;
- mean macro-F1: `0.5775 +/- 0.0095`;
- средний парный прирост: `+0.0264`, выигрыш на 3 из 3 seed.

Decision gate пройден: horizontal flip сохраняется как полезная train-аугментация.
Воспроизводимый notebook:
`notebooks/lipreading_lrw_frozen_hflip_dataset_v3_colab.ipynb`. Числа сохранены
в `reports/results/dataset_v3_frozen_hflip_*.csv`. Результат пока относится только
к validation и не заменяет уже замороженную final-test оценку.

## Dataset v3: cross-speaker проверка horizontal flip

Чтобы не принимать решение только по одному validation-спикеру `spk06`, проведена
дополнительная проверка на трех спикерах из train. `spk03`, `spk17` и `spk19`
по очереди полностью исключались из обучения и использовались как unseen-speaker
validation. У каждого из них есть примеры всех десяти слов. Для каждого fold
сравнивались original features и original + hflip по seed 42, 123 и 2026.

Итог по девяти парным сравнениям:

- original mean macro-F1: `0.3727`;
- original + hflip mean macro-F1: `0.3900`;
- средний парный прирост: `+0.0173`;
- hflip выиграл в 6 из 9 сравнений;
- разброс парной разницы: `0.0571`.

Horizontal flip подтвержден как небольшая полезная аугментация, но эффект заметно
зависит от спикера и seed. Особенно сложным оказался `spk19`: macro-F1 составлял
примерно `0.22-0.28`. Это подтверждает, что основное ограничение системы сейчас -
speaker generalization, а не неработающий training loop.

Notebook: `notebooks/lipreading_lrw_frozen_hflip_cross_speaker_colab.ipynb`.
Числа: `reports/results/dataset_v3_cross_speaker_hflip_*.csv`. Final test не
загружался и не использовался. Следующая изолированная проверка - заменить только
temporal head `BiGRU` на компактный `TCN`, оставив frozen LRW features, hflip,
folds и seeds неизменными.

## Dataset v3: transfer официального pretrained MS-TCN

Вместо случайно инициализированного компактного TCN использован temporal backend
из официального LRW checkpoint. Frozen LRW visual frontend сначала формирует
последовательность признаков `[24, 512]`, pretrained multi-scale TCN преобразует
ее в pooled embedding размерности 768, а на наших десяти словах обучается только
новый линейный классификатор. Веса visual frontend и MS-TCN не обновляются.

Эксперимент проведен на тех же held-out спикерах `spk03`, `spk17`, `spk19` и
seed 42, 123, 2026. В train использовались original + hflip признаки; final test
не загружался.

- hflip BiGRU mean macro-F1: `0.3900`;
- pretrained MS-TCN mean macro-F1: `0.5832`;
- средний парный прирост: `+0.1932`;
- pretrained MS-TCN выиграл 8 из 9 сравнений;
- единственное формальное поражение: `spk17`, seed 123, разница `-0.0006`.

По отдельным held-out спикерам MS-TCN достиг примерно `0.657-0.699` на `spk03`,
`0.526-0.551` на `spk17` и `0.504-0.538` на ранее особенно сложном `spk19`.
Все запуски предсказывали десять классов. Decision gate пройден, поэтому
официальный pretrained MS-TCN становится ведущей validation-кандидатурой.

Notebook:
`notebooks/lipreading_lrw_pretrained_mstcn_transfer_cross_speaker_colab.ipynb`.
Полные результаты:
`reports/results/dataset_v3_pretrained_mstcn_cross_speaker_*.csv`.

## Dataset v3: transfer официального pretrained DC-TCN

Проверен официальный LRW-pretrained Dense Temporal Convolutional Network как
замена hflip BiGRU. Данные, held-out speakers, seed и линейная голова были
зафиксированы; final test не загружался.

- hflip BiGRU mean validation macro-F1: `0.3900`;
- pretrained DC-TCN mean validation macro-F1: `0.2834`;
- средняя парная разница: `-0.1066`;
- DC-TCN выиграл `0 из 9` сравнений;
- decision gate: `KEEP BiGRU`.

Более сложный pretrained DC-TCN не перенёсся на маленький русский набор и
отклонён. Ведущим validation-кандидатом остаётся pretrained MS-TCN; final test
по-прежнему не используется для настройки.

## Dataset v3: temporal masking для MS-TCN

После выбора pretrained MS-TCN проверили регуляризацию по времени: в 50% train-
примеров занулялся случайный блок из четырех соседних кадров. Для честного
сравнения использовались три held-out спикера (`spk03`, `spk17`, `spk19`) и три
seed; validation оставалась без маскирования.

- обычный pretrained MS-TCN: mean validation macro-F1 `0.5832`;
- temporal masking: mean validation macro-F1 `0.5391`;
- парная разница: `-0.0441 +/- 0.0271`;
- masked-вариант выиграл `0 из 9` сравнений.

Decision gate: `KEEP UNMASKED MS-TCN`. Temporal masking не используется в
финальной конфигурации, но сохраняется как отрицательный контролируемый
эксперимент. Test split не загружался.

Результаты сохранены в:
`reports/results/dataset_v3_pretrained_mstcn_temporal_masking_runs.csv` и
`reports/results/dataset_v3_pretrained_mstcn_temporal_masking_summary.csv`.
