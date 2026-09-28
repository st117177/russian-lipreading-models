# Russian Lip-Reading Models

Semester 4 coursework on word-level visual speech recognition in Russian. The project continues the dataset and preprocessing work from the previous semester: it expands and checks the data, builds speaker-disjoint splits, compares mouth-region preprocessing, and evaluates PyTorch models with ImageNet and LRW pretraining.

The task is to classify one of 10 Russian words from a short silent mouth video. The repository contains preparation scripts, experiment notebooks, aggregate results, and research notes. **It does not contain the source videos, generated mouth clips, dataset archives, or pretrained checkpoints.** The notebooks require those inputs to be supplied separately.

## Data and evaluation

| Dataset stage | Train | Validation | Test | Purpose |
| --- | ---: | ---: | ---: | --- |
| v1 | 700 | 105 | 150 | Initial 10-word speaker-disjoint baseline |
| v2 | 1,100 | 105 | 150 | Two additional training speakers |
| v3 | 1,901 | 105 | — | Landmark-aligned crop and model selection; validation speaker `spk06` |
| Final benchmark | — | — | 508 | Three held-out speakers, `spk13`–`spk15` |

The v3 training set has nine speakers. The 508-clip benchmark is reported as **post-hoc**, not as a new untouched test: these clips were previously evaluated in a BiGRU experiment. They were not used to train the MS-TCN classifier, but prior inspection limits how independently its result can be interpreted. Macro-F1 is the main comparison metric because class frequencies differ substantially.

## Main results

The following experiments do **not** all use the same validation protocol, so their scores should not be ranked as one table:

| Controlled comparison | Validation macro-F1 |
| --- | ---: |
| ImageNet-pretrained ResNet18 + BiGRU, dataset v2 | 0.1416 ± 0.0458 |
| LRW-pretrained visual frontend + BiGRU, old crop | 0.2736 ± 0.0290 |
| Same LRW frontend + BiGRU, landmark crop | 0.5359 ± 0.0139 |

On dataset v3, changing the sampler to uniform shuffle gave 0.5510 ± 0.0081 macro-F1 on `spk06`; adding horizontal flip raised it to 0.5775 ± 0.0095. In a separate three-speaker, three-seed cross-speaker comparison, the frozen LRW frontend with a pretrained MS-TCN backend and a trainable 10-class head reached **0.5832 ± 0.0785** macro-F1, versus **0.3900** for the hflip BiGRU control.

The selected MS-TCN configuration achieved **0.6124 ± 0.0040 macro-F1** and **0.6280 ± 0.0020 accuracy** on the 508-clip post-hoc benchmark. These test numbers were not used to choose further variants. Detailed protocols, negative ablations, and limitations are in [Current Model Results](reports/FINAL_MODEL_RESULTS.md).

## Where to start

- [Pipeline and script descriptions](scripts/README.md)
- [Landmark preprocessing and boundary experiments](reports/BOUNDARY_ABLATION_V3.md)
- [Dataset v3 model comparison](reports/MODEL_COMPARISON_V3.md)
- [Cross-speaker LRW + pretrained MS-TCN notebook](notebooks/lipreading_lrw_pretrained_mstcn_transfer_cross_speaker_colab.ipynb)
- [Post-hoc MS-TCN benchmark notebook](notebooks/lipreading_lrw_pretrained_mstcn_posthoc_final_test_colab.ipynb)

Local preprocessing dependencies are listed in [`requirements-pipeline.txt`](requirements-pipeline.txt). The notebooks document their own Colab environment and expected dataset archive layout. Because the videos and archives are not published here, the experiments are documented and scripted, but cannot be rerun from this repository alone.

The pretrained visual frontend and temporal backend come from the upstream [Lipreading using Temporal Convolutional Networks](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks) project; its terms apply to the downloaded weights. No pretrained weights are redistributed here.

## Publication note

The raw videos are not included. Some committed audit CSVs retain original video filenames/IDs and clip paths so that temporal-overlap checks can be traced back to their sources. Those identifiers may reveal which public videos were used and should not be treated as anonymized metadata.
