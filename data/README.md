# Datasets (data provenance — the images themselves are NOT committed)

This file documents **where the data come from**. The dataset images are
deliberately **excluded** from the repository (see `.gitignore` → `data/`).
Download each dataset from its official source below and place it under
`data/<name>` in the MVTec AD directory layout expected by
`code/src/data/dataset.py`:

```
data/<name>/<category>/train/good/*.png
data/<name>/<category>/test/<defect_type>/*.png
data/<name>/<category>/ground_truth/<defect_type>/*.png
```

## Primary benchmark — MVTec AD (15 categories)

- Official page: <https://www.mvtec.com/company/research/datasets/mvtec-ad>
- Categories used: `bottle, cable, capsule, carpet, grid, hazelnut, leather,
  metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper`
  (see `data.mvtec_categories` in `code/configs/*.yaml`).
- Place under `data/mvtec_ad/`.

## Cross-dataset — VisA (12 categories)

- Source: <https://github.com/amazon-science/spot-anomaly-benchmark>
- Categories used: `candle, capsules, cashew, chewinggum, fryum, macaroni1,
  macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum`
  (see `VISA_CATS` in `code/scripts/_eval_full.py`).
- Place under `data/visa/` in MVTec layout.

## Auxiliary datasets

- **ELPV** — electroluminescence images of photovoltaic cells, used as a
  cross-domain migration check. Public ELPV dataset; place under
  `data/processed/pvel_ad/`.
  - Official repository: <https://github.com/zae-bayern/elpv-dataset>
    (2,624 images of 300×300 px 8-bit grayscale solar cells; also installable
    via `pip install elpv-dataset`; images CC BY-NC-SA 4.0, loader code
    Apache-2.0).
  - Please cite: Buerhop-Lutz et al., *A Benchmark for Visual Identification
    of Defective Solar Cells in Electroluminescence Imagery*, EU PVSEC 2018;
    and Deitsch et al., *Automatic classification of defective photovoltaic
    module cells in electroluminescence images*, Solar Energy 185:455–468,
    2019.
- **NEU-DET** — NEU surface-defect database (Northeastern University), used as
  a simple-data sanity check. Public dataset; place under `data/processed/neu/`.
  - Official homepage (Northeastern University, Kechen Song & Yunhui Yan):
    <http://faculty.neu.edu.cn/songkc/en/zhym/263264/list/index.htm>
  - **NEU-DET** (images with bounding-box annotations, 1,800 images of
    200×200 px, 6 defect classes) — download from the homepage above, or
    directly:
    - Google Drive: <https://drive.google.com/open?id=1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k>
    - Baidu Pan: <https://pan.baidu.com/s/1nBbO-jWDm1_NHDQsc1dRkg> (access code: `pmqx`)
  - **NEU-CLS** (classification variant, images without annotations), if only
    the classification split is needed:
    - Google Drive: <https://drive.google.com/open?id=1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc>
  - Please cite: K. Song and Y. Yan, *A noise robust method based on completed
    local binary patterns for hot-rolled steel strip surface defects*,
    Applied Surface Science 285:858–864, 2013.

## Notes

- Some scripts also reference a PCB cross-category subset. Per the manuscript,
  PCB is retained only as summary conclusions; the raw PCB images are not
  redistributed here.
- The exact evaluation categories and the frozen DiagAD configuration are in
  `code/configs/diagad.yaml` and `code/scripts/_eval_full.py`.
