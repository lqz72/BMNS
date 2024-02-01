# Implementation for BMNS
Batch-Mix Negative Sampling for Learning Recommendation Retrievers, CIKM2023


## Datasets
We provide Gowalla dataset for reproduction, you can also download other datasets and process them according to our method.

### Data Link

 Download raw datasets from: https://recbole.io/dataset_list.html

### Preprocess
1. Edit `datasets/raw_data/XXX/XXX.yaml`
2. Rewrite&Run   `datasets/data_preprocess/inter2mtx.py`
3. The data file is saved in `datasets/clean_data`

## Algs
+ `--debias 1` : SSL
+ `--debias 2` : SSL-Pop
+ `--debias 3` : correct-sfx
+ `--debias 4` : MNS, Mixed Negative Sampling
+ `--debias 5` : BMNS, Batch-Mix Negative Sampling
+ `--debias 6` : BIR, Batch Importance Resampling

## Quick Start
+ See files `run.sh` and `run_baseline.sh`
+ The experimental results are as follows:

| Method/Metric | NDCG@10            | NDCG@20           | NDCG@50           | Recall@10         | Recall@20         | Recall@50         |
| ------------- | -------------------| ------------------| ------------------| ------------------| ------------------| ------------------|
| SSL           | 0.1440±5.0         | 0.1154±3.1        | 0.1933±5.9        | 0.1678±4.3        | 0.2942±7.0        | 0.2653±4.08       |
| SSL-Pop       | 0.1597±9.4         | 0.1201±6.3        | 0.2060±13.0       | 0.1709±10.4       | 0.3027±8.3        | 0.2640±7.1        |
| correct-sfx   | 0.1548±8.4         | <u>0.1223±4.6</u> | <u>0.2056±7.3</u> | <u>0.1765±5.2</u> | <u>0.3085±9.7</u> | <u>0.2755±8.2</u> |
| MNS           | 0.1602±9.5         | 0.1207±6.6        | 0.2069±9.7        | 0.1718±7.2        | 0.3038±10.4       | 0.2651±8.6        |
| BIR           | <u>0.1607±10.4</u> | 0.1207±6.4        | 0.2069±12.4       | 0.1715±8.3        | 0.3039±11.0       | 0.2650±9.9        |
| BMNS          | **0.1720±7.4**     | **0.1344±6.1**    | **0.2270±7.6**    | **0.1935±7.7**    | **0.3391±9.9**    | **0.3022±10.8**   |
| Impv.         | 7.06%              | 9.88%             | 9.68%             | 9.64%             | 9.92%             | 9.69%             |

## Cite us

```
@inproceedings{10.1145/3583780.3614789,
author = {Fan, Yongfu and Chen, Jin and Jiang, Yongquan and Lian, Defu and Guo, Fangda and Zheng, Kai},
title = {Batch-Mix Negative Sampling for Learning Recommendation Retrievers},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3614789},
doi = {10.1145/3583780.3614789},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {494–503},
numpages = {10},
keywords = {information retrieval, negative sampling, recommender systems},
location = {<conf-loc>, <city>Birmingham</city>, <country>United Kingdom</country>, </conf-loc>},
series = {CIKM '23}
}
```

