# Index benchmark (recall@10 vs latency and memory)

Dataset size n=20000. Latency is per single query, one thread.

## Filter `none` (selectivity 100.0%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 6.96 | 13.70 | 123 | 29.3 | 29 | 0.6 |
| faiss_hnsw | {'ef_search': 16} | 0.820 | 0.44 | 1.01 | 1966 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 64} | 0.987 | 0.61 | 1.48 | 1338 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 256} | 1.000 | 1.58 | 3.08 | 576 | 34.5 | 34 | 5.7 |
| faiss_ivf | {'nprobe': 1} | 0.162 | 0.14 | 0.25 | 5942 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 8} | 0.716 | 0.25 | 0.32 | 3693 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 32} | 0.999 | 0.36 | 0.52 | 2593 | 30.3 | 167 | 1.4 |
| numpy_exact | {} | 1.000 | 153.06 | 204.46 | 6 | 29.3 | 29 | 0.2 |

## Filter `country_finland` (selectivity 40.5%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 3.04 | 4.73 | 304 | 29.3 | 29 | 0.6 |
| faiss_hnsw | {'ef_search': 16} | 0.760 | 0.34 | 0.98 | 2238 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 64} | 0.980 | 0.58 | 1.51 | 1349 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 256} | 1.000 | 1.51 | 2.13 | 633 | 34.5 | 34 | 5.7 |
| faiss_ivf | {'nprobe': 1} | 0.148 | 0.15 | 0.19 | 6180 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 8} | 0.677 | 0.26 | 0.31 | 3591 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 32} | 0.999 | 0.30 | 0.45 | 2989 | 30.3 | 167 | 1.4 |
| numpy_exact | {} | 1.000 | 157.20 | 380.28 | 5 | 29.3 | 29 | 0.2 |

## Filter `no_milk_low_sugar` (selectivity 41.3%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 3.42 | 5.54 | 270 | 29.3 | 29 | 0.6 |
| faiss_hnsw | {'ef_search': 16} | 0.521 | 0.36 | 0.75 | 2289 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 64} | 0.656 | 0.78 | 1.82 | 1023 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 256} | 0.705 | 1.46 | 2.28 | 633 | 34.5 | 34 | 5.7 |
| faiss_ivf | {'nprobe': 1} | 0.088 | 0.17 | 0.20 | 5574 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 8} | 0.450 | 0.21 | 0.33 | 4389 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 32} | 0.685 | 0.28 | 0.45 | 3414 | 30.3 | 167 | 1.4 |
| numpy_exact | {} | 1.000 | 148.43 | 931.98 | 3 | 29.3 | 29 | 0.2 |

## Filter `vegan_finland` (selectivity 25.4%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 2.38 | 4.62 | 372 | 29.3 | 29 | 0.6 |
| faiss_hnsw | {'ef_search': 16} | 0.553 | 0.59 | 1.71 | 1283 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 64} | 0.727 | 0.68 | 1.42 | 1279 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 256} | 0.762 | 1.84 | 4.42 | 431 | 34.5 | 34 | 5.7 |
| faiss_ivf | {'nprobe': 1} | 0.106 | 0.14 | 0.22 | 6053 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 8} | 0.498 | 0.23 | 0.41 | 3848 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 32} | 0.754 | 0.30 | 0.62 | 3152 | 30.3 | 167 | 1.4 |
| numpy_exact | {} | 1.000 | 172.84 | 581.17 | 4 | 29.3 | 29 | 0.2 |

## Filter `rare_halal_vegan_low_sugar` (selectivity 1.5%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 0.22 | 0.54 | 3597 | 29.3 | 29 | 0.6 |
| faiss_hnsw | {'ef_search': 16} | 0.259 | 0.39 | 1.22 | 1686 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 64} | 0.402 | 0.76 | 1.97 | 1042 | 34.5 | 34 | 5.7 |
| faiss_hnsw | {'ef_search': 256} | 0.433 | 1.41 | 2.22 | 636 | 34.5 | 34 | 5.7 |
| faiss_ivf | {'nprobe': 1} | 0.035 | 0.17 | 0.44 | 4065 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 8} | 0.221 | 0.17 | 0.20 | 5308 | 30.3 | 167 | 1.4 |
| faiss_ivf | {'nprobe': 32} | 0.434 | 0.17 | 0.43 | 4644 | 30.3 | 167 | 1.4 |
| numpy_exact | {} | 1.000 | 120.55 | 176.25 | 8 | 29.3 | 29 | 0.2 |
