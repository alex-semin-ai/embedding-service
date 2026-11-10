# Index benchmark in Docker with a Qdrant server (recall@10 vs latency and memory)

Produced by `docker compose --profile demo up`. Qdrant latency includes the HTTP round trip.
Qdrant memory is the server's resident memory growth while building the collection.

Dataset size n=5000. Latency is per single query, one thread.

## Filter `none` (selectivity 100.0%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 0.97 | 1.75 | 893 | 7.3 | 8 | 0.1 |
| faiss_hnsw | {'ef_search': 16} | 0.939 | 0.14 | 0.23 | 6440 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 64} | 1.000 | 0.33 | 0.75 | 2408 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 256} | 1.000 | 0.65 | 0.99 | 1437 | 8.6 | 10 | 3.4 |
| faiss_ivf | {'nprobe': 1} | 0.257 | 0.09 | 0.21 | 8414 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 8} | 0.915 | 0.17 | 0.47 | 4361 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 32} | 1.000 | 0.19 | 0.27 | 4961 | 7.8 | 17 | 0.2 |
| hnswlib | {'ef_search': 16} | 0.928 | 0.18 | 0.43 | 4538 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 64} | 0.999 | 0.28 | 0.44 | 3236 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 256} | 1.000 | 0.59 | 0.82 | 1608 | 9.2 | 12 | 0.7 |
| numpy_exact | {} | 1.000 | 19.32 | 50.15 | 42 | 7.3 | 7 | 0.3 |
| qdrant | {'hnsw_ef': 16} | 1.000 | 7.27 | 9.14 | 132 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 64} | 1.000 | 8.05 | 22.62 | 84 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 256} | 1.000 | 8.47 | 17.01 | 98 | n/a | 192 | 11.5 |

## Filter `country_finland` (selectivity 39.9%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 0.79 | 1.29 | 1139 | 7.3 | 8 | 0.1 |
| faiss_hnsw | {'ef_search': 16} | 0.931 | 0.18 | 0.28 | 4980 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 64} | 0.999 | 0.40 | 1.08 | 1992 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 256} | 1.000 | 0.78 | 1.15 | 1197 | 8.6 | 10 | 3.4 |
| faiss_ivf | {'nprobe': 1} | 0.229 | 0.11 | 0.18 | 8047 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 32} | 1.000 | 0.20 | 0.31 | 4604 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 8} | 0.901 | 0.21 | 0.54 | 3834 | 7.8 | 17 | 0.2 |
| hnswlib | {'ef_search': 16} | 0.994 | 0.30 | 0.55 | 2938 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 64} | 1.000 | 0.81 | 2.34 | 905 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 256} | 1.000 | 2.65 | 3.86 | 361 | 9.2 | 12 | 0.7 |
| numpy_exact | {} | 1.000 | 12.80 | 21.76 | 71 | 7.3 | 7 | 0.3 |
| qdrant | {'hnsw_ef': 64} | 1.000 | 6.22 | 8.53 | 153 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 16} | 1.000 | 7.23 | 8.78 | 135 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 256} | 1.000 | 7.92 | 15.38 | 104 | n/a | 192 | 11.5 |

## Filter `no_milk_low_sugar` (selectivity 41.4%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 0.89 | 1.40 | 1019 | 7.3 | 8 | 0.1 |
| faiss_hnsw | {'ef_search': 16} | 0.650 | 0.25 | 0.61 | 2350 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 64} | 0.806 | 0.45 | 0.76 | 1970 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 256} | 0.887 | 0.71 | 1.10 | 1329 | 8.6 | 10 | 3.4 |
| faiss_ivf | {'nprobe': 1} | 0.188 | 0.16 | 0.48 | 4348 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 32} | 0.840 | 0.17 | 0.26 | 5380 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 8} | 0.595 | 0.19 | 0.45 | 4330 | 7.8 | 17 | 0.2 |
| hnswlib | {'ef_search': 16} | 0.918 | 0.54 | 1.55 | 1365 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 64} | 0.978 | 1.02 | 2.11 | 885 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 256} | 0.999 | 2.88 | 4.26 | 373 | 9.2 | 12 | 0.7 |
| numpy_exact | {} | 1.000 | 12.11 | 27.63 | 65 | 7.3 | 7 | 0.3 |
| qdrant | {'hnsw_ef': 64} | 1.000 | 6.44 | 8.61 | 149 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 16} | 1.000 | 6.72 | 8.73 | 143 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 256} | 1.000 | 7.88 | 14.06 | 112 | n/a | 192 | 11.5 |

## Filter `vegan_finland` (selectivity 25.0%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 0.54 | 0.94 | 1689 | 7.3 | 8 | 0.1 |
| faiss_hnsw | {'ef_search': 16} | 0.746 | 0.19 | 0.29 | 4384 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 64} | 0.861 | 0.36 | 0.79 | 2291 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 256} | 0.911 | 0.85 | 1.39 | 1068 | 8.6 | 10 | 3.4 |
| faiss_ivf | {'nprobe': 1} | 0.199 | 0.15 | 0.51 | 4672 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 8} | 0.694 | 0.16 | 0.27 | 5658 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 32} | 0.882 | 0.16 | 0.25 | 5496 | 7.8 | 17 | 0.2 |
| hnswlib | {'ef_search': 16} | 0.975 | 0.40 | 1.54 | 1570 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 64} | 0.996 | 1.03 | 2.68 | 704 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 256} | 1.000 | 3.49 | 6.87 | 254 | 9.2 | 12 | 0.7 |
| numpy_exact | {} | 1.000 | 13.20 | 30.36 | 63 | 7.3 | 7 | 0.3 |
| qdrant | {'hnsw_ef': 16} | 1.000 | 6.28 | 7.13 | 156 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 64} | 1.000 | 6.32 | 8.80 | 147 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 256} | 1.000 | 7.32 | 12.01 | 121 | n/a | 192 | 11.5 |

## Filter `rare_halal_vegan_low_sugar` (selectivity 1.2%)

| index | search params | recall@10 | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |
|---|---|---|---|---|---|---|---|---|
| faiss_flat | {} | 1.000 | 0.13 | 0.25 | 6533 | 7.3 | 8 | 0.1 |
| faiss_hnsw | {'ef_search': 16} | 0.259 | 0.20 | 0.45 | 3967 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 64} | 0.406 | 0.32 | 0.58 | 2874 | 8.6 | 10 | 3.4 |
| faiss_hnsw | {'ef_search': 256} | 0.604 | 0.75 | 1.24 | 1196 | 8.6 | 10 | 3.4 |
| faiss_ivf | {'nprobe': 32} | 0.457 | 0.12 | 0.20 | 7062 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 8} | 0.238 | 0.13 | 0.22 | 6556 | 7.8 | 17 | 0.2 |
| faiss_ivf | {'nprobe': 1} | 0.060 | 0.16 | 0.48 | 4386 | 7.8 | 17 | 0.2 |
| hnswlib | {'ef_search': 16} | 0.990 | 4.56 | 7.46 | 220 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 64} | 1.000 | 7.97 | 12.72 | 114 | 9.2 | 12 | 0.7 |
| hnswlib | {'ef_search': 256} | 1.000 | 8.38 | 10.79 | 116 | 9.2 | 12 | 0.7 |
| numpy_exact | {} | 1.000 | 10.87 | 19.20 | 82 | 7.3 | 7 | 0.3 |
| qdrant | {'hnsw_ef': 256} | 1.000 | 6.05 | 7.38 | 161 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 64} | 1.000 | 6.06 | 10.60 | 149 | n/a | 192 | 11.5 |
| qdrant | {'hnsw_ef': 16} | 1.000 | 6.54 | 13.40 | 126 | n/a | 192 | 11.5 |
