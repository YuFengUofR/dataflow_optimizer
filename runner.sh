#!/bin/bash

python dataflow_search.py --dnnfile dnns/flowNetC.txt \
	--model_type 2D \
	--search_methods Constraint \
	--bufsize 1572864 \
	--bit_width 16 \
	--memory_bandwidth 25.6 \
	--sa_size 16 \
	--model_type 2D \
	--ifmap 960 576 6
