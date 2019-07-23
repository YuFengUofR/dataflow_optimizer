# Dataflow optimizer

This is a DNN dataflow optimizer to find a optimal or `approximately` optimal
dataflow for a particular DNN with a defined hardware configuration.

## Why?

The goal of this optimizer are followings:

* First, this optimizer is to aim to find a close-to optimal configuration in
  order to minimize the latency and reduce the data traffic at the same time.
* Second, this optimizer explores different searching or optimization schemes,
  so that, we can show the trade-off between different optimization schemes.
* Third, this optimizer can automatically decompose deconvolutions in the
  DNN pipeline and apply different level optimization to deconvolutions

## What's inside?

There are two main parts in this framework:
  1. The overall framework to drive different optimizers.
    * `dnn_optimizer.py`

  2. The layer-level optimizers to optimize different
    * `layer_base_method.py`
    * `layer_optimizer.py`
  * `dataflow_search.py`
  * `deconv_exhaustive_searcher.py`
  * `deconv_solver.py`               dnn_optimizer_dp.py    layer3d_static.py     layer_exhaustive_searcher.py  runner.sh

