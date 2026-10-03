.. _parameters:

Parameters
******************************************************************

The supported bring-up uses two YAML files:

1. Core parameters (map resolution/size, kernels, services, toggles)
2. A robot/setup file (subscribers + publishers)


Core Parameters
==============================================================

.. include:: ../../../elevation_mapping_cupy/config/core/core_param.yaml
  :code: yaml


Golden-Path Setup (Synthetic Demo)
================================================

.. include:: ../../../elevation_mapping_cupy/config/setups/synthetic/synthetic_depth.yaml
  :code: yaml


Example Robot Setup (Menzi)
================================================

.. include:: ../../../elevation_mapping_cupy/config/setups/menzi/base.yaml
  :code: yaml


Plugin Configuration
================================================

.. include:: ../../../elevation_mapping_cupy/config/core/plugin_config.yaml
  :code: yaml

