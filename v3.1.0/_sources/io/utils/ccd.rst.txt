CCD Utilities
=============

CCD loaders accept ``unknown_charge_policy="zero"`` (default) to fill unknown
reference charges with zero, or ``"raise"`` to reject them. ``"infer"`` and
``"keep"`` raise ``NotImplementedError``.

.. automodule:: atomworks.io.utils.ccd
   :members:
   :undoc-members:
   :show-inheritance:
