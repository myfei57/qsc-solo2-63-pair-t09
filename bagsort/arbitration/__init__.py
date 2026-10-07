"""Barcode conflict arbitration.

Import the submodules directly (``bagsort.arbitration.arbitrator`` etc.);
keeping this initializer light avoids an import cycle through the materialised
state that owns the arbitration records.
"""

from __future__ import annotations
