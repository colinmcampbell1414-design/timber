"""Registry of the Hazards & Applications modules.

The dropdown on the page is built from APPS, in this order. TIMBER focuses on
tree fall hazard; slope stability is kept as a second check of the ground.
The other Phase 5 apps (erosion, drainage, irrigation, rain garden, ice, trip
hazards, volume) were removed and are kept in archive/apps/ for reference.

To add an application: write terrain/apps/<name>.py with the shared interface
(see base.py) and add it to this list.
"""

from terrain.apps import mudslide, trees

APPS = [
    trees,
    mudslide,
]

REQUIRED_ATTRIBUTES = ("NAME", "DESCRIPTION", "REQUIRED_INPUTS", "SHAPES", "TEST_AT_HOME", "params", "run")
