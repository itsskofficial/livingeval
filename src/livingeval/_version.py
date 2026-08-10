__version__ = "0.1.0"
#: Bumped whenever the on-disk result-record shape changes in a way that older
#: readers cannot handle. Records carry it; `report.load` refuses a future schema
#: rather than misreading it.
SCHEMA_VERSION = 1
