# Marks `tests` as a package so `from tests.conftest import <helper>` resolves
# under pytest's default (prepend) import mode. Without this file the offline
# suite fails to collect on a fresh clone: test_clustering, test_flags,
# test_saturation, and test_gui_visuals import make_passages from tests.conftest.
