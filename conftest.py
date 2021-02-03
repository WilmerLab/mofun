
import pytest

pytest.register_assert_rewrite("tests.fixtures")

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runveryslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "veryslow: mark test as VERY slow to run")

def pytest_collection_modifyitems(config, items):
    runveryslow = config.getoption("--runveryslow")
    runslow = config.getoption("--runslow") or runveryslow

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_very_slow = pytest.mark.skip(reason="need --runveryslow option to run")

    for item in items:
        if "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)
        if "veryslow" in item.keywords and not runveryslow:
            item.add_marker(skip_very_slow)
