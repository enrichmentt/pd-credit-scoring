from pathlib import Path

import pandas as pd
import pytest

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "sample_raw.csv"


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_RAW)
