import pandas as pd
from pathlib import Path
from google.cloud import bigquery
from colorama import Fore, Style

def load_local_data(cache_path: Path) -> pd.DataFrame:
    """
    Load data from a local CSV file.
    """
    try:
        df = pd.read_csv(cache_path)
        print(f"✅ Data loaded from {cache_path}, with shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found at {cache_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)
    client = bigquery.Client()


    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else
    data.columns = [f'_{column}' if not str(column)[0].isalpha() and not str(column)[0] == '_' else str(column) for column in data.columns]

    if truncate:
        write_mode = "WRITE_TRUNCATE"
    else:
        write_mode = "WRITE_APPEND"

    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
