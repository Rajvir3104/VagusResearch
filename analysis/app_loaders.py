import io
import pandas as pd

# ---------- Loaders ----------

# def load_data_old_4col(decoded_text):
#     df = pd.read_csv(
#         io.StringIO(decoded_text),
#         sep=r"\s+",
#         header=None
#     )

#     df.columns = ["time", "ecg", "bp", "vagus", "resp"]

#     return df


def load_data_msna_5col(decoded_text):

    rows = []

    for line in decoded_text.splitlines():

        parts = line.strip().split()

        if len(parts) >= 5:

            try:
                [float(x) for x in parts[:5]]

                rows.append(parts[:5])

            except ValueError:
                pass

    df = pd.DataFrame(
        rows,
        columns=["time", "resp", "ecg", "bp", "nerve"]
    ).astype(float)

    return df


# def load_data_msna_4col(decoded_text):

#     rows = []

#     for line in decoded_text.splitlines():

#         parts = line.strip().split()

#         if len(parts) == 4:

#             try:
#                 [float(x) for x in parts]

#                 rows.append(parts)

#             except ValueError:
#                 pass

#     df = pd.DataFrame(
#         rows,
#         columns=["time", "ecg", "raw_arm", "resp"]
#     ).astype(float)

#     return df

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a NEW df with consistent column names:
      time, ecg, resp, nerve, (optional) bp
    Works with both:
      old: time breath ecg vagus
      msna: time ecg bp resp raw_arm
    """
    cols = set(df.columns)

    # vagus format
    if {"time", "breath", "ecg", "vagus"}.issubset(cols):
        out = df.rename(columns={"breath": "resp", "vagus": "nerve"}).copy()
        return out

    # msna format
    if {"time", "ecg", "raw_arm", "resp"}.issubset(cols):
        out = df.rename(columns={"raw_arm": "nerve"}).copy()
        return out

    raise ValueError(f"Unknown data format. Columns found: {sorted(df.columns)}")

def load_any(decoded_text):

    # try:
    #     df = load_data_old_4col(decoded_text)

    #     df = canonicalize(df)

    #     df = df.dropna().reset_index(drop=True)

    #     return df

    # except Exception:

        # try:
    df = load_data_msna_5col(decoded_text)

    # df = canonicalize(df)

    df = df.dropna().reset_index(drop=True)

    return df

        # except Exception:

            # df = load_data_msna_4col(decoded_text)

            # df = canonicalize(df)

            # df = df.dropna().reset_index(drop=True)

            # return df