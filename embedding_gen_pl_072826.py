# Setup and Config - Sets CUDA device, batch size, and file paths for ICD mappings, demographics, and event data.
# My input is the patient event file (either CSV subsets or full parquet), and outputs go to a versioned embedding directory.

# %%
# embedding generation script
# modified
#!/usr/bin/env python
import os
import pandas as pd
import polars as pl
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# %%
# cuda
cuda_num = 2
print(f"cuda:{str(cuda_num)}") 

# %%
# batch size
batch_size_ = 2048  # 1024

# change variables
icd_file = "icd_mapping.csv"
output_fname = "patient_embedding_metadata.csv"
demographic_file = 'CKD-supplemental_pull.rpt'

custom_separator = True  # <<
if not custom_separator:
    subset_size = "25"  # 10, 100, all/full # <<
    output_dir = f"outputs"
    event_file = f"path/file.csv"
if custom_separator:
    output_dir = "outputs"
    event_file = "path/to/event/data"

# ICD Code Mapping - Loads ICD-10 codes and their text descriptions, creates a dictionary mapping codes to human-readable diagnoses (e.g., "N18.3" → "Chronic kidney disease, stage 3").

# ckd script
output_dir += "_icd"  # <<

# filter ckd stage
filter_ckd_stage = True  # <<
if filter_ckd_stage:
    output_dir += "_stage_filter"

print(output_dir)

# %%
try:
    os.mkdir(output_dir)
except FileExistsError:
    pass

# %%
# DEBUG
csv = event_file
icd = icd_file
output_dir = output_dir
model_name = (
    "path/huggingface/"
    "models--qwen--qwen3-embedding-8b/"
    "snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
)
embed_dim = 4096
batch_size = batch_size_

os.makedirs(output_dir, exist_ok=True)

# %%
icd_df = pd.read_csv(icd)
icd_df["icd_code"] = icd_df["icd_code"].astype(str).str.replace(".", "", regex=False)
icd_map = dict(zip(icd_df["icd_code"], icd_df["long_title"]))

# CKD Stage Filtering (lines 47-62)
# Extracts CKD stage from ICD codes (N18.1-N18.9), maps them to numeric stages, filters patients to only those who reached stage 3+, and keeps only those patient IDs for downstream processing.

# %%
df = pl.read_parquet(csv)

# %%
custom_map = {
    1: 1,  # "1",
    2: 2,  # "2",
    3: 3,  # "3",
    4: 4,  # "4",
    5: 5,  # "5",
    6: 6,  # "ESRD",
    9: 0,  # "CKD"
}
print("Building Filter")
ckd_icd_df = (
    df.filter(pl.col("DataCategory").str.contains("N18"))
    .with_columns(
        pl.col("DataCategory")
        .str.extract(r"N18\.([1-9])", 1)
        .cast(pl.Int64)
        .replace(custom_map, default=None)
        .alias("CKD_stage_numeric")
    )
    .select(
        [
            pl.col("PatientID"),
            pl.col("EventTimeStamp"),
            pl.col("META_1"),
            pl.col("CKD_stage_numeric"),
        ]
    )
    .with_columns(
        pl.col("CKD_stage_numeric").max().over("PatientID").alias("max_stage")
    )
    .filter(pl.col("max_stage") >= 3)
    .unique()
)
print("Filtering and converting")

# Date Processing - Converts timestamps to dates, handles null categories by filling with META_2 field.

df = (
    df.join(
        ckd_icd_df.select("PatientID").unique(),
        on="PatientID",
        how="inner",
    )
    .drop_nulls(subset=["DataNumeric"])
    .with_columns(
        [
            pl.col("EventTimeStamp").str.strptime(pl.Datetime("us")),
            pl.col("DataCategory").fill_null(pl.col("META_2")),
        ]
    )
    .with_columns(pl.col("EventTimeStamp").dt.date().alias("EventDate"))
)

# %% demographics pseudonote and encounter - Demographics Pseudonote
# Creates patient intro sentences like "Patient [ID] born [DOB] is a [race] [sex]" and builds a lookup dictionary.

demographic_df = (
    pl.read_csv(demographic_file, separator="$")
    .with_columns(
        pl.format(
            "Patient born {} is a {} {}.",
            pl.col("DOB"),
            pl.col("EthnoRacialCategory")
            .replace("No Usable Values", "Unknown Race")
            .replace("Multiple Ethnoracial Categories", "Multiracial"),
            pl.col("Sex"),
        ).alias("sentence")
    )
)


# demographic_df = (
#     df.filter(pl.col("DataType") == "Demographics")
#     .drop_nulls(subset="DataCategory")
#     .group_by("PatientID")
#     .first()
#     .with_columns(
#         pl.format(
#             "Patient {} is a {} patient.",
#             pl.col("PatientID"),
#             pl.col("DataCategory")
#             .str.replace("//", " ")
#             .str.replace("/", " or ")
#             .str.replace("Do not identify with Race", "unknown race")
#             .str.replace("Unknown Not Reported", "")
#             .str.strip_chars(),
#         ).alias("sentence")
#     )
# )


demographic_map = dict(
    zip(demographic_df["PatientID"], demographic_df["sentence"])
)

## Clinical Events to Sentences (lines 92-120)
# Converts structured data into natural language:

# Diagnosis → "ICD-10 code N18.3: Chronic kidney disease, stage 3"
# Medications → "Medication administered: [drug name]"
# Labs → "[test name]: [value]"
# Procedures → "Procedure performed: [procedure]"

# %% Constructing the clauses to append to sentences
sentences_df = (
    df.filter(~pl.col("DataType").is_in(["Demographics", "Encounter"]))
    .with_columns(
        pl.when(pl.col("DataType") == "Diagnosis")
        .then(
            pl.col("DataCategory")
            .str.replace(".", "", literal=True)
            .replace(icd_map, default="Unknown")
        )
        .otherwise(pl.format("NA"))
        .alias("long_title"),
        pl.when(
            (pl.col("DataType") == "Encounter")
            & (pl.col("DataCategory").str.contains("//"))
        )
        .then(pl.col("DataCategory").str.replace("//", ": "))
        .otherwise(pl.format("NA"))
        .alias("EncounterType"),
    )
    .filter(pl.col("long_title") != "Unknown")  # mostly ICD-9
    .with_columns(
        pl.when(pl.col("DataType") == "Diagnosis")
        .then(
            # pl.format(" - ICD-10 code {}", pl.col("DataCategory"))
            pl.format(
                " - ICD-10 code {}: {}", pl.col("DataCategory"), pl.col("long_title")
            )
        )
        .when(pl.col("DataType") == "Medications")
        .then(pl.format(" - Medication administered: {}", pl.col("DataCategory")))
        .when(pl.col("DataType") == "Procedure")
        .then(pl.format(" - Procedure performed: {}", pl.col("DataCategory")))
        .when(pl.col("DataType") == "Labs")
        .then(pl.format(" - {}: {}", pl.col("DataCategory"), pl.col("DataNumeric")))
        .when(pl.col("DataType") == "Encounter")
        .then(
            pl.when(pl.col("EncounterType") != "NA")
            .then(pl.format(" - {}", pl.col("DataCategory")))
            .otherwise(
                pl.format(
                    " - {}: {}", pl.col("DataCategory"), pl.col("DataNumeric")
                )
            )
        )
        .otherwise(pl.format("Not parsed."))
        .alias("sentence")
    )
    .filter(
        pl.col("META_2").is_null()
        | (pl.col("META_2") != "MedicalHistDx")
    )
    .drop(pl.col("META_2"))
    .unique()
)

## Daily Aggregation - Groups all events within the same day into "On [date], the patient had the following records: [all events]"

# %%
# day_grouped_df = (
#     sentences_df.group_by("PatientID", "META_1", "EventDate")
#     .agg(
#         [
#             pl.col("sentence").str.concat(" ").alias("day_summary"),
#             pl.col("EventDate").first().alias("date"),
#         ]
#     )
#     .with_columns(
#         pl.format(
#             "On {}, the patient had the following records: {}",
#             pl.col("date"),
#             pl.col("day_summary"),
#         ).alias("day_summary")
#     )
# )

day_grouped_df = (
    sentences_df.group_by("PatientID", "META_1", "EventDate")
    .agg(
        [
            pl.when(pl.col("DataType") == "Diagnosis")
            .then(
                pl.format(
                    "ICD-10 code {}: {}",
                    pl.col("DataCategory"),
                    pl.col("long_title"),
                )
            )
            .otherwise(None)
            .drop_nulls()
            .str.concat(", ")
            .alias("diagnoses"),

            pl.when(pl.col("DataType") == "Medications")
            .then(pl.col("DataCategory"))
            .otherwise(None)
            .drop_nulls()
            .str.concat(", ")
            .alias("medications"),

            pl.when(pl.col("DataType") == "Procedure")
            .then(pl.col("DataCategory"))
            .otherwise(None)
            .drop_nulls()
            .str.concat(", ")
            .alias("procedures"),

            pl.when(pl.col("DataType") == "Labs")
            .then(
                pl.format(
                    "{}: {}",
                    pl.col("DataCategory"),
                    pl.col("DataNumeric"),
                )
            )
            .otherwise(None)
            .drop_nulls()
            .str.concat(", ")
            .alias("labs"),

            # Preserve any other record types
            pl.when(
                ~pl.col("DataType").is_in(
                    ["Diagnosis", "Medications", "Procedure", "Labs"]
                )
            )
            .then(pl.col("sentence"))
            .otherwise(None)
            .drop_nulls()
            .str.concat(" ")
            .alias("other_records"),

            pl.col("EventDate").first().alias("date"),
        ]
    )
    .with_columns(
        [
            pl.when(
                pl.col("diagnoses").is_not_null()
                & (pl.col("diagnoses") != "")
            )
            .then(pl.format(" - Diagnoses: {}", pl.col("diagnoses")))
            .otherwise(pl.lit(""))
            .alias("diagnosis_record"),

            pl.when(
                pl.col("medications").is_not_null()
                & (pl.col("medications") != "")
            )
            .then(
                pl.format(
                    " - Medications administered: {}",
                    pl.col("medications"),
                )
            )
            .otherwise(pl.lit(""))
            .alias("medication_record"),

            pl.when(
                pl.col("procedures").is_not_null()
                & (pl.col("procedures") != "")
            )
            .then(
                pl.format(
                    " - Procedures performed: {}",
                    pl.col("procedures"),
                )
            )
            .otherwise(pl.lit(""))
            .alias("procedure_record"),

            pl.when(
                pl.col("labs").is_not_null()
                & (pl.col("labs") != "")
            )
            .then(pl.format(" - Labs: {}", pl.col("labs")))
            .otherwise(pl.lit(""))
            .alias("lab_record"),

            pl.col("other_records").fill_null(""),
        ]
    )
    .with_columns(
        pl.concat_str(
            [
                "diagnosis_record",
                "medication_record",
                "procedure_record",
                "lab_record",
                "other_records",
            ],
            separator="",
        ).alias("day_summary")
    )
    .with_columns(
        pl.format(
            "On {}, the patient had the following records: {}",
            pl.col("date"),
            pl.col("day_summary"),
        ).alias("day_summary")
    )
    .drop(
        [
            "diagnoses",
            "medications",
            "procedures",
            "labs",
            "other_records",
            "diagnosis_record",
            "medication_record",
            "procedure_record",
            "lab_record",
        ]
    )
)

## Encounter-Level Pseudonotes - Aggregates daily summaries within each encounter (META_1), prepends demographic info, creates final text

# %%
enc_grouped_df = (
    day_grouped_df.group_by("PatientID", "META_1")
    .agg(
        [
            pl.col("day_summary").str.concat("\n").alias("enc_summary"),
            pl.col("EventDate").min().alias("date"),
            pl.col("META_1").first().alias("enc_id"),
            pl.col("PatientID")
            .first()
            .replace(demographic_map, default="Demographics not available.")
            .alias("demographic_string"),
        ]
    )
    .with_columns(
        pl.format(
            "Encounter starting {}. {} {}",
            pl.col("date"),
            pl.col("demographic_string"),
            pl.col("enc_summary"),
        ).alias("enc_summary")
    )
    .drop("demographic_string")
    .sort("PatientID", "date")
    .with_columns(pl.int_range(pl.len()).over("PatientID").alias("emb_id"))
)

enc_grouped_df.head()  # this is the final dataframe which we can treat as the meta.

## Metadata Export - Saves encounter metadata with CKD staging info to CSV files (one with pseudonotes, one clean).

# %%
# just do a for loop through all of them... should be fine...
enc_grouped_df = enc_grouped_df.join(
    ckd_icd_df.select(["PatientID", "META_1", "CKD_stage_numeric", "max_stage"]),
    on=["PatientID", "META_1"],
    how="left",
)
enc_grouped_df.write_csv(
    os.path.join(output_dir, "meta_v3_all.csv"), separator="$"
)
print("Saved meta df")

# %%
enc_grouped_df.drop("enc_summary").write_csv(
    os.path.join(output_dir, "meta_v3.csv"), separator="$"
)
print("Saved meta df clean")

## Embedding Generation - Loads Qwen3-Embedding-8B locally, tokenizes pseudonotes (32,768 token max), applies last-token pooling and L2 normalization, and saves 4,096-dimensional numpy arrays organized by patient ID folders.

# %% generating embeddings
print(f"[INFO] Loading model from: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
    padding_side="left",
)
model = AutoModel.from_pretrained(
    model_name,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
)
device = f"cuda:{cuda_num}"
model.to(device)
model.eval()

# %%
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def get_qwen_embeddings(texts, tokenizer, model, device, embed_dim):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32_768,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = last_token_pool(
            outputs.last_hidden_state, inputs["attention_mask"]
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    if embeddings.size(1) > embed_dim:
        embeddings = embeddings[:, :embed_dim]
    else:
        pad = embed_dim - embeddings.size(1)
        embeddings = torch.nn.functional.pad(embeddings, (0, pad), value=0)
    return embeddings[0].float().cpu().numpy()


patient_grouped_df = (
    enc_grouped_df.sort(["PatientID", "emb_id"])
    .group_by("PatientID")
    .agg(pl.col("enc_summary").alias("pseudonotes"))
)
grouped_dict = dict(
    zip(patient_grouped_df["PatientID"], patient_grouped_df["pseudonotes"])
)

truncated_notes = []
print("Generating embeddings")
for pid, texts in tqdm(grouped_dict.items()):
    patient_folder = os.path.join(output_dir, str(pid))
    os.makedirs(patient_folder, exist_ok=True)

    embeddings = []

    for emb_id, t in enumerate(texts):
        original_token_count = len(
            tokenizer(t, truncation=False)["input_ids"]
        )

        if original_token_count > 32_768:
            truncated_notes.append({
                "PatientID": pid,
                "emb_id": emb_id,
                "original_token_count": original_token_count,
            })

            print(f"Truncated so far: {len(truncated_notes)}")

        embeddings.append(
            get_qwen_embeddings(t, tokenizer, model, device, embed_dim)
        )


    fname = f"{pid}"
    fpath = os.path.join(patient_folder, fname)
    np.save(fpath, np.array(embeddings))

print(f"Number of truncated pseudo-notes: {len(truncated_notes)}")

if truncated_notes:
    pd.DataFrame(truncated_notes).to_csv(
        os.path.join(output_dir, "truncated_pseudonotes.csv"),
        index=False,
    )
