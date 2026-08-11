import argparse
from pathlib import Path

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 GDC sample sheet 的列名统一成小写下划线格式。
    例如：
    File ID -> file_id
    File Name -> file_name
    Project ID -> project_id
    Case ID -> case_id
    """
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace(".", "_")
        for c in df.columns
    ]
    return df


def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find any column from {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def collect_local_svs(wsi_dir: Path):
    """
    GDC 默认下载结构：
    wsi/
      <file_uuid>/
        xxx.svs
        annotations.txt
        logs/
    """
    svs_paths = list(wsi_dir.glob("*/*.svs"))

    by_filename = {}
    by_uuid = {}

    for p in svs_paths:
        by_filename[p.name] = p
        by_uuid[p.parent.name] = p

    return by_filename, by_uuid


def build_csv(dataset: str, sample_sheet: Path, wsi_dir: Path, out_csv: Path):
    if dataset == "rcc":
        label_map = {
            "TCGA-KIRC": "CCRCC",
            "TCGA-KIRP": "PRCC",
            "TCGA-KICH": "CRCC",
        }
    elif dataset == "lung":
        label_map = {
            "TCGA-LUAD": "LUAD",
            "TCGA-LUSC": "LUSC",
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    df = pd.read_csv(sample_sheet, sep="\t", dtype=str)
    df = normalize_columns(df)

    file_id_col = find_col(df, ["file_id", "id"])
    file_name_col = find_col(df, ["file_name", "filename"])
    project_id_col = find_col(df, ["project_id", "cases_project_project_id"])
    case_id_col = find_col(df, ["case_id", "case_submitter_id", "cases_submitter_id"])

    by_filename, by_uuid = collect_local_svs(wsi_dir)

    rows = []
    missing_local = []
    unknown_project = []

    for _, r in df.iterrows():
        file_id = str(r[file_id_col]).strip()
        file_name = str(r[file_name_col]).strip()
        project_id = str(r[project_id_col]).strip()
        case_id = str(r[case_id_col]).strip()

        if project_id not in label_map:
            unknown_project.append((file_id, file_name, project_id))
            continue

        # 优先用文件名匹配；如果文件名不匹配，再用 UUID 目录匹配
        svs_path = by_filename.get(file_name)
        if svs_path is None:
            svs_path = by_uuid.get(file_id)

        if svs_path is None:
            missing_local.append((file_id, file_name))
            continue

        real_file_name = svs_path.name
        if not real_file_name.endswith(".svs"):
            continue

        slide_id = real_file_name[:-4]

        rows.append(
            {
                "case_id": case_id,
                "slide_id": slide_id,
                "label": label_map[project_id],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.drop_duplicates()
    out_df = out_df.sort_values(["label", "case_id", "slide_id"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print("=" * 80)
    print(f"Saved: {out_csv}")
    print(f"Rows: {len(out_df)}")
    print()
    print("Label counts:")
    print(out_df["label"].value_counts())
    print()
    print(f"Local SVS count: {len(by_filename)}")
    print(f"Sample sheet rows: {len(df)}")
    print(f"Missing local files: {len(missing_local)}")
    print(f"Unknown project rows: {len(unknown_project)}")

    if missing_local:
        miss_path = out_csv.with_suffix(".missing_local.txt")
        with open(miss_path, "w") as f:
            for file_id, file_name in missing_local:
                f.write(f"{file_id}\t{file_name}\n")
        print(f"Missing local list saved to: {miss_path}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["rcc", "lung"])
    parser.add_argument("--sample_sheet", required=True)
    parser.add_argument("--wsi_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    build_csv(
        dataset=args.dataset,
        sample_sheet=Path(args.sample_sheet),
        wsi_dir=Path(args.wsi_dir),
        out_csv=Path(args.out_csv),
    )


if __name__ == "__main__":
    main()