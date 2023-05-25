import json
import pandas as pd

# Load mapping source data
f = open("data/capec-stride-mapping.json", "r")
mapping = json.load(f)

stride_terms = mapping["CAPEC S.T.R.I.D.E. Mapping"]["children"]

capec_stride_mapping = dict()


def capec_stride_mapping_builder(cat, children, depth):
    if len(children) != 0:
        for elem in children:
            print(f"Depth: {depth}" + "\t" * depth + f" {elem}")

            ref = elem.lstrip().split(":")[0]
            if ref not in capec_stride_mapping:
                capec_stride_mapping[ref] = set()
            capec_stride_mapping[ref].add(cat)

            depth += 1
            capec_stride_mapping_builder(cat, children[elem]["children"], depth)
            depth -= 1


for category in stride_terms:
    capec_stride_mapping_builder(category, stride_terms[category]["children"], 1)

# Load CAPEC attack patterns from original source
# Note: CSV has been converted to XLSX and sheet name has been renamed to "Threats"
df_capec = pd.read_excel("data/capec_src.xlsx", sheet_name="Threats")

refs = df_capec["Ref"].tolist()

for capec_ref, capec_stride_set in capec_stride_mapping.items():
    if capec_ref in refs:
        # Set all STRIDE values to 0
        for stride_term in stride_terms:
            df_capec.loc[df_capec["Ref"] == capec_ref, stride_term[0]] = 0

        # Set STRIDE values to 1 if found
        for stride_term in capec_stride_set:
            df_capec.loc[df_capec["Ref"] == capec_ref, stride_term[0]] = 1

    else:
        # There should be 3 cases: CAPEC-629 which is deprecated and two more that don't have a CAPEC ID
        print(f"{capec_ref} not found in Excel")

# Move rows without mapping to another dataframe
df_no_mapping_found = df_capec.loc[pd.isnull(df_capec.S), :]
df_capec.drop(df_no_mapping_found.index, inplace=True)

# Write results to new file
print("Nº of elements found:")
print(len(df_capec))
print("Nº of elements without mapping:")
print(len(df_no_mapping_found))

with pd.ExcelWriter("data/raw_capec_data.xlsx", engine="openpyxl") as writer:
    df_capec.to_excel(writer, sheet_name="Threats")
    df_no_mapping_found.to_excel(writer, sheet_name="NoMappingFound")
