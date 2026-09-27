MASKER_PROFILES = {
    "Tabular": {"contracts": ["dtype", "shape", "background", "masking"]},
    "Partition": {"contracts": ["background", "clustering", "masking", "shape"]},
    "Image": {"contracts": ["shape", "masking"]},
    "Text": {"contracts": ["tokenization", "masking", "shape"]},
    "Composite": {"contracts": ["delegation", "shape"]},
}
