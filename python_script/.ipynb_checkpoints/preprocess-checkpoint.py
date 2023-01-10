def str_to_label(df, label_cols):
    labels = list(set(df['label'].to_list()))
    label_to_idx = {label:i for i,label in enumerate(labels)}
    df['label'] = df['label'].apply(lambda x: label_to_idx[x])
    return df

