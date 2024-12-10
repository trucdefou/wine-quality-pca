
def check_bool(x):
    if x in [0, 0.0, "false", "no", "No","no", "N", "n"]:
        return False

    if x in [1, 1.0, "true", "yes", "Yes", "si", "Y", "y"]:
        return True

    return x


def convert_bolean_columns(df):
    columns = df.columns.tolist()

    converted_columns = []

    boolean_patterns = [
        [0, 1],
        [0.0, 1.0],
        ["false", "true"],
        ["No", "Yes"],
        ["no", "yes"],
        ["no", "si"],
        ["N", "Y"],
        ["n", "y"],
    ]
    for column in columns:
        unique_values = df[column].dropna().unique()

        is_boolean = False

        if len(unique_values) == 2 and column !="sex":
            for pattern in boolean_patterns:
                if set(unique_values) == set(pattern):
                    df[column] = df[column].apply(check_bool)
                    df[column] = df[column].astype("bool")
                    is_boolean = True
                    break

        if is_boolean:
            converted_columns.append(column)