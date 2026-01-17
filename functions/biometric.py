def aadhar_biometric_fraud_detection(
    files_list,
    show_plots=True,
    save_outputs=False,
    random_state=42
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    df = pd.concat(
        [pd.read_csv(f) for f in files_list],
        ignore_index=True
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=True
    )

    df = df.dropna(subset=["state", "pincode", "district"])

    state_features = (
        df.groupby("state")
        .agg(
            total_records=("date", "count"),
            active_days=("date", "nunique"),
            unique_pincodes=("pincode", "nunique"),
            unique_districts=("district", "nunique")
        )
        .reset_index()
    )

    state_features["records_per_day"] = (
        state_features["total_records"] /
        state_features["active_days"].clip(lower=1)
    )

    state_features["districts_per_pincode"] = (
        state_features["unique_districts"] /
        state_features["unique_pincodes"].clip(lower=1)
    )

    anomaly_base = state_features[
        ["records_per_day", "districts_per_pincode"]
    ]

    q1 = anomaly_base.quantile(0.25)
    q3 = anomaly_base.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr

    outlier_mask = (anomaly_base > upper).any(axis=1)
    contamination = float(
        np.clip(outlier_mask.mean(), 0.01, 0.2)
    )

    print(f"Auto contamination used: {contamination:.3f}")

    feature_cols = [
        "records_per_day",
        "districts_per_pincode",
        "unique_pincodes",
        "active_days"
    ]

    X = state_features[feature_cols].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state
    )

    state_features["anomaly"] = iso.fit_predict(X_scaled)
    state_features["fraud_score"] = -iso.decision_function(X_scaled)

    fraud_states = (
        state_features[state_features["anomaly"] == -1]
        .sort_values("fraud_score", ascending=False)
    )

    if save_outputs:
        state_features.to_csv(
            "state_fraud_features.csv",
            index=False
        )
        fraud_states.to_csv(
            "fraud_prone_states.csv",
            index=False
        )
        with open("contamination_used.txt", "w") as f:
            f.write(str(contamination))

    if show_plots:
        plt.figure(figsize=(6,4))
        plt.hist(state_features["fraud_score"], bins=30)
        plt.title("Fraud Score Distribution (Isolation Forest)")
        plt.xlabel("Fraud Score")
        plt.ylabel("States")
        plt.tight_layout()
        plt.show()

    return {
        "combined_df": df,
        "state_features": state_features,
        "fraud_states": fraud_states,
        "contamination": contamination,
        "model": iso
    }
