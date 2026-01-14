def aadhaar_youth_analysis(
    files_list,
    n_clusters=3,
    show_plots=True
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    plt.rcParams["figure.figsize"] = (12, 6)
    sns.set_style("whitegrid")

    combined_df = pd.concat(
        [pd.read_csv(file) for file in files_list],
        ignore_index=True
    )

    print("Combined Shape:", combined_df.shape)

    combined_df["date"] = pd.to_datetime(
        combined_df["date"], dayfirst=True, errors="coerce"
    )

    combined_df["total_youth"] = (
        combined_df["demo_age_5_17"] + combined_df["demo_age_17_"]
    )

    combined_df["total_population"] = combined_df.filter(
        regex="demo_age"
    ).sum(axis=1)

    combined_df["year_month"] = combined_df["date"].dt.to_period("M")

    monthly_youth = (
        combined_df
        .groupby("year_month")["total_youth"]
        .sum()
        .reset_index()
    )

    monthly_youth["year_month"] = monthly_youth["year_month"].astype(str)
    monthly_youth["mom_growth"] = monthly_youth["total_youth"].pct_change() * 100
    monthly_youth["rolling_3m"] = monthly_youth["total_youth"].rolling(3).mean()

    if show_plots:
        plt.plot(
            monthly_youth["year_month"],
            monthly_youth["total_youth"],
            marker="o"
        )
        plt.title("Monthly Youth Aadhaar Activity Trend")
        plt.xlabel("Month")
        plt.ylabel("Total Youth Count")
        plt.xticks(rotation=45)
        plt.show()

    top_states_youth = (
        combined_df
        .groupby("state")["total_youth"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    if show_plots:
        top_states_youth.plot(kind="bar")
        plt.title("Top 10 States by Youth Aadhaar Activity")
        plt.xlabel("State")
        plt.ylabel("Total Youth Count")
        plt.xticks(rotation=45, ha="right")
        plt.show()

        top5_states = top_states_youth.head(5)

        plt.figure(figsize=(8, 8))
        plt.pie(
            top5_states.values,
            labels=top5_states.index,
            autopct="%1.1f%%",
            startangle=140
        )
        plt.title("Share of Youth Aadhaar Activity – Top 5 States")
        plt.axis("equal")
        plt.show()

        state_monthly = (
            combined_df[combined_df["state"].isin(top5_states.index)]
            .groupby(["state", "year_month"])["total_youth"]
            .sum()
            .unstack()
        )

        state_monthly.columns = state_monthly.columns.astype(str)

        plt.figure(figsize=(14, 6))
        sns.heatmap(state_monthly, cmap="Blues")
        plt.title("Monthly Youth Aadhaar Activity Heatmap – Top 5 States")
        plt.xlabel("Month")
        plt.ylabel("State")
        plt.show()

    combined_df = combined_df.sort_values(["state", "date"])

    combined_df["youth_lag_1m"] = (
        combined_df.groupby("state")["total_youth"].shift(1)
    )

    combined_df["youth_lag_3m"] = (
        combined_df.groupby("state")["total_youth"].shift(3)
    )

    combined_df["youth_growth_rate"] = (
        (combined_df["total_youth"] - combined_df["youth_lag_1m"])
        / combined_df["youth_lag_1m"]
    )

    combined_df["youth_volatility_3m"] = (
        combined_df
        .groupby("state")["total_youth"]
        .rolling(3)
        .std()
        .reset_index(level=0, drop=True)
    )

    combined_df["youth_population_ratio"] = (
        combined_df["total_youth"] / combined_df["total_population"]
    )

    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    ml_df = combined_df[[
        "state",
        "total_youth",
        "youth_growth_rate",
        "youth_volatility_3m",
        "youth_population_ratio"
    ]].dropna()

    state_ml = (
        ml_df
        .groupby("state")
        .mean()
    )

    X = StandardScaler().fit_transform(state_ml)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    state_ml["cluster"] = kmeans.fit_predict(X)

    cluster_summary = (
        state_ml
        .groupby("cluster")
        .agg({
            "total_youth": "mean",
            "youth_growth_rate": "mean",
            "youth_volatility_3m": "mean",
            "youth_population_ratio": "mean"
        })
    )

    print("\nState-wise ML Cluster Assignment:\n")
    print(state_ml.sort_values("cluster"))

    for cluster_id in sorted(state_ml["cluster"].unique()):
        print(f"\nCluster {cluster_id} States:")
        print(
            state_ml[state_ml["cluster"] == cluster_id]
            .index
            .tolist()
        )

    print("\nCluster-wise Average Characteristics:\n")
    print(cluster_summary)

    return {
        "combined_df": combined_df,
        "monthly_youth": monthly_youth,
        "state_ml": state_ml,
        "cluster_summary": cluster_summary
    }
