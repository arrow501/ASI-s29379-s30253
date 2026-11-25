"""Optional marimo exploration; dependencies: requirements/visual-exploration.txt."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Visual exploration")


@app.cell(hide_code=True)
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px

    from asi_project.config import get_contract, load_project_config
    from asi_project.data import raw_data_path
    from asi_project.exploration import make_geometry, segment_rates, training_view, tsne_map
    from asi_project.pca_widget import PCATour

    repo_root = Path(__file__).resolve().parents[1]
    return (
        PCATour,
        get_contract,
        load_project_config,
        make_geometry,
        mo,
        np,
        os,
        pd,
        px,
        raw_data_path,
        repo_root,
        segment_rates,
        training_view,
        tsne_map,
    )


@app.cell(hide_code=True)
def exploration_navigation(mo):
    mo.sidebar(
        mo.outline(label="Visual exploration"),
        footer=mo.md("PCA/t-SNE: train 80% · split diagnostics: full sequence"),
        width=260,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    exploration_intro = mo.md(r"""
    # Visual exploration
    **Bank Marketing · look, rotate, select, question.**

    PCA, t-SNE and sections 1–5 explore the earlier chronological **training 80%**.
    Sections 6–8 explicitly diagnose splits and drift across the **full sequence**,
    including the already-inspected test. They do not fit representations on the test
    or turn it into a new unseen holdout.

    Start with segment rates, then rotate the PCA cloud. Select a region and ask
    which original features distinguish it. Finish by testing whether a t-SNE
    neighbourhood persists under another seed or perplexity.
    """)
    return (exploration_intro,)


@app.cell(hide_code=True)
def _(get_contract, load_project_config, mo, os, raw_data_path, repo_root):
    startup_config = load_project_config(repo_root)
    startup_contract = get_contract(startup_config)
    default_source = (
        "CI sample"
        if (
            os.environ.get("ASI_DATA_MODE") == "sample"
            or not raw_data_path(startup_config, startup_contract).is_file()
        )
        else "Full data (local)"
    )
    data_source = mo.ui.dropdown(
        {"Full data (local)": "full", "CI sample": "sample"},
        value=default_source,
        label="Data source",
    )
    return (data_source,)


@app.cell(hide_code=True)
def _(data_source, repo_root, training_view):
    config, contract, training_X, training_y = training_view(repo_root, data_source.value)
    return config, training_X, training_y


@app.cell(hide_code=True)
def _(data_source, exploration_intro, mo, np, training_X, training_y):
    positive_rate = float(training_y.mean())
    entropy_bits = -sum(p * np.log2(p) for p in (positive_rate, 1 - positive_rate) if p > 0)
    overview_stats = mo.hstack(
        [
            mo.stat(label="Training rows", value=f"{len(training_X):,}"),
            mo.stat(label="Positive share", value=f"{positive_rate:.1%}"),
            mo.stat(label="Always-no accuracy", value=f"{1 - positive_rate:.1%}"),
            mo.stat(label="Target entropy", value=f"{entropy_bits:.2f} bits / 1"),
        ],
        justify="space-around",
    )
    mo.vstack([exploration_intro, data_source, overview_stats])
    return (positive_rate,)


@app.cell(hide_code=True)
def _(mo):
    segments_intro = mo.md("""
    ## 1 · Segments before embeddings
    A high rate with tiny support can be a mirage. Error bars below are 95% Wilson
    intervals, assuming independent observations; campaign dependence can make
    them optimistic. These are associations, not effects of an intervention.
    """)
    segments_intro.style({"max-width": "74ch", "line-height": "1.7"})
    return


@app.cell(hide_code=True)
def _(config, mo, training_X):
    segment_feature = mo.ui.dropdown(
        list(config.categorical_features), value="job", label="Segment"
    )
    minimum_support = mo.ui.slider(
        1,
        max(2, min(500, len(training_X) // 4)),
        value=1,
        label="Minimum group size",
        show_value=True,
    )
    return minimum_support, segment_feature


@app.cell(hide_code=True)
def _(
    minimum_support,
    mo,
    positive_rate,
    px,
    segment_feature,
    segment_rates,
    training_X,
    training_y,
):
    segment_summary = segment_rates(
        training_X[segment_feature.value], training_y, minimum_support.value
    )
    segment_figure = px.scatter(
        segment_summary,
        x="rate",
        y="segment",
        size="rows",
        size_max=30,
        error_x=segment_summary.high - segment_summary.rate,
        error_x_minus=segment_summary.rate - segment_summary.low,
        hover_data=["rows", "positive", "low", "high"],
        title="Positive rate, uncertainty and group support",
        template="plotly_white",
    )
    segment_figure.add_vline(x=positive_rate, line_dash="dash", line_color="#2a9d8f")
    segment_figure.update_xaxes(tickformat=".0%", title="Positive rate")
    mo.vstack(
        [
            mo.hstack([segment_feature, minimum_support], justify="start"),
            mo.ui.plotly(segment_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    geometry_intro = mo.md(r"""
    ## 2 · Build a geometry
    Numeric features are imputed and scaled; categories are one-hot encoded.
    Standard scaling matches the baseline. Robust scaling uses medians/IQRs and
    changes the geometry; it does not remove extreme points. No labels enter PCA.

    Choose inputs, then **Apply geometry**. Display controls and rotations reuse
    these fitted coordinates. Sampling is uniform, with natural class prevalence.
    """)
    geometry_intro.style({"max-width": "74ch", "line-height": "1.7"})
    return


@app.cell(hide_code=True)
def _(config, mo, training_X):
    geometry_form = (
        mo.md("""
    {features}

    {scaling} {sample_size} {seed}
    """)
        .batch(
            features=mo.ui.multiselect(
                list(config.model_features),
                value=list(config.model_features),
                label="Features (ALLOWED only)",
            ),
            scaling=mo.ui.dropdown(
                {"Standard (baseline)": "standard", "Robust (median / IQR)": "robust"},
                value="Standard (baseline)",
                label="Numeric scaling",
            ),
            sample_size=mo.ui.slider(
                100,
                max(100, min(2500, len(training_X))),
                step=100,
                value=min(1500, max(100, (len(training_X) // 100) * 100)),
                label="Maximum map rows",
                show_value=True,
            ),
            seed=mo.ui.number(0, 2**32 - 1, value=config.random_state, label="Sampling seed"),
        )
        .form(submit_button_label="Apply geometry")
    )
    return (geometry_form,)


@app.cell(hide_code=True)
def geometry_controls(geometry_form, mo):
    mo.vstack(
        [
            mo.md("### Choose the representation"),
            geometry_form,
            mo.md(
                "**Apply geometry** updates fitted coordinates. Rotating, hovering and "
                "recolouring the plot below do not refit PCA."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(config, geometry_form, make_geometry, mo, training_X):
    geometry_settings = geometry_form.value or {
        "features": list(config.model_features),
        "scaling": "standard",
        "sample_size": min(1500, max(100, (len(training_X) // 100) * 100)),
        "seed": config.random_state,
    }
    mo.stop(not geometry_settings["features"], mo.md("Select at least one feature and Apply."))
    geometry = make_geometry(
        training_X,
        [f for f in config.numeric_features if f in geometry_settings["features"]],
        [f for f in config.categorical_features if f in geometry_settings["features"]],
        scaling=geometry_settings["scaling"],
        max_rows=geometry_settings["sample_size"],
        seed=int(geometry_settings["seed"]),
    )
    return (geometry,)


@app.cell(hide_code=True)
def _(geometry, mo, np, px):
    scree = px.bar(
        x=np.arange(1, len(geometry.pca.explained_variance_ratio_) + 1),
        y=geometry.pca.explained_variance_ratio_,
        labels={"x": "Principal component", "y": "Variance fraction"},
        title="How much geometry fits into a flat picture?",
        template="plotly_white",
    )
    scree.update_yaxes(tickformat=".0%")
    scree_panel = mo.vstack(
        [
            mo.md(
                f"**{geometry.encoded_sample.shape[1]} encoded dimensions** · "
                f"PC1 + PC2: **{geometry.pca.explained_variance_ratio_[:2].sum():.1%}** total variance. "
                "Hover a bar for its contribution. The tour uses up to the first eight PCs."
            ),
            mo.ui.plotly(scree),
        ]
    )
    scree_panel
    return


@app.cell(hide_code=True)
def _(mo):
    rotation_intro = mo.md(r"""
    ## 3 · Rotate the viewing plane

    Think of a fixed cloud in many dimensions and a moving sheet catching its shadow.
    PCA is fitted once. We rotate the viewing sheet; we do not refit the model or
    interpolate individual points between unrelated pictures.

    ### Follow one quarter-turn

    For a rotation in the PC1–PC3 plane, the horizontal axis becomes

    $$u(\theta)=\cos\theta\,e_1+\sin\theta\,e_3,\qquad v=e_2.$$

    At **0°**, the projection is PC1 × PC2. At **90°**, it is PC3 × PC2.
    Every intermediate frame is the corresponding orthographic projection.

    ### What stays true during motion

    With score matrix $Z$ and orthonormal viewing basis $B$, the displayed points
    are $ZB$. A Givens rotation updates $B\leftarrow R_{ij}(\theta)B$, preserving
    $B^\top B=I$. There is no whitening or per-frame stretching. Projected distances
    can change because we are changing what we can see.

    The variance badge uses **all encoded dimensions** as its denominator.
    The live equations underneath the plot identify the current viewing axes.
    """)
    rotation_intro.style({"max-width": "74ch", "line-height": "1.7"})
    return


@app.cell(hide_code=True)
def _(PCATour, geometry, mo, np, training_X, training_y):
    map_records = geometry.sample.copy()
    map_records["outcome"] = np.where(
        training_y.iloc[geometry.positions].to_numpy() == 1, "yes", "no"
    )
    map_records["source_row"] = geometry.sample.index.to_numpy() + 1
    map_records["chronology"] = geometry.positions / max(1, len(training_X) - 1)
    # JSON roundtrip converts nullable pandas values to browser-safe nulls.
    import json

    pca_tour = mo.ui.anywidget(
        PCATour(
            scores=geometry.scores[:, : min(8, geometry.scores.shape[1])].tolist(),
            variance=geometry.pca.explained_variance_ratio_.tolist(),
            records=json.loads(map_records.to_json(orient="records")),
        )
    )
    mo.vstack(
        [
            mo.md("**Try it:** choose **PC1 ↔ PC3**, then move the angle from **0° to 90°**."),
            pca_tour,
            mo.md(
                "A plane outside both current axes has no initial effect. Mix it with a visible "
                "component first. **Box select** pauses the motion; selected identities persist."
            ),
        ]
    )
    return map_records, pca_tour


@app.cell(hide_code=True)
def _(geometry, mo, pca_tour, pd, training_y):
    selected_indices = pca_tour.value.get("selected_ids", [])
    selection_frame = geometry.sample.iloc[selected_indices]
    selection_rate = (
        float(training_y.iloc[geometry.positions[selected_indices]].mean())
        if selected_indices
        else None
    )
    numeric_selection = selection_frame.select_dtypes(include="number").median()
    numeric_reference = geometry.sample.select_dtypes(include="number").median()
    selection_comparison = pd.DataFrame(
        {
            "selected median": numeric_selection,
            "map-sample median": numeric_reference,
        }
    )
    if selected_indices:
        selection_panel = mo.vstack(
            [
                mo.md(
                    f"### Inspect a region\n**{len(selected_indices)} rows** · "
                    f"positive share **{selection_rate:.1%}**. "
                    "Comparison is against the same map sample, not a predictive score."
                ),
                mo.ui.table(selection_comparison.reset_index(), selection=None),
                mo.ui.table(selection_frame, selection=None, page_size=10),
            ]
        )
    else:
        selection_panel = mo.md(
            "### Inspect a region\nUse **Box select** in the plot above. "
            "The selected rows and feature comparison will appear here."
        )
    selection_panel
    return


@app.cell(hide_code=True)
def _(mo):
    tsne_intro = mo.md("""
    ## 4 · t-SNE neighbourhood experiment
    This is a different map, not a continuation of the PCA rotation. t-SNE tries to
    keep neighbours close; island areas and distances between islands are not
    population sizes or reliable global distances. Compare seeds and perplexities.
    **Trustworthiness** checks false neighbours against the original encoded sample;
    a high value is not evidence of a good classifier. It uses quadratic memory,
    so the map is capped at 2,500 points. Only **Run t-SNE** starts a fit.
    """)
    tsne_intro.style({"max-width": "74ch", "line-height": "1.7"})
    return


@app.cell(hide_code=True)
def _(config, geometry, mo):
    tsne_form = (
        mo.md("{perplexity} {seed} {iterations}")
        .batch(
            perplexity=mo.ui.slider(
                5,
                max(5, min(80, len(geometry.sample) - 1)),
                value=min(30, len(geometry.sample) - 1),
                label="Perplexity",
                show_value=True,
            ),
            seed=mo.ui.number(0, 2**32 - 1, value=config.random_state, label="t-SNE seed"),
            iterations=mo.ui.slider(
                250, 1500, step=250, value=750, label="Iterations", show_value=True
            ),
        )
        .form(submit_button_label="Run t-SNE")
    )
    return (tsne_form,)


@app.cell(hide_code=True)
def _(geometry, mo, tsne_form, tsne_map):
    if tsne_form.value is None:
        tsne_result = None
    else:
        with mo.status.spinner(title="Fitting the neighbourhood map…"):
            tsne_result = tsne_map(
                geometry,
                perplexity=tsne_form.value["perplexity"],
                seed=int(tsne_form.value["seed"]),
                iterations=tsne_form.value["iterations"],
            )
    return (tsne_result,)


@app.cell(hide_code=True)
def _(mo):
    tsne_colour = mo.ui.dropdown(
        ["outcome", "chronology", "job", "contact", "poutcome"],
        value="outcome",
        label="t-SNE colour (does not refit)",
    )
    return (tsne_colour,)


@app.cell(hide_code=True)
def _(map_records, mo, px, tsne_colour, tsne_form, tsne_result):
    if tsne_result is None:
        tsne_view = mo.md("Ready when you are — choose settings and click **Run t-SNE**.")
    else:
        tsne_coordinates, tsne_trust, tsne_kl = tsne_result
        tsne_frame = map_records.assign(
            horizontal=tsne_coordinates[:, 0], vertical=tsne_coordinates[:, 1]
        )
        tsne_figure = px.scatter(
            tsne_frame,
            x="horizontal",
            y="vertical",
            color=tsne_colour.value,
            hover_data=[
                "source_row",
                "outcome",
                "age",
                "job",
                "balance",
                "month",
            ],
            template="plotly_white",
            title=f"t-SNE · trustworthiness {tsne_trust:.3f} · KL {tsne_kl:.3f}",
            color_discrete_map={"no": "#7e95ac", "yes": "#f4a340"},
        )
        tsne_figure.update_traces(marker={"size": 6, "opacity": 0.7})
        tsne_view = mo.ui.plotly(tsne_figure)
    mo.vstack([tsne_form, tsne_colour, tsne_view])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### From a picture to a useful hypothesis
    Ask: **What did I observe? What could explain it? What check could disprove it?**
    Look for campaign timing, known category boundaries, missingness and extreme
    numeric values before naming a cluster. Standard scaling gives numeric features
    unit variance; one-hot features have a different weighting. Neither view is
    a neutral map of customer similarity. No predictive model is trained here.
    """)
    return


@app.cell(hide_code=True)
def chronology_intro(mo):
    mo.md(r"""
    ## 5 · Watch the training data move

    The records are chronologically ordered, but they are **not evenly spaced time
    samples** and do not contain complete dates. These views use source-record order:
    they reveal changes along the campaign sequence, not daily rates, seasonality or
    forecasts. The final test portion remains excluded.

    A trailing window answers “what did the most recent *N records* look like?”
    Equal-sized blocks give a coarser view of feature and category movement.
    Overlapping windows share observations, so neighbouring values are dependent.
    """).style({"max-width": "74ch", "line-height": "1.7"})
    return


@app.cell(hide_code=True)
def chronology_controls(config, mo, training_X):
    sequence_window = mo.ui.slider(
        10,
        max(10, min(5000, (len(training_X) // 20) * 10)),
        step=10,
        value=min(1000, max(10, (len(training_X) // 20) * 10)),
        label="Trailing window (records)",
        show_value=True,
    )
    sequence_blocks = mo.ui.slider(
        5, min(40, len(training_X)), value=12, label="Equal-sized blocks", show_value=True
    )
    sequence_numeric = mo.ui.dropdown(
        list(config.numeric_features), value="balance", label="Numeric feature"
    )
    sequence_category = mo.ui.dropdown(
        list(config.categorical_features), value="contact", label="Category mix"
    )
    mo.vstack(
        [
            mo.hstack([sequence_window, sequence_blocks], justify="start", wrap=True),
            mo.hstack([sequence_numeric, sequence_category], justify="start", wrap=True),
            mo.md(
                "Controls update these views only. Hover for counts; drag to zoom; double-click to reset."
            ),
        ]
    )
    return (
        sequence_blocks,
        sequence_category,
        sequence_numeric,
        sequence_window,
    )


@app.cell(hide_code=True)
def chronology_data(
    np,
    pd,
    sequence_blocks,
    sequence_numeric,
    sequence_window,
    training_X,
    training_y,
):
    import plotly.graph_objects as go

    sequence_positions = np.arange(len(training_X))
    sequence_source_rows = training_X.index.to_numpy() + 1
    sequence_window_size = int(sequence_window.value)
    sequence_sample_positions = np.unique(
        np.linspace(
            0,
            len(training_X) - 1,
            min(1200, len(training_X)),
            dtype=int,
        )
    )
    sequence_rate = (
        training_y.reset_index(drop=True)
        .rolling(
            sequence_window_size,
            min_periods=sequence_window_size,
        )
        .mean()
        * 100
    )
    sequence_numeric_values = (
        training_X[sequence_numeric.value].astype(float).reset_index(drop=True)
    )
    sequence_minimum_valid = max(2, sequence_window_size // 5)
    sequence_feature_roll = sequence_numeric_values.rolling(
        sequence_window_size,
        min_periods=sequence_minimum_valid,
    )
    sequence_median = sequence_feature_roll.median()
    sequence_lower = sequence_feature_roll.quantile(0.25)
    sequence_upper = sequence_feature_roll.quantile(0.75)
    sequence_valid = sequence_numeric_values.rolling(sequence_window_size, min_periods=1).count()
    sequence_block_ids = np.minimum(
        sequence_positions * sequence_blocks.value // len(training_X),
        sequence_blocks.value - 1,
    )
    sequence_block_rows = pd.Series(sequence_source_rows).groupby(sequence_block_ids).mean()
    sequence_block_sizes = pd.Series(sequence_positions).groupby(sequence_block_ids).size()
    return (
        go,
        sequence_block_ids,
        sequence_block_rows,
        sequence_block_sizes,
        sequence_lower,
        sequence_median,
        sequence_minimum_valid,
        sequence_rate,
        sequence_sample_positions,
        sequence_source_rows,
        sequence_upper,
        sequence_valid,
        sequence_window_size,
    )


@app.cell(hide_code=True)
def chronology_rate_view(
    go,
    mo,
    positive_rate,
    sequence_rate,
    sequence_sample_positions,
    sequence_source_rows,
    sequence_window_size,
):
    sequence_rate_figure = go.Figure()
    sequence_rate_figure.add_trace(
        go.Scatter(
            x=sequence_source_rows[sequence_sample_positions],
            y=sequence_rate.iloc[sequence_sample_positions],
            mode="lines",
            name="Trailing positive rate",
            line={"color": "#d99127", "width": 2},
            hovertemplate="Source row %{x:,.0f}<br>Positive rate %{y:.2f}%<extra></extra>",
        )
    )
    sequence_rate_figure.add_hline(
        y=100 * positive_rate,
        line_dash="dash",
        line_color="#2a9d8f",
        annotation_text="All-training positive rate",
    )
    sequence_rate_figure.update_layout(
        template="plotly_white",
        height=350,
        title=f"How does the outcome rate move? · trailing {sequence_window_size:,} records",
        xaxis_title="Source-record order (not elapsed time)",
        yaxis_title="Positive rate (%)",
        margin={"t": 55, "b": 45},
    )
    mo.vstack(
        [
            mo.md(
                "### Outcome movement\nThe first window has no value until it is complete. "
                "Every estimate uses the preceding records, never future records."
            ),
            mo.ui.plotly(sequence_rate_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def chronology_feature_view(
    go,
    mo,
    sequence_lower,
    sequence_median,
    sequence_minimum_valid,
    sequence_numeric,
    sequence_sample_positions,
    sequence_source_rows,
    sequence_upper,
    sequence_valid,
):
    sequence_feature_figure = go.Figure()
    sequence_feature_figure.add_trace(
        go.Scatter(
            x=sequence_source_rows[sequence_sample_positions],
            y=sequence_lower.iloc[sequence_sample_positions],
            mode="lines",
            line={"width": 0},
            name="25th percentile",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    sequence_feature_figure.add_trace(
        go.Scatter(
            x=sequence_source_rows[sequence_sample_positions],
            y=sequence_upper.iloc[sequence_sample_positions],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(42,157,143,.18)",
            name="Middle 50% of observed values",
            hoverinfo="skip",
        )
    )
    sequence_feature_figure.add_trace(
        go.Scatter(
            x=sequence_source_rows[sequence_sample_positions],
            y=sequence_median.iloc[sequence_sample_positions],
            mode="lines",
            name="Median",
            line={"color": "#268a80", "width": 2},
            customdata=sequence_valid.iloc[sequence_sample_positions],
            hovertemplate="Source row %{x:,.0f}<br>Median %{y:.2f}<br>Observed in window %{customdata:,.0f}<extra></extra>",
        )
    )
    sequence_feature_figure.update_layout(
        template="plotly_white",
        height=350,
        title=f"{sequence_numeric.value} · trailing median and interquartile range",
        xaxis_title="Source-record order (not elapsed time)",
        yaxis_title=sequence_numeric.value,
        legend={"orientation": "h", "y": 1.1},
        margin={"t": 70, "b": 45},
    )
    mo.vstack(
        [
            mo.md(
                "### Feature movement\nThe band is the observed interquartile range, **not a confidence "
                "interval**. Missing values are omitted. "
                f"At least {sequence_minimum_valid:,} observed values are required per window; "
                "early windows may be partial. Gaps mean insufficient observations."
            ),
            mo.ui.plotly(sequence_feature_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def chronology_mix_view(
    mo,
    pd,
    px,
    sequence_block_ids,
    sequence_block_rows,
    sequence_category,
    training_X,
):
    sequence_mix = (
        pd.crosstab(
            sequence_block_ids,
            training_X[sequence_category.value].fillna("<missing>").astype(str).to_numpy(),
            normalize="index",
        )
        * 100
    )
    sequence_mix.index = sequence_block_rows.to_numpy()
    sequence_mix.index.name = "source_row_midpoint"
    sequence_mix_long = sequence_mix.reset_index().melt(
        id_vars="source_row_midpoint",
        var_name="category",
        value_name="share_percent",
    )
    sequence_mix_figure = px.area(
        sequence_mix_long,
        x="source_row_midpoint",
        y="share_percent",
        color="category",
        title=f"{sequence_category.value} · composition across equal-sized record blocks",
        labels={
            "source_row_midpoint": "Source-record block midpoint",
            "share_percent": "Share (%)",
        },
        template="plotly_white",
    )
    sequence_mix_figure.update_layout(height=380, yaxis_range=[0, 100], margin={"t": 55, "b": 45})
    mo.vstack(
        [
            mo.md(
                "### Who is in each part of the campaign?\nCompare category movement with the "
                "outcome trend above. Coloured areas show composition, not conversion rate or causality. "
                "Lines connect block summaries; they do not reconstruct between-record timing."
            ),
            mo.ui.plotly(sequence_mix_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def history_support_control(mo, training_X):
    history_minimum = mo.ui.slider(
        1,
        500,
        value=min(30, max(1, len(training_X) // 100)),
        label="Minimum group size for outcome rates",
        show_value=True,
    )
    return (history_minimum,)


@app.cell(hide_code=True)
def chronology_contact_history(
    go,
    history_minimum,
    mo,
    np,
    pd,
    sequence_block_ids,
    sequence_block_rows,
    sequence_block_sizes,
    sequence_blocks,
    training_X,
    training_y,
):
    from plotly.subplots import make_subplots

    history_frame = pd.DataFrame(
        {
            "block": sequence_block_ids,
            "history": np.where(
                training_X.previous.to_numpy() > 0,
                "Recorded prior contact",
                "No recorded prior contact",
            ),
            "positive": training_y.to_numpy(),
        }
    )
    history_groups = (
        history_frame.groupby(["block", "history"])["positive"]
        .agg(
            positive="sum",
            rows="count",
        )
        .reindex(
            pd.MultiIndex.from_product(
                [
                    range(sequence_blocks.value),
                    ["No recorded prior contact", "Recorded prior contact"],
                ],
                names=["block", "history"],
            ),
            fill_value=0,
        )
        .reset_index()
    )
    history_groups["midpoint"] = history_groups.block.map(sequence_block_rows)
    history_groups["rate"] = history_groups.positive / history_groups.rows.replace(0, np.nan)
    history_n = history_groups.rows.replace(0, np.nan)
    history_z = 1.959963984540054
    history_centre = (history_groups.rate + history_z**2 / (2 * history_n)) / (
        1 + history_z**2 / history_n
    )
    history_half = (
        history_z
        * np.sqrt(
            history_groups.rate * (1 - history_groups.rate) / history_n
            + history_z**2 / (4 * history_n**2)
        )
        / (1 + history_z**2 / history_n)
    )
    history_groups["low"] = np.minimum(
        history_groups.rate, np.maximum(0, history_centre - history_half)
    )
    history_groups["high"] = np.maximum(
        history_groups.rate, np.minimum(1, history_centre + history_half)
    )
    history_repeat = history_groups.loc[history_groups.history.eq("Recorded prior contact")].copy()
    history_repeat["share"] = history_repeat.rows.to_numpy() / sequence_block_sizes.to_numpy() * 100
    history_plot = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.16,
        subplot_titles=[
            "Audience mix · share with recorded prior contact",
            "Outcomes within each audience · do the groups also change?",
        ],
        row_heights=[0.32, 0.68],
    )
    history_plot.add_trace(
        go.Scatter(
            x=history_repeat.midpoint,
            y=history_repeat.share,
            mode="lines+markers",
            line={"color": "#2a9d8f", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(42,157,143,.12)",
            name="Prior-contact share",
            showlegend=False,
            customdata=np.column_stack([history_repeat.rows, sequence_block_sizes]),
            hovertemplate="Source-row midpoint %{x:,.0f}<br>Prior-contact share %{y:.1f}%"
            "<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} records<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for history_name, history_colour in [
        ("No recorded prior contact", "#7e95ac"),
        ("Recorded prior contact", "#2a9d8f"),
    ]:
        history_part = history_groups.loc[history_groups.history.eq(history_name)].copy()
        history_visible = history_part.rows >= history_minimum.value
        history_plot.add_trace(
            go.Scatter(
                x=history_part.midpoint,
                y=(100 * history_part.rate).where(history_visible),
                mode="lines+markers",
                name=history_name,
                connectgaps=False,
                line={"color": history_colour, "width": 2},
                marker={
                    "size": 6 + 18 * np.sqrt(history_part.rows / history_groups.rows.max()),
                    "color": history_colour,
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (100 * (history_part.high - history_part.rate)).where(history_visible),
                    "arrayminus": (100 * (history_part.rate - history_part.low)).where(
                        history_visible
                    ),
                    "thickness": 1,
                    "width": 3,
                },
                customdata=np.column_stack(
                    [history_part.rows, history_part.positive, history_part.block + 1]
                ),
                hovertemplate="Block %{customdata[2]:.0f} · source-row midpoint %{x:,.0f}"
                "<br>Positive rate %{y:.2f}%<br>%{customdata[1]:,.0f} positives / "
                "%{customdata[0]:,.0f} records<extra>%{fullData.name}</extra>",
            ),
            row=2,
            col=1,
        )

    history_plot.update_yaxes(
        title_text="Prior-contact share (%)", rangemode="tozero", row=1, col=1
    )
    history_plot.update_yaxes(title_text="Positive rate (%)", rangemode="tozero", row=2, col=1)
    history_plot.update_xaxes(
        title_text="Source-record order · equal-count block midpoints", row=2, col=1
    )
    history_plot.update_layout(
        template="plotly_white",
        height=620,
        hovermode="closest",
        legend={"orientation": "h", "y": -0.16},
        margin={"t": 45, "b": 100},
    )
    history_hidden = int(
        ((history_groups.rows > 0) & (history_groups.rows < history_minimum.value)).sum()
    )
    mo.vstack(
        [
            mo.md(
                "### Is the audience changing, or are outcomes changing within it?\n"
                "Read vertically: the top shows **how common prior-contact history becomes**; "
                "the bottom compares **positive rates within each group** at the same point in the campaign."
            ),
            history_minimum,
            mo.ui.plotly(history_plot),
            mo.md(
                f"Marker size shows group support; bars are **95% Wilson intervals**. "
                f"{history_hidden} nonempty group-blocks below the support threshold are hidden; "
                "absent groups are gaps, not zero conversion. Intervals assume independence and may "
                "be optimistic for repeated contacts. Prior history is `previous > 0`, not proof of "
                "a unique returning customer. A difference between curves is an association, not "
                "the causal effect of contacting someone again."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def chronology_reading(mo):
    mo.md("""
    ### Read the views together

    When the positive rate changes, ask whether the feature distributions, contact
    channels or data coverage change at the same point. Then change the window size:
    does the pattern persist, or was it noise at one smoothing scale?

    Abrupt movements are hypotheses about campaign regimes or sampling changes.
    They are not automatic change-point detections. Periodicity in record order
    would not establish calendar seasonality. No forecasting model is fitted here.
    """).style({"max-width": "74ch", "line-height": "1.7"})
    return


@app.cell(hide_code=True)
def split_imports():
    from asi_project.split_exploration import (
        PART_COLOURS,
        PART_LABELS,
        category_drift,
        full_sequence_view,
        read_oof_catalog,
        split_layout,
        split_summary,
        threshold_counts,
        threshold_curve,
    )

    return (
        PART_COLOURS,
        PART_LABELS,
        category_drift,
        full_sequence_view,
        read_oof_catalog,
        split_layout,
        split_summary,
        threshold_counts,
        threshold_curve,
    )


@app.cell(hide_code=True)
def split_intro(mo):
    mo.md(r"""
    ## 6 · Splits and drift — what are we comparing?

    This panel shows the **full sequence**, including the already-inspected test set.
    It describes data after exploration; it is not a new independent model audit.
    PCA, t-SNE and the earlier views still use only the first 80%.

    Check whether a single validation window resembles training and the later period.
    The outer test must preserve chronology; equal record counts do not imply equal
    amounts of time. We do not reconstruct dates or years from `day` and `month` alone.
    """).style({"max-width": "82ch", "line-height": "1.65"})
    return


@app.cell(hide_code=True)
def split_source(
    config,
    data_source,
    full_sequence_view,
    read_oof_catalog,
    repo_root,
):
    split_X, split_y, split_input_fingerprint = full_sequence_view(repo_root, data_source.value)
    split_oof_catalog = read_oof_catalog(
        repo_root, split_input_fingerprint, split_y, config.test_size
    )
    return split_X, split_oof_catalog, split_y


@app.cell(hide_code=True)
def split_protocol_control(mo, split_y):
    split_protocol = mo.ui.dropdown(
        {
            "Holdout · default 64/16/20": "holdout",
            "80/20 · random bagging folds": "bagged",
            "80/20 · expanding chronological folds": "chronological",
        },
        value="Holdout · default 64/16/20",
        label="Protocol",
        full_width=True,
    )
    split_window = mo.ui.slider(
        10,
        max(10, min(5000, len(split_y) // 10 * 10)),
        step=10,
        value=max(10, min(1000, len(split_y) // 10)),
        label="Complete rolling window (records)",
        show_value=True,
    )
    mo.hstack([split_protocol, split_window], widths=[2, 1], align="end", wrap=True)
    return split_protocol, split_window


@app.cell(hide_code=True)
def split_fold_control(mo, split_protocol):
    split_validation_percent = mo.ui.slider(
        5,
        30,
        value=20,
        step=1,
        label="Validation as % of development (first 80%)",
        show_value=True,
        disabled=split_protocol.value == "bagged",
    )
    split_fold = mo.ui.slider(
        1,
        5 if split_protocol.value == "bagged" else 3,
        value=3 if split_protocol.value == "chronological" else 1,
        label="Displayed fold",
        show_value=True,
        disabled=split_protocol.value == "holdout",
    )
    mo.vstack(
        [
            mo.hstack([split_validation_percent, split_fold], justify="start", wrap=True),
            mo.md(
                "Changing these controls does not train a model or recompute geometry. "
                "For bagging, the map illustrates a random stratified split; AutoGluon manages its own folds, and RF/XT may use OOB."
            ),
        ]
    )
    return split_fold, split_validation_percent


@app.cell(hide_code=True)
def split_membership_data(
    config,
    np,
    split_X,
    split_fold,
    split_layout,
    split_protocol,
    split_summary,
    split_validation_percent,
    split_y,
):
    split_membership = split_layout(
        split_y,
        split_protocol.value,
        validation_fraction=split_validation_percent.value / 100,
        fold=int(split_fold.value),
        test_size=config.test_size,
        seed=config.random_state,
    )
    split_stats = split_summary(split_X, split_y, split_membership)
    split_plot_positions = np.unique(
        np.linspace(0, len(split_y) - 1, min(1800, len(split_y)), dtype=int)
    )
    split_development_cut = len(split_y) - int(np.ceil(len(split_y) * config.test_size))
    return (
        split_development_cut,
        split_membership,
        split_plot_positions,
        split_stats,
    )


@app.cell(hide_code=True)
def split_sequence_cache(split_window, split_y):
    split_rolling_rate = (
        split_y.reset_index(drop=True)
        .rolling(int(split_window.value), min_periods=int(split_window.value))
        .mean()
        * 100
    )
    return (split_rolling_rate,)


@app.cell(hide_code=True)
def split_membership_view(
    PART_COLOURS,
    PART_LABELS,
    go,
    mo,
    split_membership,
    split_plot_positions,
):
    split_part_order = list(PART_LABELS)
    split_part_codes = {part: number for number, part in enumerate(split_part_order)}
    split_map_figure = go.Figure(
        go.Heatmap(
            x=split_plot_positions + 1,
            y=[0],
            z=[split_membership.iloc[split_plot_positions].map(split_part_codes).to_list()],
            text=[split_membership.iloc[split_plot_positions].map(PART_LABELS).to_list()],
            zmin=-0.5,
            zmax=3.5,
            zsmooth=False,
            colorscale=[
                [0, PART_COLOURS["train"]],
                [0.2499, PART_COLOURS["train"]],
                [0.25, PART_COLOURS["validation"]],
                [0.4999, PART_COLOURS["validation"]],
                [0.5, PART_COLOURS["test"]],
                [0.7499, PART_COLOURS["test"]],
                [0.75, PART_COLOURS["unused"]],
                [1, PART_COLOURS["unused"]],
            ],
            showscale=False,
            hovertemplate="Position %{x:,.0f}<br>%{text}<extra></extra>",
        )
    )
    split_map_figure.update_layout(
        template="plotly_white",
        height=150,
        title="Which records belong to the displayed fold?",
        yaxis={"visible": False},
        xaxis_title="Record position · train green, validation amber, test red, unused grey",
        margin={"t": 45, "b": 45, "l": 45, "r": 20},
    )
    mo.vstack(
        [
            mo.ui.plotly(split_map_figure),
            mo.md(
                "The map shows up to 1,800 positions; counts and statistics use every record. "
                "The grey development tail in an earlier fold is used for neither its training nor its validation."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def split_rate_view(
    PART_COLOURS,
    go,
    mo,
    np,
    split_X,
    split_development_cut,
    split_membership,
    split_plot_positions,
    split_protocol,
    split_rolling_rate,
    split_window,
    split_y,
):
    split_rate_figure = go.Figure(
        go.Scatter(
            x=split_plot_positions + 1,
            y=split_rolling_rate.iloc[split_plot_positions],
            mode="lines",
            line={"color": "#285e8e", "width": 2},
            name="Positive rate",
            customdata=split_X.index.to_numpy()[split_plot_positions],
            hovertemplate="Position %{x:,.0f}<br>Source index %{customdata}<br>Positives %{y:.2f}%<extra></extra>",
        )
    )
    split_rate_figure.add_vrect(
        x0=split_development_cut + 0.5,
        x1=len(split_y) + 0.5,
        fillcolor=PART_COLOURS["test"],
        opacity=0.12,
        line_width=0,
        annotation_text="previously inspected test 20%",
    )
    if split_protocol.value != "bagged":
        split_validation_positions = np.flatnonzero(split_membership.eq("validation"))
        split_rate_figure.add_vrect(
            x0=int(split_validation_positions.min()) + 0.5,
            x1=int(split_validation_positions.max()) + 1.5,
            fillcolor=PART_COLOURS["validation"],
            opacity=0.17,
            line_width=0,
            annotation_text="validation",
        )
    split_rate_figure.update_layout(
        template="plotly_white",
        height=330,
        title=f"Does validation reflect the later period? · window {int(split_window.value):,}",
        xaxis_title="Record order, not elapsed time",
        yaxis_title="Positives (%)",
        margin={"t": 55, "b": 45, "l": 50, "r": 20},
    )
    mo.ui.plotly(split_rate_figure)
    return


@app.cell(hide_code=True)
def split_summary_view(PART_LABELS, mo, split_stats):
    split_summary_table = (
        split_stats.assign(
            part=split_stats.part.map(PART_LABELS),
            rate=100 * split_stats.rate,
            low=100 * split_stats.low,
            high=100 * split_stats.high,
            previous_contact=100 * split_stats.previous_contact,
        )
        .rename(
            columns={
                "part": "Partition",
                "rows": "Records",
                "positive": "Positives",
                "rate": "Positives %",
                "low": "Wilson 95% lower",
                "high": "Wilson 95% upper",
                "previous_contact": "Previous contact %",
                "first_position": "First position",
                "last_position": "Last position",
            }
        )
        .round(2)
    )
    mo.vstack(
        [
            mo.ui.table(
                split_summary_table, selection=None, pagination=False, show_column_summaries=False
            ),
            mo.md(
                "Wilson intervals describe binomial uncertainty assuming independence; contacts and campaigns may be dependent. "
                "The intervals do not correct for drift or campaign selection. In a random fold, first–last position does not describe a continuous block."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def split_drift_controls(config, mo):
    split_category = mo.ui.dropdown(
        list(config.categorical_features), value="month", label="Category distribution"
    )
    mo.vstack(
        [
            mo.md("""
    ## 7 · Have the features changed too, alongside the class balance?

    Choose a feature and compare the partitions. Inspect categories absent from training
    and the previous-contact shares in the table above. These describe differences,
    not their causes.
    """),
            split_category,
        ]
    )
    return (split_category,)


@app.cell(hide_code=True)
def split_drift_view(
    PART_COLOURS,
    PART_LABELS,
    category_drift,
    mo,
    px,
    split_X,
    split_category,
    split_membership,
):
    split_category_data = category_drift(split_X, split_membership, split_category.value)
    split_category_plot = split_category_data.assign(
        share=split_category_data.share * 100, part=split_category_data.part.map(PART_LABELS)
    )
    split_category_figure = px.bar(
        split_category_plot,
        x="category",
        y="share",
        color="part",
        barmode="group",
        custom_data=["rows", "unseen_in_train"],
        color_discrete_map={PART_LABELS[part]: colour for part, colour in PART_COLOURS.items()},
        labels={
            "category": split_category.value,
            "share": "Share within partition (%)",
            "part": "Partition",
        },
        title="Train / validation / test composition",
        height=330,
    )
    split_category_figure.update_traces(
        hovertemplate="%{x}<br>Share %{y:.2f}%<br>Records %{customdata[0]}<br>Absent from training: %{customdata[1]}<extra>%{fullData.name}</extra>"
    )
    split_novel_categories = split_category_data.loc[
        split_category_data.unseen_in_train & split_category_data.rows.gt(0),
        ["part", "category", "rows", "share"],
    ].assign(part=lambda df: df.part.map(PART_LABELS))
    mo.vstack(
        [
            mo.ui.plotly(split_category_figure),
            mo.md(
                "Categories absent from training: **"
                + (
                    "; ".join(
                        f"{row.part}: {row.category} ({row.rows} records)"
                        for row in split_novel_categories.itertuples()
                    )
                    if len(split_novel_categories)
                    else "none in the current split"
                )
                + "**. `unknown` remains a category and stays in the denominator."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def split_stability_view(mo, pd, px, split_oof_catalog):
    split_stability_rows = [
        {
            "model": record["model"],
            "fold": number + 1,
            "f1": score,
            "threshold": record["scores"]["decision_threshold"],
            "run": record["run"],
        }
        for record in split_oof_catalog.values()
        if record["protocol"] == "chronological"
        for number, score in enumerate(record["scores"].get("fold_f1", []))
    ]
    mo.stop(
        not split_stability_rows,
        mo.callout(
            "No local fold results for this data source. Distribution views work independently.",
            kind="info",
        ),
    )
    split_stability_frame = pd.DataFrame(split_stability_rows)
    split_stability_figure = px.line(
        split_stability_frame,
        x="fold",
        y="f1",
        color="model",
        markers=True,
        custom_data=["threshold", "run"],
        height=330,
        labels={
            "fold": "Successive validation window",
            "f1": "F1 at the threshold selected across all windows",
        },
        title="Does the model ranking persist across periods?",
    )
    split_stability_figure.update_xaxes(dtick=1)
    split_stability_figure.update_traces(
        hovertemplate="Fold %{x}<br>F1 %{y:.3f}<br>Threshold %{customdata[0]:.4f}<br>%{customdata[1]}<extra>%{fullData.name}</extra>"
    )
    mo.vstack(
        [
            mo.md("### Saved results, without retraining"),
            mo.ui.plotly(split_stability_figure),
            mo.md(
                "Results come from a local run with a matching data fingerprint. The shared threshold was selected on these same windows: "
                "this chart diagnoses stability; it is not an independent estimate of generalisation."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def split_threshold_intro(mo):
    mo.md(r"""
    ## 8 · F1 = 0: no signal, or a decision at threshold 0.5?

    The model estimates probabilities; the threshold turns them into decisions. If all
    scores are below 0.5, it predicts only negatives and F1 is zero — even if it ranks
    cases better than chance. Move the threshold and inspect precision, recall and errors.

    We use only saved **OOF/forward predictions from the first 80%**, matching the current
    source fingerprint and labels. No predictions from the final test enter this panel.
    """).style({"max-width": "82ch", "line-height": "1.65"})
    return


@app.cell(hide_code=True)
def split_threshold_controls(mo, split_oof_catalog):
    mo.stop(
        not split_oof_catalog,
        mo.callout(
            "No saved OOF predictions for this source. This optional panel does not require a model or AutoGluon. "
            "Files are read from data/08_reporting/experiments/*/*_oof.json. The CI sample does not use full-dataset results.",
            kind="info",
        ),
    )
    split_oof_choice = mo.ui.dropdown(
        list(split_oof_catalog),
        value=next(iter(split_oof_catalog)),
        label="Saved model and protocol",
        full_width=True,
    )
    split_threshold = mo.ui.slider(
        0, 1, step=0.001, value=0.5, show_value=True, include_input=True, label="Threshold P(y=1)"
    )
    mo.vstack([split_oof_choice, split_threshold])
    return split_oof_choice, split_threshold


@app.cell(hide_code=True)
def split_threshold_data(
    np,
    split_oof_catalog,
    split_oof_choice,
    threshold_curve,
):
    split_selected_oof = split_oof_catalog[split_oof_choice.value]
    split_curve = threshold_curve(split_selected_oof["target"], split_selected_oof["probability"])
    split_best_threshold = float(
        split_curve.loc[split_curve.f1.eq(split_curve.f1.max())].iloc[-1].threshold
    )
    split_curve_positions = np.unique(
        np.linspace(0, len(split_curve) - 1, min(1500, len(split_curve)), dtype=int)
    )
    split_curve_plot = split_curve.iloc[split_curve_positions]
    return split_best_threshold, split_curve_plot, split_selected_oof


@app.cell(hide_code=True)
def split_threshold_view(
    go,
    mo,
    split_best_threshold,
    split_curve_plot,
    split_selected_oof,
    split_threshold,
    threshold_counts,
):
    split_threshold_metrics = threshold_counts(
        split_selected_oof["target"], split_selected_oof["probability"], split_threshold.value
    )
    split_threshold_figure = go.Figure()
    for _metric in ["f1", "precision", "recall"]:
        split_threshold_figure.add_trace(
            go.Scatter(
                x=split_curve_plot.threshold,
                y=split_curve_plot[_metric],
                name=_metric,
                mode="lines",
                hovertemplate="Threshold %{x:.4f}<br>Score %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    split_threshold_figure.add_vline(
        x=0.5, line_dash="dash", line_color="#89929d", annotation_text="0.5"
    )
    split_threshold_figure.add_vline(
        x=split_best_threshold,
        line_dash="dot",
        line_color="#2a9d8f",
        annotation_text="max F1 on OOF",
    )
    split_threshold_figure.add_vline(x=split_threshold.value, line_color="#d99127")
    split_threshold_figure.update_layout(
        template="plotly_white",
        height=330,
        xaxis_title="Threshold",
        yaxis_title="OOF / forward score",
        yaxis_range=[0, 1.03],
        margin={"t": 35, "b": 45},
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        label=metric.upper() if metric == "f1" else metric,
                        value=f"{split_threshold_metrics[metric]:.3f}",
                    )
                    for metric in ["f1", "precision", "recall"]
                ]
            ),
            mo.ui.plotly(split_threshold_figure),
            mo.md(
                f"TP **{split_threshold_metrics['tp']}** · FP **{split_threshold_metrics['fp']}** · "
                f"FN **{split_threshold_metrics['fn']}** · TN **{split_threshold_metrics['tn']}** · "
                f"maximum F1 at threshold **{split_best_threshold:.5f}**. No fitting and no use of the final test."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def split_reading(mo):
    mo.md("""
    ### What should we check before the next experiment?

    - Does strong performance persist across successive windows, or only one period?
    - Does the positive rate rise alongside changes in contact channel and campaign history?
    - Does F1 change because the model separates classes better, or because we moved the threshold?
    - Are we comparing the same test records? Scores on the final 15% and 20% are not directly comparable.

    Do not search for the split boundary that maximises the test score. This test has
    already been inspected; base further decisions on training and validation, and treat
    test results as a previously inspected benchmark.
    """).style({"max-width": "82ch", "line-height": "1.65"})
    return


if __name__ == "__main__":
    app.run()
