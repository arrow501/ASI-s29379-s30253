"""Geometry invariants and training-only exploration regressions."""

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from asi_project.exploration import make_geometry, segment_rates, training_view, tsne_map


def test_geometry_is_label_free_and_sampling_keeps_source_positions():
    X = pd.DataFrame(
        {"num": [0, 1, np.nan, 3, 100, 2], "cat": list("aabbcc")}, index=[10, 20, 30, 40, 50, 60]
    )
    first = make_geometry(X, ["num"], ["cat"], max_rows=4, seed=7)
    repeated = make_geometry(X, ["num"], ["cat"], max_rows=4, seed=7)
    pd.testing.assert_frame_equal(first.sample, X.iloc[first.positions])
    np.testing.assert_array_equal(first.positions, repeated.positions)
    np.testing.assert_allclose(first.scores, repeated.scores)
    assert np.isfinite(first.encoded_sample).all()
    assert np.isclose(first.pca.explained_variance_ratio_.sum(), 1)


def test_training_view_excludes_last_chronological_rows(repo_root):
    config, contract, X, y = training_view(repo_root, "sample")
    raw = pd.read_csv(config.ci_sample_path)
    assert len(X) == len(raw) - int(np.ceil(len(raw) * config.test_size))
    assert X.index.max() < len(X)
    assert contract.target not in X and not set(contract.excluded_features).intersection(X)
    pd.testing.assert_index_equal(X.index, y.index)


def test_wilson_endpoints_and_support_filter():
    values = pd.Series(["zero"] * 10 + ["all"] * 10 + ["tiny"])
    y = pd.Series([0] * 10 + [1] * 10 + [1])
    rates = segment_rates(values, y, minimum=2).set_index("segment")
    assert "tiny" not in rates.index
    assert rates.loc["zero", "low"] == 0
    assert rates.loc["all", "high"] == 1
    assert ((rates.low <= rates.rate) & (rates.rate <= rates.high)).all()
    assert rates.loc["zero", "high"] == pytest.approx(0.2775328, abs=1e-6)


def test_invalid_geometry_and_perplexity_fail_clearly():
    X = pd.DataFrame({"age": [20, 30, 40]})
    with pytest.raises(ValueError, match="Select"):
        make_geometry(X, [], [])
    X["balance"] = [1, 3, 2]
    geometry = make_geometry(X, ["age", "balance"], [], max_rows=3)
    with pytest.raises(ValueError, match="Perplexity"):
        tsne_map(geometry, perplexity=3)


def test_tsne_returns_finite_coordinates_on_the_same_sample():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    geometry = make_geometry(X, list(X.columns), [], max_rows=40)
    coordinates, quality, kl = tsne_map(geometry, perplexity=10, iterations=500)
    assert coordinates.shape == (40, 2)
    assert np.isfinite(coordinates).all() and np.isfinite(kl)
    assert 0 <= quality <= 1


def test_actual_browser_rotation_preserves_orthonormality(repo_root):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; needed to verify the browser's rotation implementation.")
    script = r"""
    import assert from 'node:assert/strict';
    const {rotateBasis, project} = await import(process.argv[1]);
    const initial = Array.from({length:8},(_,i)=>[Number(i===0),Number(i===1)]);
    const quarter = rotateBasis(initial,0,2,Math.PI/2);
    const result = project([[3,5,7,0,0,0,0,0]],quarter)[0];
    assert.ok(Math.abs(result[0]-7)<1e-12 && Math.abs(result[1]-5)<1e-12);
    let basis=initial;
    for(let k=0;k<10000;k++) basis=rotateBasis(basis,k%8,(k%8+3)%8,.013);
    for(let i=0;i<2;i++) for(let j=0;j<2;j++) {
      const dot=basis.reduce((sum,row)=>sum+row[i]*row[j],0);
      assert.ok(Math.abs(dot-Number(i===j))<1e-10);
    }
    const backwards=rotateBasis(quarter,0,2,-Math.PI/2);
    backwards.forEach((row,i)=>row.forEach((v,j)=>assert.ok(Math.abs(v-initial[i][j])<1e-12)));
    assert.throws(()=>rotateBasis(initial,0,0,1));
    """
    module = (repo_root / "src/asi_project/pca_tour.mjs").as_uri()
    subprocess.run(
        [node, "--input-type=module", "-e", script, module],
        check=True,
        capture_output=True,
        text=True,
    )
