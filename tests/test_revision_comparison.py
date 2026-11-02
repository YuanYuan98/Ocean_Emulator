"""Small numerical checks; synthetic fields never enter the report directory."""

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

spec = importlib.util.spec_from_file_location(
    'revision_comparison', Path(__file__).parents[1] / 'scripts/compare_revision_baselines.py'
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_land_mask_and_weighted_correlation():
    area = np.array([[1., 3.], [2., 4.]])
    weights = area * np.array([[True, False], [True, False]])
    truth = np.array([[2., 1e20], [5., np.nan]])
    np.testing.assert_allclose(comparison.weighted_mean(truth, weights), 4.)
    np.testing.assert_allclose(comparison.correlation(truth, 3*truth+7, weights), 1.)
    np.testing.assert_allclose(comparison.correlation(truth, -truth, weights), -1.)


def test_seasonality_area_and_forecast_r2():
    day = np.tile(np.arange(1, 5), 3)
    field = np.tile(np.arange(4), 3).astype(float)[:, None, None]
    np.testing.assert_allclose(comparison.seasonal_anomaly(field, day), 0.)
    nodes = np.polynomial.legendre.leggauss(8)[0]
    area = comparison.gaussian_area(np.degrees(np.arcsin(nodes)), np.arange(.5, 360, 1))
    np.testing.assert_allclose(area.sum(), 4*np.pi*6.371e6**2)
    metrics = comparison.time_metrics(np.arange(10), np.arange(10)+1)
    assert metrics['rmse'] == 1
    assert metrics['r2'] < 1  # Correlation squared would incorrectly equal one.


def test_identical_rollout_end_to_end(tmp_path, monkeypatch):
    lat = np.degrees(np.arcsin(np.polynomial.legendre.leggauss(4)[0]))
    lon = np.arange(22.5, 360, 45)
    time = xr.date_range('2014-10-20T12:00:00', periods=598, freq='5D', calendar='julian', use_cftime=True)
    t = np.arange(598)[:, None, None]
    pattern = np.arange(32).reshape(1, 4, 8)/100
    field = np.sin(t/13)*(1+pattern)+t/1000+pattern
    truth = xr.Dataset(coords={'time':time, 'y':lat, 'x':lon})
    wet = np.ones((4, 8), dtype=bool); wet[0, 0] = False
    for level in range(19):
        truth[f'mask_{level}'] = (('y','x'), wet)
        for var in ['thetao','so','uo','vo']:
            truth[f'{var}_{level}'] = (('time','y','x'), np.where(wet, field/(level+1), np.nan))
    truth['zos'] = (('time','y','x'), np.where(wet, field, np.nan))
    pred = truth.drop_vars([f'mask_{i}' for i in range(19)]).rename({'y':'lat','x':'lon'})
    marker = tmp_path/'predictions.zarr'; marker.mkdir()
    monkeypatch.setattr(comparison,'RUNS',{'samudra2':('Samudra 2',marker,None)})
    monkeypatch.setattr(comparison,'REGIONS',{'Global':None,'Niño 3.4':(0,360,-90,90)})
    monkeypatch.setattr(comparison.xr,'open_zarr',lambda path: truth if str(path).endswith('/OM4.zarr') else pred)
    output = tmp_path/'synthetic_validation'
    comparison.reduce_model('samudra2',output)
    frame = comparison.pd.read_csv(output/'samudra2/metrics.csv')
    np.testing.assert_allclose(frame[['raw_rmse','raw_bias','direct_rmse','detrend_rmse','variance_rmse']],0,atol=1e-12)
    np.testing.assert_allclose(frame['variance_corr'],1,atol=1e-12)
    np.testing.assert_allclose(frame['variance_ratio'],1,atol=1e-12)
    arrays = np.load(output/'samudra2/diagnostics.npz')
    assert np.isnan(arrays['thetao_0_truth_variance'][0,0])
    np.testing.assert_allclose(arrays['ohc_truth_0'],arrays['ohc_pred_0'])
    comparison.render(output)
    assert (output/'comparison.ipynb').is_file()
    assert (output/'figures/nino34.png').is_file()
