# Observation Evaluation Summary - Updated Result Figures (2026-07-22)

- Document ID: 1rf7-BlilpIvkqdiSmXIRVhckPmukkwf2awMaRVKo4ZQ
- Revision ID: ALtnJHw_xDFAnBTK6kYmyhJ5L8v9IgmwZcm9M8AsNTUcu0crC9PqommgDU5jnqE2Nh7F7jhNYjvzfyScbpoA7eFADfnjB5c7aNB6LYk9TBc
- Selected tab: all
- Protected controls: 0
- Opaque controls: 0
- Authoritative dropdowns: 0

Protected-control annotations are preservation instructions. Do not insert their displayed placeholder text to recreate a native control.

## Tab 1 (t.0)

[P00001 | 1:32 | TITLE]
Observation Evaluation Summary

[P00002 | 32:51 | HEADING_1]
1. Metrics Summary

[P00003 | 51:86 | NORMAL_TEXT | LIST id=kix.mkvjinnwd60x level=0]
Total RMSE + spatial maps of RMSE:

[P00004 | 86:98 | NORMAL_TEXT | LIST id=kix.s8o3xva5xucg level=0]
Velocities:

[P00005 | 98:139 | NORMAL_TEXT | LIST id=kix.s8o3xva5xucg level=1]
Global: EKE spatial and temporal spectra

[P00006 | 139:185 | NORMAL_TEXT | LIST id=kix.s8o3xva5xucg level=1]
Regional diagnostics: Gulf Stream and Agulhas

[P00007 | 185:198 | NORMAL_TEXT | LIST id=kix.s8o3xva5xucg level=0]
Temperature:

[P00008 | 198:231 | NORMAL_TEXT | LIST id=kix.s8o3xva5xucg level=1]
SST spatial and temporal spectra

[P00009 | 231:269 | NORMAL_TEXT | LIST id=kix.s8o3xva5xucg level=1]
OHC: total 700-2000 m and upper 700 m

[P00010 | 269:270 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00011 | 270:281 | NORMAL_TEXT | LIST id=kix.l97m3krt0rp4 level=0]
Anomalies:

[P00012 | 281:362 | NORMAL_TEXT | LIST id=kix.l97m3krt0rp4 level=1]
Computed on time series for SST, EKE, and upper-700 m OHC over the whole rollout

[P00013 | 362:386 | NORMAL_TEXT | LIST id=kix.l97m3krt0rp4 level=2]
Linear trend comparison

[P00014 | 386:475 | NORMAL_TEXT | LIST id=kix.l97m3krt0rp4 level=2]
Variance of anomalies relative to trend and seasonal cycle: detrended and deseasonalized

[P00015 | 475:537 | NORMAL_TEXT | LIST id=kix.xcpyx7w1q2yi level=0]
Maps of residual anomaly variance for SST and upper-700 m OHC

[P00016 | 537:538 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00017 | 538:560 | HEADING_1]
2. Metric Definitions

[P00018 | 560:561 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00019 | 561:696 | NORMAL_TEXT]
The code implementation of these metrics is here: [https://github.com/YuanYuan98/Ocean_Emulator/blob/main/viz_jupyter/obs_evaluation.py](https://github.com/YuanYuan98/Ocean_Emulator/blob/main/viz_jupyter/obs_evaluation.py)

[P00020 | 696:697 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00021 | 697:866 | NORMAL_TEXT]
All metrics are evaluated on matched model-observation times and finite ocean cells only. Area weights A_i are used whenever a spatial field is collapsed to one scalar.

[P00022 | 866:867 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00023 | 867:900 | HEADING_2]
2.1. Notation and Area Weighting

[P00024 | 900:902 | NORMAL_TEXT]
[INLINE_OBJECT kix.1dofdldwxo00]

[P00025 | 902:1073 | NORMAL_TEXT]
M and O are model and observation fields after the common time window, grid alignment, and diagnostic mask. e_i(t) is always model minus observation at cell i and time t.

[P00026 | 1073:1074 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00027 | 1074:1112 | HEADING_2]
2.2. Total RMSE and Spatial RMSE Maps

[P00028 | 1112:1114 | NORMAL_TEXT]
[INLINE_OBJECT kix.99ce4iu88cv8]

[P00029 | 1114:1310 | NORMAL_TEXT]
Spatial RMSE maps are computed independently at every grid cell. Total RMSE is the area-weighted RMS of that map. The RMSE map figures use the 2022 overlap window for velocity, EKE, SST, and OHC.

[P00030 | 1310:1311 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00031 | 1311:1338 | HEADING_2]
2.3. Velocity RMSE and EKE

[P00032 | 1338:1340 | NORMAL_TEXT]
[INLINE_OBJECT kix.aa2buw5zkjf7]

[P00033 | 1340:1533 | NORMAL_TEXT]
Velocity vector RMSE uses both surface geostrophic components. DUACS geostrophic velocity comparisons mask |lat| < 5 degrees. EKE is computed from velocity anomalies relative to the time mean.

[P00034 | 1533:1534 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00035 | 1534:1568 | HEADING_2]
2.4. Spatial and Temporal Spectra

[P00036 | 1568:1570 | NORMAL_TEXT]
[INLINE_OBJECT kix.dby45f16sj4e]

[P00037 | 1570:1852 | NORMAL_TEXT]
Spatial spectra are computed for ocean basins/regions after linear spatial detrending and Hann windowing, then radially averaged to isotropic spectra. The current regional diagnostics are Gulf Stream and Agulhas. Temporal spectra use Welch PSD with linear detrending, Hann windows.

[P00038 | 1852:1853 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00039 | 1853:1870 | HEADING_2]
2.5. SST and OHC

[P00040 | 1870:1872 | NORMAL_TEXT]
[INLINE_OBJECT kix.u3vjoqa2px11]

[P00041 | 1872:2102 | NORMAL_TEXT]
SST RMSE uses the 2022 OISST overlap after grid alignment. SST spectra use SST anomalies relative to calendar-day climatology. OHC is computed per unit area for 0-700 m and 700-2000 m and then summed to global totals when needed.

[P00042 | 2102:2103 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00043 | 2103:2149 | HEADING_2]
2.6. Anomalies, Trends, and Residual Variance

[P00044 | 2149:2151 | NORMAL_TEXT]
[INLINE_OBJECT kix.gdaxt8k1q6e8]

[P00045 | 2151:2153 | NORMAL_TEXT]
[INLINE_OBJECT kix.a6cx8uftkl6]

[P00046 | 2153:2536 | NORMAL_TEXT]
Time-series anomalies are compared over the whole rollout. SST and EKE keep 5-day cadence; upper-700 m OHC is monthly. The residual variance is the metric of interest: it is computed after removing both the seasonal cycle and a linear trend. For <=10-day cadence, the seasonal cycle uses pentad climatology when repeated pentads are available; otherwise monthly climatology is used.

[P00047 | 2536:2537 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00048 | 2537:2565 | HEADING_2]
2.7. Residual Variance Maps

[P00049 | 2565:2567 | NORMAL_TEXT]
[INLINE_OBJECT kix.cearnnhm4165]

[P00050 | 2567:2873 | NORMAL_TEXT]
The same residual processing is applied independently at each grid cell. SST residual variance maps are computed on native model grids and interpolated to OISST. Upper-700 m OHC residual variance maps use monthly per-area OHC on the ARGO-IAP grid. Pattern correlation is computed over finite paired cells.

[P00051 | 2873:2874 | HEADING_1]
⟦EMPTY PARAGRAPH⟧

[P00052 | 2874:2889 | HEADING_1]
3. Key Metrics

[P00053 | 2889:2890 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00054 | 2893:2900 | NORMAL_TEXT | TABLE row=0 col=0]
Metric

[P00055 | 2901:2911 | NORMAL_TEXT | TABLE row=0 col=1]
Samudra 2

[P00056 | 2912:2916 | NORMAL_TEXT | TABLE row=0 col=2]
OM4

[P00057 | 2918:2960 | NORMAL_TEXT | TABLE row=1 col=0]
Velocity vector total RMSE, 2022 (m s^-1)

[P00058 | 2961:2968 | NORMAL_TEXT | TABLE row=1 col=1]
0.2127

[P00059 | 2969:2976 | NORMAL_TEXT | TABLE row=1 col=2]
0.2117

[P00060 | 2978:3022 | NORMAL_TEXT | TABLE row=2 col=0]
Surface EKE map total RMSE, 2022 (m^2 s^-2)

[P00061 | 3023:3031 | NORMAL_TEXT | TABLE row=2 col=1]
0.02594

[P00062 | 3032:3040 | NORMAL_TEXT | TABLE row=2 col=2]
0.02504

[P00063 | 3042:3071 | NORMAL_TEXT | TABLE row=3 col=0]
SST total RMSE, 2022 (deg C)

[P00064 | 3072:3079 | NORMAL_TEXT | TABLE row=3 col=1]
0.9768

[P00065 | 3080:3087 | NORMAL_TEXT | TABLE row=3 col=2]
0.7923

[P00066 | 3089:3137 | NORMAL_TEXT | TABLE row=4 col=0]
OHC per-area total RMSE, 0-700 m, 2022 (J m^-2)

[P00067 | 3138:3148 | NORMAL_TEXT | TABLE row=4 col=1]
2.569e+09

[P00068 | 3149:3159 | NORMAL_TEXT | TABLE row=4 col=2]
2.608e+09

[P00069 | 3161:3212 | NORMAL_TEXT | TABLE row=5 col=0]
OHC per-area total RMSE, 700-2000 m, 2022 (J m^-2)

[P00070 | 3213:3223 | NORMAL_TEXT | TABLE row=5 col=1]
2.262e+09

[P00071 | 3224:3234 | NORMAL_TEXT | TABLE row=5 col=2]
2.265e+09

[P00072 | 3236:3279 | NORMAL_TEXT | TABLE row=6 col=0]
SST residual variance map RMSE ((deg C)^2)

[P00073 | 3280:3286 | NORMAL_TEXT | TABLE row=6 col=1]
0.427

[P00074 | 3287:3294 | NORMAL_TEXT | TABLE row=6 col=2]
0.3336

[P00075 | 3296:3342 | NORMAL_TEXT | TABLE row=7 col=0]
SST residual variance map pattern correlation

[P00076 | 3343:3350 | NORMAL_TEXT | TABLE row=7 col=1]
0.6268

[P00077 | 3351:3358 | NORMAL_TEXT | TABLE row=7 col=2]
0.7027

[P00078 | 3360:3415 | NORMAL_TEXT | TABLE row=8 col=0]
Upper-700m OHC residual variance map RMSE ((J m^-2)^2)

[P00079 | 3416:3426 | NORMAL_TEXT | TABLE row=8 col=1]
1.139e+18

[P00080 | 3427:3437 | NORMAL_TEXT | TABLE row=8 col=2]
1.314e+18

[P00081 | 3439:3496 | NORMAL_TEXT | TABLE row=9 col=0]
Upper-700m OHC residual variance map pattern correlation

[P00082 | 3497:3504 | NORMAL_TEXT | TABLE row=9 col=1]
0.6236

[P00083 | 3505:3512 | NORMAL_TEXT | TABLE row=9 col=2]
0.6651

[P00084 | 3513:3514 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00085 | 3514:3599 | NORMAL_TEXT]
| Metric                                                   | Samudra 2 | OM4       |

[P00086 | 3599:3684 | NORMAL_TEXT]
|----------------------------------------------------------|-----------|-----------|

[P00087 | 3684:3769 | NORMAL_TEXT]
| Velocity vector total RMSE, 2022 (m s^-1)                | 0.2127    | 0.2117    |

[P00088 | 3769:3854 | NORMAL_TEXT]
| Surface EKE map total RMSE, 2022 (m^2 s^-2)              | 0.02594   | 0.02504   |

[P00089 | 3854:3939 | NORMAL_TEXT]
| SST total RMSE, 2022 (deg C)                             | 0.9768    | 0.7923    |

[P00090 | 3939:4024 | NORMAL_TEXT]
| OHC per-area total RMSE, 0-700 m, 2022 (J m^-2)          | 2.569e+09 | 2.608e+09 |

[P00091 | 4024:4109 | NORMAL_TEXT]
| OHC per-area total RMSE, 700-2000 m, 2022 (J m^-2)       | 2.262e+09 | 2.265e+09 |

[P00092 | 4109:4194 | NORMAL_TEXT]
| SST residual variance map RMSE ((deg C)^2)               | 0.427     | 0.3336    |

[P00093 | 4194:4279 | NORMAL_TEXT]
| SST residual variance map pattern correlation            | 0.6268    | 0.7027    |

[P00094 | 4279:4364 | NORMAL_TEXT]
| Upper-700m OHC residual variance map RMSE ((J m^-2)^2)   | 1.139e+18 | 1.314e+18 |

[P00095 | 4364:4449 | NORMAL_TEXT]
| Upper-700m OHC residual variance map pattern correlation | 0.6236    | 0.6651    |

[P00096 | 4449:4460 | HEADING_1]
4. Results

[P00097 | 4460:4503 | HEADING_2]
4.1. Total RMSE + Spatial RMSE Maps (2022)

[P00098 | 4503:4567 | HEADING_3]
Figure 4.1.1. Surface Geostrophic Velocity Vector RMSE vs DUACS

[P00099 | 4567:4569 | NORMAL_TEXT]
[INLINE_OBJECT kix.6a9nvq51bfrn]

[P00100 | 4569:4570 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00101 | 4570:4618 | HEADING_3]
Figure 4.1.2. Surface EKE Spatial RMSE vs DUACS

[P00102 | 4618:4620 | NORMAL_TEXT]
[INLINE_OBJECT kix.6b3xe4gh2kbx]

[P00103 | 4620:4657 | HEADING_3]
Figure 4.1.3. SST RMSE Maps vs OISST

[P00104 | 4657:4659 | NORMAL_TEXT]
[INLINE_OBJECT kix.8x9t28dwzsmw]

[P00105 | 4659:4708 | HEADING_3]
Figure 4.1.4. OHC Per-Area RMSE Maps vs ARGO-IAP

[P00106 | 4708:4710 | NORMAL_TEXT]
[INLINE_OBJECT kix.5k726gbytmub]

[P00107 | 4710:4726 | HEADING_2]
4.2. Velocities

[P00108 | 4726:4767 | NORMAL_TEXT | LIST id=kix.h0xs816frz7c level=0]
Global: EKE spatial and temporal spectra

[P00109 | 4767:4813 | NORMAL_TEXT | LIST id=kix.h0xs816frz7c level=0]
Regional diagnostics: Gulf Stream and Agulhas

[P00110 | 4813:4857 | HEADING_3]
Figure 4.2.1. Mean Surface EKE Map vs DUACS

[P00111 | 4857:4859 | NORMAL_TEXT]
[INLINE_OBJECT kix.4otnnoxrv3w0]

[P00112 | 4859:4932 | HEADING_3]
Figure 4.2.2. Surface EKE Spatial Spectra (Global, Gulf Stream, Agulhas)

[P00113 | 4932:4934 | NORMAL_TEXT]
[INLINE_OBJECT kix.sycz8ymfs1df]

[P00114 | 4934:5019 | HEADING_3]
Figure 4.2.3. Surface Geostrophic KE Temporal Spectra (Global, Gulf Stream, Agulhas)

[P00115 | 5019:5021 | NORMAL_TEXT]
[INLINE_OBJECT kix.dzcn80mautas]

[P00116 | 5021:5022 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00117 | 5022:5039 | HEADING_2]
4.3. Temperature

[P00118 | 5039:5072 | NORMAL_TEXT | LIST id=kix.rz77i1ruf9ez level=0]
SST spatial and temporal spectra

[P00119 | 5072:5110 | NORMAL_TEXT | LIST id=kix.rz77i1ruf9ez level=0]
OHC: total 700-2000 m and upper 700 m

[P00120 | 5110:5175 | HEADING_3]
Figure 4.3.1. SST Spatial Spectra (Global, Gulf Stream, Agulhas)

[P00121 | 5175:5177 | NORMAL_TEXT]
[INLINE_OBJECT kix.jnnm8negzv74]

[P00122 | 5177:5244 | HEADING_3]
Figure 4.3.2. SST Isotropic Spectra (Global, Gulf Stream, Agulhas)

[P00123 | 5244:5246 | NORMAL_TEXT]
[INLINE_OBJECT kix.j6ro8x3clr5k]

[P00124 | 5246:5290 | HEADING_3]
Figure 4.3.3. Monthly Total OHC vs ARGO-IAP

[P00125 | 5290:5292 | NORMAL_TEXT]
[INLINE_OBJECT kix.si64u3fw423z]

[P00126 | 5292:5343 | HEADING_3]
Figure 4.3.4. Annual OHC Per-Area Maps vs ARGO-IAP

[P00127 | 5343:5345 | NORMAL_TEXT]
[INLINE_OBJECT kix.ujhikijn9enx]

[P00128 | 5345:5401 | HEADING_3]
Figure 4.3.5. Annual OHC Per-Area Bias Maps vs ARGO-IAP

[P00129 | 5401:5403 | NORMAL_TEXT]
[INLINE_OBJECT kix.smdwlbat56v7]

[P00130 | 5403:5418 | HEADING_2]
4.4. Anomalies

[P00131 | 5418:5499 | NORMAL_TEXT | LIST id=kix.e9ao66dq98nf level=0]
Computed on time series for SST, EKE, and upper-700 m OHC over the whole rollout

[P00132 | 5499:5523 | NORMAL_TEXT | LIST id=kix.e9ao66dq98nf level=0]
Linear trend comparison

[P00133 | 5523:5612 | NORMAL_TEXT | LIST id=kix.e9ao66dq98nf level=0]
Variance of anomalies relative to trend and seasonal cycle: detrended and deseasonalized

[P00134 | 5612:5674 | NORMAL_TEXT | LIST id=kix.e9ao66dq98nf level=0]
Maps of residual anomaly variance for SST and upper-700 m OHC

[P00135 | 5674:5723 | HEADING_3]
Figure 4.4.1. Global SST Linear Trend Comparison

[P00136 | 5723:5725 | NORMAL_TEXT]
[INLINE_OBJECT kix.t3m3m26ojqnp]

[P00137 | 5725:5787 | HEADING_3]
Figure 4.4.2. Global SST Detrended and Deseasonalized Anomaly

[P00138 | 5787:5789 | NORMAL_TEXT]
[INLINE_OBJECT kix.j14q9lfnm8f0]

[P00139 | 5789:5840 | HEADING_3]
Figure 4.4.3. Global SST Residual Anomaly Variance

[P00140 | 5840:5842 | NORMAL_TEXT]
[INLINE_OBJECT kix.k7zks8q6udvj]

[P00141 | 5842:5891 | HEADING_3]
Figure 4.4.4. Global EKE Linear Trend Comparison

[P00142 | 5891:5893 | NORMAL_TEXT]
[INLINE_OBJECT kix.53zdnbm6j3k7]

[P00143 | 5893:5955 | HEADING_3]
Figure 4.4.5. Global EKE Detrended and Deseasonalized Anomaly

[P00144 | 5955:5957 | NORMAL_TEXT]
[INLINE_OBJECT kix.1eyvm0yrexo5]

[P00145 | 5957:6008 | HEADING_3]
Figure 4.4.6. Global EKE Residual Anomaly Variance

[P00146 | 6008:6010 | NORMAL_TEXT]
[INLINE_OBJECT kix.k09f2vw7cmrg]

[P00147 | 6010:6011 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00148 | 6011:6071 | HEADING_3]
Figure 4.4.7. Global Upper-700m OHC Linear Trend Comparison

[P00149 | 6071:6073 | NORMAL_TEXT]
[INLINE_OBJECT kix.94t950ftidoe]

[P00150 | 6073:6146 | HEADING_3]
Figure 4.4.8. Global Upper-700m OHC Detrended and Deseasonalized Anomaly

[P00151 | 6146:6148 | NORMAL_TEXT]
[INLINE_OBJECT kix.o3vhwfcn1jv1]

[P00152 | 6148:6210 | HEADING_3]
Figure 4.4.9. Global Upper-700m OHC Residual Anomaly Variance

[P00153 | 6210:6212 | NORMAL_TEXT]
[INLINE_OBJECT kix.wyzlpipq7xdi]

[P00154 | 6212:6262 | HEADING_3]
Figure 4.4.10. SST Residual Anomaly Variance Maps

[P00155 | 6262:6264 | NORMAL_TEXT]
[INLINE_OBJECT kix.ci1y3onokfok]

[P00156 | 6264:6325 | HEADING_3]
Figure 4.4.11. Upper-700m OHC Residual Anomaly Variance Maps

[P00157 | 6325:6327 | NORMAL_TEXT]
[INLINE_OBJECT kix.rpxfw4432jjv]

[P00158 | 6327:6328 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

