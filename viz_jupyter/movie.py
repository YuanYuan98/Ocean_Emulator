# ============================================================
# Animated Movies (GIF) — one frame per year
# ============================================================
# print("\n" + "="*60)
# print("Animated Movies (GIF)")
# print("="*60)

# GIF_DIR = os.path.join(OUTPUT_DIR, "gif_movies")
# os.makedirs(GIF_DIR, exist_ok=True)

# # OHC Snapshot Bias Movie
# movie_ohc_snapshot_bias_together(
#     emulators_to_analyze,
#     output_path=os.path.join(GIF_DIR, "movie_ohc_snapshot_bias.gif"),
#     colorbar_fix=True,
# )

# # Temperature Snapshot Bias Movie
# movie_temperature_snapshot_bias_together(
#     emulators_to_analyze,
#     output_path=os.path.join(GIF_DIR, "movie_temperature_snapshot_bias.gif"),
#     colorbar_fix=True,
# )

# # Zonal Bias by Ocean Basin Movie
# movie_snapshot_zonal_bias_ocean_together(
#     emulators_to_analyze,
#     output_path=os.path.join(GIF_DIR, "movie_snapshot_zonal_bias_thetao.gif"),
#     variable_name='thetao',
#     units='°C',
#     masks_list=masks_list,
#     colorbar_fix=True,
# )

# # OHC Zonal Bias Movie
# movie_snapshot_ohc_zonal_bias_ocean_together(
#     emulators_to_analyze,
#     output_path=os.path.join(GIF_DIR, "movie_snapshot_ohc_zonal_bias.gif"),
#     masks_list=masks_list,
#     colorbar_fix=True,
# )

# # Zonal Snapshot Comparison Movie
# movie_zonal_snapshot_together(
#     emulators_to_analyze,
#     output_path=os.path.join(GIF_DIR, "movie_zonal_snapshot_thetao.gif"),
#     variable_name='thetao',
#     units='$^\\circ$C',
#     masks_list=masks_list,
#     colorbar_fix=True,
# )

# # OHC Zonal Snapshot Movie
# movie_ohc_zonal_snapshot_together(
#     emulators_to_analyze,
#     output_path=os.path.join(GIF_DIR, "movie_ohc_zonal_snapshot.gif"),
#     masks_list=masks_list,
#     colorbar_fix=True,
# )

# Multi-Depth Temperature Snapshot Movie

# MOVIE_FPS = {'yearly': 2, 'monthly': 8, 'all': 50}
# MOVIE_DPI = {'yearly': 200, 'monthly': 200, 'all': 200}

# MOVIE_FREQ = args.freq


# if args.var == 'temperature':
#     print(f"  -> Generating SST tropical Pacific movie ({MOVIE_FREQ})...")
#     movie_sst_tropical_pacific_together(
#         emulators_to_analyze,
#         output_path=os.path.join(GIF_DIR, f"movie_sst_tropical_pacific_{MOVIE_FREQ}_{MOVIE_FPS[MOVIE_FREQ]}.mp4"),
#         var_name='thetao',
#         depth_index=0,
#         lon_range=(-230, -70),
#         lat_range=(-30, 35),
#         fix_colorbar=True,
#         vmin=18,
#         vmax=32,
#         end_idx=None,
#         freq=MOVIE_FREQ,
#         fps=MOVIE_FPS[MOVIE_FREQ],
#         dpi=MOVIE_DPI[MOVIE_FREQ],
#     )

#     exit()

#     print(f"  -> Generating temperature movie ({MOVIE_FREQ})...")
#     movie_snapshot_multidepth_together(
#         emulators_to_analyze,
#         output_path=os.path.join(GIF_DIR, f"movie_snapshot_multidepth_thetao_{MOVIE_FREQ}_{MOVIE_FPS[MOVIE_FREQ]}.gif"),
#         var_name='thetao',
#         depth_configs=DEPTH_LAYERS,
#         fix_colorbar=True,
#         end_idx=None,
#         freq=MOVIE_FREQ,
#         fps=MOVIE_FPS[MOVIE_FREQ],
#         dpi=MOVIE_DPI[MOVIE_FREQ]
#     )

# elif args.var =='ohc':
#     # OHC Multi-Depth Snapshot Movie
#     print(f"  -> Generating OHC movie ({MOVIE_FREQ})...")
#     movie_ohc_snapshot_multidepth_together(
#         emulators_to_analyze,
#         output_path=os.path.join(GIF_DIR, f"movie_ohc_snapshot_multidepth_{MOVIE_FREQ}.gif"),
#         depth_configs=DEPTH_SLICES,
#         colorbar_fix=True,
#         end_idx=None,
#         freq=MOVIE_FREQ,
#         fps=MOVIE_FPS[MOVIE_FREQ],
#         dpi=MOVIE_DPI[MOVIE_FREQ]
#     )
