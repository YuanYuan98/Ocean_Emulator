"""Plot a world map with the six spectrum analysis regions highlighted."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

SPECTRUM_REGIONS = [
    ('Gulf',     slice(300, 320), slice(25, 45)),
    ('Kuroshio', slice(150, 170), slice(25, 45)),
    ('Agulhas',  slice(40, 60),   slice(-50, -30)),
    ('Malvinas', slice(311, 331), slice(-51, -31)),
    ('Niño 3.4',        slice(190, 240), slice(-5, 5)),
    ('Tropical Pacific', slice(130, 290), slice(-30, 30)),
]

# Colors for each region
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))

ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')

legend_patches = []

for (name, lon_slice, lat_slice), color in zip(SPECTRUM_REGIONS, COLORS):
    # Convert from 0-360 to -180..180 for cartopy
    lon_start = lon_slice.start
    lon_stop = lon_slice.stop
    if lon_start > 180:
        lon_start -= 360
    if lon_stop > 180:
        lon_stop -= 360

    lat_start = lat_slice.start
    lat_stop = lat_slice.stop

    # Width and height
    width = lon_stop - lon_start
    height = lat_stop - lat_start

    # Handle wrap-around (e.g., if width becomes negative)
    if width < 0:
        width += 360

    rect = mpatches.Rectangle(
        (lon_start, lat_start),
        width, height,
        linewidth=2.5,
        edgecolor=color,
        facecolor=color,
        alpha=0.25,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(rect)

    # Add label at center of region, with manual offsets for overlapping ones
    center_lon = lon_start + width / 2
    center_lat = lat_start + height / 2
    label_offsets = {
        'Tropical Pacific': (0, 22),   # shift label to top edge
        'Niño 3.4':         (0, 0),  # shift label below center
    }
    dx, dy = label_offsets.get(name, (0, 0))
    ax.text(
        center_lon + dx, center_lat + dy, name,
        transform=ccrs.PlateCarree(),
        ha='center', va='center',
        fontsize=9, fontweight='bold',
        color=color,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8),
    )

    legend_patches.append(mpatches.Patch(
        facecolor=color, edgecolor=color, alpha=0.4,
        label=f'{name} ({lon_slice.start}–{lon_slice.stop}°E, {lat_slice.start}–{lat_slice.stop}°N)',
    ))

ax.legend(handles=legend_patches, loc='lower left', fontsize=8, framealpha=0.9)
ax.set_title('Selected Regions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('region_map.pdf', dpi=200, bbox_inches='tight')
plt.savefig('region_map.png', dpi=200, bbox_inches='tight')
print("Saved region_map.pdf and region_map.png")
