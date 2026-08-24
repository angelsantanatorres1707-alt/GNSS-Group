# GNSS Group

A Jupyter interface for exploring GNSS station positions, velocity fields, and
position time series from the NGF (EarthScope), UNR, and JPL analysis centers,
plus local CWU-format files you supply yourself.

The notebook is aimed at geodetic work: station metadata and velocities stay in
view alongside the map, uncertainties are carried through the fits rather than
dropped, and every plotted vector has a stated scale.

## What it does

- **Map** — plot stations and their neighbours within a search radius, draw
  absolute or relative velocity vectors with a screen-fixed magnitude reference,
  colour by vertical rate, and read per-station metadata from the popup or the
  neighbour table.
- **Time series** — plot North/East/Up series for one or many stations, trim by
  epoch window, filter by sigma, detrend, remove outliers, apply catalogued
  break offsets, and stage display shifts.
- **Availability** — check whether a station exists in a given catalog and
  inspect the record stored for it before using it anywhere else.
- **Database Fetcher** — refresh the local catalogs, choose where data is
  stored, and register local files.

## Requirements

Python 3.11 with `ipyleaflet`, `ipywidgets`, `pandas`, `numpy`, `matplotlib`,
`geopy`, `requests`, and `earthscope-sdk`. The notebook installs `ipyleaflet` and
`geopy` on first run if they are missing.

NGF downloads come from the EarthScope archive and need an EarthScope account.
Authenticate once with:

```
es login
```

## Quick start

1. Open `GNSS_Analysis_ipyleaflet.ipynb` and run all cells.
2. In **Database Fetcher**, press **Latest NGF**, **Latest UNR**, or
   **Latest JPL**. The catalogs must be fetched once before any station lookup,
   map plot, or time series will work.
3. Use **Map** to plot stations, **Timeseries** to plot series.

## Data location

Downloaded catalogs are written to a data directory that defaults to `data/`
beside the notebook. Because that default follows the working directory, the
**Data Location** section lets you pick a fixed folder instead:

- browse to a folder and press **Use This Folder**, or type a path,
- or set the `GNSS_DATA_ROOT` environment variable,
- the choice is remembered in `~/.gnss_analysis.json`.

Subfolders `NGF/`, `UNR/`, `JPL/`, and `LOC/` are created inside it.

## Local files

**Add Local File** registers a CWU-format `.csv` position file as a station under
the `LOC` source. The first four characters of the filename become the station
ID, the reference position is read from the file header, and a velocity is fitted
so the station behaves like a catalog station on the map and in the time series.

## Layout

```
GNSS_Analysis_ipyleaflet.ipynb   main interface
gnss_core.py                     pure analysis helpers (fits, geometry, formatting)
gnss_ui.py                       widget palette, stylesheet, and layout components
gnss_map_runtime.py              ipyleaflet layer staging
tests/                           pytest suite and frozen validation fixtures
```

`gnss_core.py` holds no widget or filesystem code, so the numerical parts can be
tested without a kernel.

## Tests

```
python -m pytest -q
```

The suite covers the analysis helpers against an independently derived oracle,
checks the frozen fixtures against recorded digests, and executes the whole
notebook end to end.

## License

MIT — see [LICENSE](LICENSE).

## Data sources

- EarthScope / GAGE (NGF): position, velocity, and offset products
- University of Nevada, Reno (UNR): MIDAS velocities and station positions
- Jet Propulsion Laboratory (JPL): position and velocity tables

Please cite the originating analysis center for any data used in published work.
