# create-dataset-galaxy-detection

## Steps:
1) Install the requirements from: ``` requirements.txt ```
2) Download: ```desi.py``` and ```create_reference_dataset.py```
3) Usage:
```
   python create_reference_dataset.py <OUT DIR> <number of images>
```

## Possible changes:
store the float values of the pixel "index":
```
"idx_x_ra": int(x_1)
"idx_y_dec": int(y_1)
```

to

```
"idx_x_ra": x_1
"idx_y_dec": y_1
```

## Default parameters:

    # The borders of the sky region to insure data access. (could be updated from https://www.legacysurvey.org/status/ info)
    # Units: degrees
    min_ra = 0
    max_ra = 90
    min_dec = -90
    max_dec = 10

    # Worth changing:
    Radius=0.5       # Unit: arcminutes
    band="grz"       # Possible bands (no. channels): g, r, z.  For data release 10 i-band is available as well.

    # Possible change:
    psf_width=1.5    # Unit arcsecond. This is an impiric value, could be updated and improved, but works well like this.

    # Not worth changing:
    pixsize=0.262    # Unit: arcseconds
    data_release=10


Conversion:

    1 degree = 60 arcminutes = 3600 arcseconds
