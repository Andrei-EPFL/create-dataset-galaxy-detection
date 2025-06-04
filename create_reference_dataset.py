import os
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import tifffile
from astropy.coordinates import Angle  # Angles
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.wcs import WCS

from desi import DESILegacySurvey


def query_single_image(
    RA=25.8317,
    DEC=9.7475,
    Radius=0.5,
    pixsize=0.262,
    band="grz",
    data_release=10,
    psf_width=1.5,  # To be checked
):
    ra_s = RA
    dec_s = DEC
    image_size = Radius
    image_band = band
    dr = data_release
    image_size = Angle(image_size * u.arcmin)
    pixsize = Angle(pixsize * u.arcsec)
    npix = int(2 * image_size / pixsize)
    source = SkyCoord(ra_s, dec_s, unit="degree")
    rad_a = Angle(Radius * u.arcmin)

    query_i_raw = DESILegacySurvey.get_images(
        position=source,
        survey="dr%d" % dr,
        coordinates="icrs",
        data_release=dr,
        pixels=npix,
        radius=image_size,
        image_band=image_band,
    )

    hdu_i_raw = query_i_raw[0][0]
    image_i_raw = hdu_i_raw.data  # Three channels, one per band
    w = WCS(hdu_i_raw.header)

    query_c = DESILegacySurvey.query_region(
        coordinates=source,
        radius=rad_a,
        data_release=dr,
    )

    s_r = query_c["shape_r"]
    s_e1 = query_c["shape_e1"]
    s_e2 = query_c["shape_e2"]
    ra_c = query_c["ra"]
    dec_c = query_c["dec"]
    type_c = query_c["type"]

    coord = SkyCoord(ra_c, dec_c, unit="degree")
    x, y, _ = w.world_to_pixel(coord, u.Quantity(10))
    
    epsilon = np.sqrt(s_e1**2 + s_e2**2)
    b_o_a = (1 - epsilon) / (1 + epsilon)
    phi = (0.5 * np.arctan(s_e2 / s_e1)) * (180 / np.pi)

    records = []
    for i, (ra_1, dec_1, s_r_1, t_1, b_o_a_1, phi_1, x_1, y_1) in enumerate(
        zip(ra_c, dec_c, s_r + psf_width, type_c, b_o_a, phi, x, y)
    ):
        if s_e1[i] < 0 and s_e2[i] < 0:
            a_phi_1 = -phi_1
        elif s_e1[i] < 0 and s_e2[i] > 0:
            a_phi_1 = -phi_1
        elif s_e1[i] > 0 and s_e2[i] > 0:
            a_phi_1 = 90 - phi_1
        elif s_e1[i] > 0 and s_e2[i] < 0:
            a_phi_1 = 90 - phi_1
        else:
            a_phi_1 = phi_1

        if t_1 in ["PSF", "REX"]:
            width = s_r_1 / 3600
            height = s_r_1 / 3600
            angle = 0
        else:
            width = s_r_1 / 1800
            height = b_o_a_1 * s_r_1 / 1800
            angle = a_phi_1

        records.append(
            {
                "shape_r": s_r_1,
                "ra": ra_1,  # = x
                "dec": dec_1,  # = y
                "type": t_1,
                "width": width,
                "height": height,
                "angle": angle,
                "idx_x_ra": int(x_1),
                "idx_y_dec": int(y_1)
            }
        )

    ellipses_df = pd.DataFrame.from_records(records)

    image_meta_df = dict(hdu_i_raw.header)
    image_meta_df["RA"] = RA
    image_meta_df["DEC"] = DEC
    image_meta_df["Radius"] = Radius
    image_meta_df["pixsize"] = pixsize
    image_meta_df["band"] = band
    image_meta_df["data_release"] = data_release
    image_meta_df["psf_width"] = psf_width
    image_meta_df = pd.DataFrame(image_meta_df)

    image_xyc = np.transpose(image_i_raw, axes=[1, 2, 0])  # XYC order

    return (image_xyc, ellipses_df, image_meta_df)


if __name__ == "__main__":
    # Run the script (example, to save 10 images in the out_dir `dataset/`): python create_reference_dataset.py dataset/ 10
    _, out_dir, n_images = sys.argv
    n_images = int(n_images)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f"Created folder: {Path(out_dir).resolve()}")

    min_ra = 0
    max_ra = 90
    min_dec = -90
    max_dec = 10

    # Perhaps we should fix a seed or do a regular grid survey, so that running the script to generate the reference dataset is reproducible
    # random generation of ra and dec
    # ras = np.random.random(n_images) * (max_ra - min_ra) + min_ra
    # decs = np.random.random(n_images) * (max_dec - min_dec) + min_dec

    delta_ra = (max_ra - min_ra) / n_images
    delta_dec = (max_dec - min_dec) / n_images

    ra = min_ra
    dec = min_dec
    for idx in range(n_images):
        ra = ra + delta_ra
        dec = dec + delta_dec

        image_xyc, ellipses_df, image_meta_df = query_single_image(
            RA=ra,
            DEC=dec,
            Radius=0.5,
            pixsize=0.262,
            band="grz",
            data_release=10,
            psf_width=1.5,  # To check
        )

        image_dir = Path(out_dir) / f"{idx:03d}"
        
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
            print(f"Created folder: {image_dir.resolve()}")

        # Save the image
        image_out_path = image_dir / "image.tif"
        tifffile.imwrite(image_out_path, image_xyc)
        print(f"Saved: {image_out_path.resolve()}")

        # Save the image metadata
        image_metadata_path = image_dir / "metadata.json"
        image_meta_df.to_json(image_metadata_path, indent=4)
        print(f"Saved: {image_metadata_path.resolve()}")

        # Save the ellipses detection
        ellipses_csv_path = image_dir / "ellipses.csv"
        ellipses_df.to_csv(ellipses_csv_path)
        print(f"Saved: {ellipses_csv_path.resolve()}")
