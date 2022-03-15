# nedem

Implicitly represent a digital elevation model as the weights of a neural network.

Benefits
- Flexible up/down scaling of the model's parameters to increase/decrease details
- Weighted sampling during training to focus on details in regions of interest

There is prior work to do this for 3d models; we try to answer the question if it is possible and reasonable for a digital elevation model.


## Quadkeys

We use quadkeys as the unit of abstraction. They can be generated from an arbitrary raster file with `rio` from `rasterio`

    rio warp in.tif 3857.tif --dst-crs EPSG:3857
    rio bounds 3857.tif | mercantile tiles 10 > z10.txt

    while read tile; do
      rio clip 3857.tif $(mercantile quadkey "$tile").tif --with-complement --bounds "$(mercantile shapes --extents --mercator "$tile")"
    done < z10.txt


## Hillshade

The digital elevation model is best visualize as a hillshaded tif; you can use `gdaldem` from `gdal-bin` for hill shading

    for p in step-*.tif; do gdaldem hillshade $p -multidirectional -co compress=deflate -co predictor=2 hillshade-$p ; done

The model fitting happens over multiple steps; you can use `ffmpeg` to animate the hill shaded tifs

    ffmpeg -y -loglevel error -r 5 -f image2 -pattern_type glob -i "hillshade-step-*.tif" -vf 'crop=w=768:h=768' -c:v libx264 -crf 23 -profile:v high -preset veryslow -pix_fmt yuv420p -movflags faststart hillshade-step.mp4


## References

- https://arxiv.org/abs/2009.09808
- https://registry.opendata.aws/copernicus-dem/


## License

Copyright © 2022 Daniel J. Hofmann

Distributed under the MIT License (MIT).
