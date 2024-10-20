#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# draw the persistence image of a molecule

from PH import PI_vector_h0h1h2
import collections
collections.Iterable = collections.abc.Iterable

pixel_m = 40
pixel_n = pixel_m

sigma = 0.002
Max = 2.5

M = PI_vector_h0h1h2('mols/aspirin.xyz', pixelx=pixel_m, pixely=pixel_m, myspread=sigma, myspecs={"maxBD": Max, "minBD":-.10}, showplot=True)

