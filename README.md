# TaLiSSman: TrAnsmitted LIght Stack SegMentAtioN
Segmentation of bacteria growing in agar-pads, imaged by de-focused transmitted light stacks

## Network architecture:
- Based on U-net
- At first layer, Z-axis is both:
  - Considered as channel axis and treated with with 2D convolutions
  - Reduced using 3D convolutions and 3D max-pooling

## How it works
- Expected input images are stacks of 2D images, with Z-axis last : Image = [batch, Y, X, Z]. We advise stacks of 5 slices with a 0.2µm step, in range [-0.6µm, -1.4µm] (relatively to the focal plane)
- Segmentation is performed by regression of the Euclidean Distance Map (EDM).
- This repository does not include the downstream watershed step to obtain labeled images.

| Input transmitted-light Stack | Predicted EDM | Segmented Bacteria |
| :---:         |          :---: |          :---: |
| <img src="wiki/resources/inputStackREV.gif" width="300"> | <img src="wiki/resources/edm.png" width="300">    | <img src="wiki/resources/outputStackREV.gif" width="300"> |
