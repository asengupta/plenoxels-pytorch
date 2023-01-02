ffmpeg -framerate 30 -pattern_type glob -i './code/pytorch-learn/plenoxels/random-images/frames/*.png' -c:v libx264 -pix_fmt yuv420p out.mp4
