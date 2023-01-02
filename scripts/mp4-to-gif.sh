# Convert MP4 to GIF
ffmpeg -i out.mp4 -r 15 output.gif

# Scale GIF
ffmpeg -hide_banner -v warning -i output2.gif -filter_complex "[0:v] scale=320:-1:flags=lanczos,split [a][b]; [a] palettegen=reserve_transparent=on:transparency_color=ffffff [p]; [b][p] paletteuse" output2-large.gif
