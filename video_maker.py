#video_maker.py

from dataHelperTif import *
import moviepy.video.io.ImageSequenceClip

figs = get_files('./som_figures/', '*.png')
figs.sort()

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(figs, fps=2)
clip.write_videofile('som_video.mp4')