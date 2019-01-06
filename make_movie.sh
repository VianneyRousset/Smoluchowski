#!/usr/bin/env bash
if [ -z $1 ]; then
	echo "Usage $0 images" 
fi
cat ${@} | ffmpeg -y -f image2pipe -i - -c:v libx264 -vf "fps=25,format=yuv420p" output.mp4
