#!/usr/bin/env bash
if [ -z $1 ]; then
	echo "Usage $0 images" 
fi
cat ${@} | ffmpeg -y -f image2pipe -i - output.mp4
