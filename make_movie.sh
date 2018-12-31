#!/usr/bin/env bash
cat *.png | ffmpeg -y -f image2pipe -i - output.mp4
