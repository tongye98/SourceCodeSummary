#!/bin/sh
git status
git add *
git commit -m "add some code from g2"
git pull --rebase origin main
git push origin main
