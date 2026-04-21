#!/usr/bin/env bash
# Quick status checker for Euro NCAP downloads

echo "======================================"
echo "Euro NCAP Download Status"
echo "======================================"
echo ""

VIDEO_DIR="test_data/euroncap_source/videos"
TARGET=100

if [[ -d "$VIDEO_DIR" ]]; then
    CURRENT=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l | xargs)
    echo "Downloaded: $CURRENT / $TARGET videos"
    echo "Progress: $(( CURRENT * 100 / TARGET ))%"
    echo ""

    if [[ $CURRENT -gt 0 ]]; then
        TOTAL_SIZE=$(du -sh "$VIDEO_DIR" | cut -f1)
        echo "Total size: $TOTAL_SIZE"
        echo ""
        echo "Latest 3 downloads:"
        ls -lhtr "$VIDEO_DIR"/*.mp4 2>/dev/null | tail -3 | awk '{print "  " $9 " (" $5 ")"}'
    fi
else
    echo "Download not started yet (video directory doesn't exist)"
fi

echo ""
echo "======================================"
echo "To monitor live:"
echo "  watch -n 5 ./check_download_status.sh"
echo "======================================"
