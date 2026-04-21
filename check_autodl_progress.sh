#!/usr/bin/env bash
# Monitor AutoDL download and processing progress

AUTODL_HOST="connect.cqa1.seetacloud.com"
AUTODL_PORT="11224"
AUTODL_USER="root"
AUTODL_PASS="VhKOq2k0uGlQ"

echo "=========================================="
echo " AutoDL Progress Monitor"
echo "=========================================="
echo ""

# Check video download count
VIDEO_COUNT=$(sshpass -p "$AUTODL_PASS" ssh -p $AUTODL_PORT -o StrictHostKeyChecking=no \
    $AUTODL_USER@$AUTODL_HOST \
    "ls /root/autodl-tmp/clipscore_experiment/test_data/euroncap_source/videos/*.mp4 2>/dev/null | wc -l")

echo "Videos downloaded: $VIDEO_COUNT / 100"
echo ""

# Check if queue bundle exists
QUEUE_EXISTS=$(sshpass -p "$AUTODL_PASS" ssh -p $AUTODL_PORT -o StrictHostKeyChecking=no \
    $AUTODL_USER@$AUTODL_HOST \
    "test -f /root/autodl-tmp/clipscore_experiment/test_data/euroncap_source/queue_seed.json && echo 'YES' || echo 'NO'")

if [[ "$QUEUE_EXISTS" == "YES" ]]; then
    echo "✓ Queue bundle generated"
else
    echo "⏳ Queue bundle not yet generated"
fi

echo ""
echo "=========================================="
echo "To check detailed logs on AutoDL:"
echo "  ssh -p $AUTODL_PORT $AUTODL_USER@$AUTODL_HOST"
echo "  cd /root/autodl-tmp/clipscore_experiment"
echo "=========================================="
