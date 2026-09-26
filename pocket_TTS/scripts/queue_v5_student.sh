#!/bin/bash
# Launch the v5 student as soon as the v5 teacher finishes: wait for the teacher
# process to exit and for the 250k listen clips from eval_v5_teacher.sh (one GPU).
# Refuses to distill from anything but the final checkpoint, so a crashed or
# early-stopped teacher does not silently become the distillation source.
set -u
R=/root/tts/TTS_training/pocket_TTS
C=/workspace/v5_teacher_24l/checkpoint_00250000.pt
say() { echo "[$(date '+%F %H:%M')] $*"; }
say "waiting for teacher pid $(cat /tmp/v5_teacher.pid)"
while kill -0 $(cat /tmp/v5_teacher.pid) 2>/dev/null; do sleep 300; done
[ -f $C ] || { say "teacher exited without $C; NOT launching student"; exit 1; }
say "teacher done; waiting for listen loop"
while kill -0 $(cat /tmp/v5_teacher_eval.pid) 2>/dev/null; do sleep 60; done
say "launching student"
$R/scripts/train_student_v5.sh > $R/train_student_v5.log 2>&1 &
echo $! > /tmp/v5_student.pid
sleep 600
grep -q "step 500 " $R/train_student_v5.log && say "student running: $(grep 'step 500 ' $R/train_student_v5.log | cut -c1-110)" \
  || { say "student not at step 500 after 10 min:"; tail -5 $R/train_student_v5.log; }
