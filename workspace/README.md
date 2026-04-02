# Workspace Layout

Yerel calisma dosyalari `workspace/` altinda tutulur.

- `workspace/models/`: ONNX, engine, `.pt` ve benzeri agir model dosyalari
- `workspace/videos/input/`: test ve demo giris videolari
- `workspace/outputs/`: uretilen video, resim, CSV ve JSON ciktilari
- `workspace/outputs/python_test/`: `python_test.py` kayitlari
- `workspace/outputs/tensorrt_test/`: `tensorrt_test.py` ciktilari
- `workspace/logs/`: cron ve yerel log dosyalari
- `workspace/cache/`: yeniden uretilmesi kolay cache dosyalari
- `workspace/scripts/archive/`: eski kopyalar ve yedek scriptler
- `workspace/notes/`: yerel notlar

Bu klasor `git` tarafinda buyuk olcude ignore edilir; sadece bu aciklama dosyasi izlenir.
