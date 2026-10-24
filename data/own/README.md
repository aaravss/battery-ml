# Your own photos go here

One folder per class (already created):

```
data/own/button_cell/
data/own/aa_aaa_9v/
data/own/cylindrical_18650/
data/own/phone_pouch/
```

## The one rule that matters: one sub-folder per physical battery

```
data/own/button_cell/cr2032_A/IMG_0001.jpg
data/own/button_cell/cr2032_A/IMG_0002.jpg      <- same coin cell, moved/flipped/relit
data/own/button_cell/lr44_A/IMG_0003.jpg
```

Every photo inside one sub-folder is treated as "the same battery", so the whole folder lands in exactly one of
train / validation / test. Without this, near-identical shots of one battery end up on both sides of the split and
the test score is inflated. (Alternative if you don't want sub-folders: name files `cr2032_A__0001.jpg`; the part
before the double underscore is the battery ID.)

Shoot with the real device camera, at the real ~70 mm distance and tray. JPG or PNG, any resolution (the pipeline
pads to square and resizes). Don't crop the photos yourself, because the Pi will see the full frame.

Then re-run everything with one command from the project folder:

```
run_pipeline.bat
```

As soon as `data/own/button_cell/` has photos, button_cell enters validation/test and its metrics stop being
labelled UNRELIABLE.
