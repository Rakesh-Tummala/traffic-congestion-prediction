## Lesson 11: Evaluating the Computer Vision

### Why this lesson exists

The R-squared of 0.977 belongs to the historical model only. For a long time the computer vision part
had no accuracy number at all. This lesson is the measurement of the vision part, found in `cv_eval/`.

### The evaluation set

| Item | Value |
|---|---|
| Frames | 162 real Caltrans frames, all 12 districts, captured 2026-10-13 around 09:00-09:25 PDT |
| Extra frames | Interstate and US-route cameras from the busy districts, so there are congested examples |
| Labels per frame | vehicle count, congestion class (Low / Moderate / High / Severe), usable yes/no |
| Usable | 124 frames (Low 92, Moderate 22, High 8, Severe 2) |
| Unusable | 33 "Temporarily Unavailable" images, 3 glare, 1 haze, 1 parking lot |

**Honest note about the labels:** they were made by looking at the frames (with Claude's help), not taken
from a sensor. Counts are approximate (small distant cars are counted, which makes the detector look
worse) and the classes are a visual judgement. Only 10 frames are High or Severe, so results for those
classes are weak evidence. Re-check them with `python cv_eval/make_sheets.py --labels`.

### Finding 1: one in five cameras was "Temporarily Unavailable"

33 of 162 sampled cameras served a mostly white placeholder image. A detector finds no cars in it, so the
app reported an empty road ("Low"). The placeholder is about 86 percent near-white pixels, and the most
washed-out real frame was 22 percent, so a 60 percent cutoff separates them. `analyze_frame` now raises
`CameraUnavailableError`, and the UI shows "camera unavailable" instead of a badge.

### Finding 2: the detector undercounted, and a lower confidence fixed most of it

| Setting (YOLOv8s, 640 px) | Count error (MAE) | Average miss | Within tolerance |
|---|---|---|---|
| Confidence 0.25 (YOLO default) | 7.43 vehicles | -7.3 (too low) | 39 percent |
| Confidence 0.08 (shipped) | 4.55 vehicles | -2.5 | 59 percent |

On dense frames (21 or more vehicles) the error fell from 23.5 to 12.6; on near-empty frames it moved
from 1.55 to 1.41, so empty roads did not get worse. The app now uses `DEFAULT_CONF = 0.08`.

### Finding 3: what did not help

- **A bigger model:** YOLOv8m at 0.08 reached 4.15 (a small gain) but takes 0.56 seconds per frame
  against 0.24, which hurts when a route checks 16 cameras. The app stays on YOLOv8s.
- **Higher input size (1280 px):** no gain.
- **Refitting the density thresholds** (0.05 / 0.15 / 0.30) on labelled data with cross-validation:
  worse macro-F1 than the original thresholds once the detector was fixed.
- **Sliced inference** (also searching four overlapping windows so far cars have more pixels):
  4.55 to 4.32, only about 5 percent, at roughly five times the compute. Kept as an option, not the default.
- **A learned congestion classifier** (logistic regression, random forest, gradient boosting): balanced
  accuracy 0.41-0.54 against 0.62 for the plain thresholds. 124 frames is too few for a learned model to win.

### Reading the label metrics correctly

74 percent of usable frames are Low, so "always answer Low" already scores 0.74 accuracy. That is why the
report also gives **balanced accuracy** (average recall per class) and **macro-F1**.

| Setting | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| Before tuning (confidence 0.25) | 0.73 | 0.32 | 0.33 |
| Shipped (confidence 0.08) | 0.75 | 0.62 | 0.64 |
| Always answer "Low" | 0.74 | 0.25 | - |

### Caveats to say out loud

The confidence value was picked from a handful of options on the same frames it is scored on, so the gain is
slightly optimistic. The set is daytime only, from one 25-minute window. Dense traffic is still undercounted
(one wide freeway frame with about 45 vehicles was read as 9). Nothing was fine-tuned: that needs bounding
boxes, not the counts collected here.

### Interview answer

> I measured the vision part on 162 live frames I labelled across all 12 districts. That found two real
> problems: a fifth of the cameras serve an unavailable image that looked like an empty road, and the
> detector at its default confidence undercounted by about 7 vehicles. Lowering the confidence cut the counting
> error from 7.4 to 4.6. I also tried bigger models, sliced inference, and a learned classifier, and kept the
> simple version because they did not clearly help. The labels are approximate and daytime only, so I treat
> the numbers as evidence, not proof.

---

## Lesson 12: Tracking and Speed From Live Video

### The problem

Speed is distance over time. The time is easy (the video stream has a clock). The distance is hard: the
cameras publish no calibration, so a pixel on the screen has no known size in metres.

### The pipeline (src/live_monitor.py and src/vehicle_tracking.py)

1. **Frames:** the stream is 15 frames per second. Every 4th frame is analysed (about 3.75 per second) and its
   time is read from the stream's own clock, because network delays make wall-clock time wrong.
2. **Detect and track:** YOLO finds the vehicles. A simple tracker predicts where each vehicle should be next
   from its recent motion and matches the nearest detection, so the same car keeps one id.
3. **Fit a camera model from the cars themselves** (first about 25 seconds). For a camera looking along a flat
   road, a car's bottom edge in the image (y) and its pixel width (w) follow a line: `y = horizon + a * w`.
   Fitting that line over many cars gives the horizon row and the camera height (`a * 1.8 m`, taking a typical
   car to be 1.8 m wide). If the fit is not convincing (too few cars, no trend, implausible height), speeds
   are switched off for that camera.
4. **Pixels to metres:** the focal length is assumed from a typical 62 degree field of view. Then
   `distance ahead = f * height / (y - horizon)` and `sideways = (x - centre) * distance / f`.
5. **Speed:** for each car, take its last 2.5 seconds, fit a straight line to its ground position over time,
   and take the slope. Only cars between 8 and 75 metres from the camera count.
6. **One correction:** the zoom was only assumed, so speeds are scaled so the median tracked car matches the
   "typical traffic speed" setting (90 percent of the posted limit by default). If that needs a correction
   outside 0.4 to 2.5 times, the camera is marked unreliable and issues nothing.
7. **Flag:** the car must stay over the limit plus a margin for 3 measurements in a row.

### The older method (Single camera, Speed check)

You measure a lane's width in pixels and type it in. Speed is pixels moved times (3.7 m divided by lane pixels)
divided by time. It ignores perspective and depends on your measurement.

### How far to trust it

- **Verified:** a synthetic car moving at a known speed reads within 4 percent.
- **Not verified:** there is no radar or other ground truth. On the real cameras tried, the raw model needed a
  correction of about 2 times, and far-away cars read much faster than near ones in the same traffic. That is
  why the distance band and the traffic-speed scaling exist.
- **What the numbers mean:** treat speeds as "faster or slower than the rest of the traffic", not exact mph.

### Interview answer

> There is no calibration data for these cameras, so I fit a flat-road camera model from the traffic itself:
> a car's ground row and width follow a line that gives the horizon and camera height. With an assumed focal
> length I convert image positions to road distance, track each car over a couple of seconds, and take the
> slope as speed. Because the zoom is unknown I scale speeds to a typical traffic speed, and if the model can't
> be calibrated the app refuses to issue anything. My math tests pass on synthetic cars within 4 percent, but
> I haven't validated against radar, so I describe the speeds as estimates.

---

## Lesson 13: The Live e-Challan Tab

### What you see

One tab with, top to bottom: camera selection (district, filter, camera), the camera's live video, the
automatic speed limit, a Start monitoring button, the annotated live detection (boxes with estimated speeds, a
vehicle counter, a congestion badge), the challans issued for real cars, then a form to make a challan from
made-up values, the challan history, a CSV download and a clear-history button.

### How it works

- **Live video:** an hls.js player runs in your browser and plays Caltrans's stream directly (their server
  allows this). Nothing is proxied or recorded by the app.
- **The monitor:** a background thread (`LiveMonitor`) reads the stream and runs detection, tracking and speed.
  It uses its **own** YOLO instance because the app's shared model is not safe to use from two threads. The
  page polls it once a second with a Streamlit fragment, and the monitor stops itself after the run time or if
  the page stops asking (45 seconds).
- **A challan:** when a car is clearly over the limit, the monitor crops its picture, optionally reads the plate
  and identifies its color and type, and builds a simulated challan. Plate reading and identification run in a
  separate worker so the video does not stall.
- **What a challan shows:** reference number, time in California time with its UTC offset, location, vehicle
  (with a note if no plate could be read), appearance, estimated speed, speed limit and where it came from, the
  illustrative fine in rupees, and the simulation disclaimer.
- **Privacy:** evidence pictures stay in memory only. The on-disk history holds text fields only.

### Safeguards

3 confirmations in a row; at most 12 challans per session, at least 3 seconds apart; no challans at all if the
camera could not be calibrated; a warning if the limit is set below typical traffic speed (which would flag most
cars).

### Plate, color, make and model

- **Plate (EasyOCR):** often unreadable on real footage; the challan then says "NOT READ".
- **Color and body type:** CLIP zero-shot and color clustering; best-effort guesses.
- **Make and model:** tried and gated. The median detected car is only 19 pixels wide, and even the 20 largest
  crops gave 5-15 percent confidence, so crops under 70 px are refused and a model is reported only above
  30 percent confidence. On these cameras it will rarely say anything, on purpose.

### Bug stories (good for interviews)

- **Hidden feature:** the e-challan used to appear only after a speed check found a speeder, so nobody saw it.
  It now has its own tab.
- **Runaway challans:** a test with the limit below normal traffic speed produced 293 challans in two minutes.
  That led to the cap, the spacing, and the warning.
- **Wrong wording:** a challan with no plate said "entered manually". It now says no plate could be read.

### Interview answer

> The Live e-Challan tab watches a real camera, tracks vehicles, and generates a clearly-labelled simulated challan
> from a real observed vehicle, with its picture, estimated speed, time, location and the posted limit. It runs in
> a background thread with its own detector, has safeguards against false or runaway challans, and refuses to issue
> anything when the camera can't be calibrated. It is a demonstration of the pipeline, not enforcement.

---

## Lesson 14: Time Zone, Speed Limits and Live Video

### The time-zone bug

The historical models take an hour, weekday and month. The app used the computer's own clock. A laptop on India
time (16:17) asked the models about Sunday afternoon rush hour while it was 03:46 in California, and the app showed
a false "Unusual congestion detected". Found by a full test pass. Fix: `src/camera_time.py` gives California time
(`America/Los_Angeles`), used for predictions, the forecast, the report footer and challan timestamps.

### The automatic speed limit (src/speed_limits.py)

OpenStreetMap stores a `maxspeed` tag on many roads (Overpass API, free, no key). The app finds the road the camera
watches and reads it:

1. A nearby road whose route reference matches the camera's (so a camera on an overpass reports the freeway,
   not the street crossing it), nearest first.
2. Otherwise the nearest motorway, then the nearest road of any kind.
3. If the road has no tag, a typical value for its class is used and labelled "assumed".

Checked live: the I-110 Arroyo Seco Parkway reports 55 mph and the I-5 freeway 65 mph, matching the posted limits.

**Why it never blocks the page:** the public Overpass servers are often slow or busy. The lookup runs in a
background thread, the page shows a clearly-labelled provisional value (65 mph for Interstate and US routes) and
updates itself when the real answer arrives. It tries three mirrors, and a found limit is cached on disk so each
camera needs only one successful lookup. Limits are only as good as what OpenStreetMap contributors entered.

### Live video in the browser

An hls.js player (version pinned) plays the camera's HLS playlist. About two thirds of cameras have a stream, and
many of those are offline at any moment, so a dead stream shows a message instead of a blank box.

### Interview answer

> I get the posted speed limit from OpenStreetMap by finding the road the camera is on and reading its maxspeed tag.
> Because the public server is slow, I do it in a background thread with a provisional value, mirrors, retries and
> a disk cache. I also found and fixed a bug where the models used the laptop's clock instead of California's.

---

## Lesson 15: Route Map, Design Decisions and Bug Stories

### The Route tab

Three ways to build a route, all giving the same result:

| Mode | How |
|---|---|
| Pick cameras | choose 2 or more cameras in a district |
| Search places | type a start and destination (Nominatim place search, one request per second) |
| Click the map | click once for the start, again for the destination |

The app gets the real road path from OSRM (OpenStreetMap routing, free), finds every camera within 1.5 km of it
(up to 16, evenly spaced), checks each one, and colors the road by the nearest camera: solid within 2 km, faded
2-8 km away (a weaker guess), gray beyond 8 km. The overall route level is the worst segment. Colors are
estimates from scattered snapshots, not continuous traffic data like a navigation app.

### Design decisions worth mentioning

- **Per-segment live reading:** each route segment and each grid badge shows that camera's own live reading. The
  historical label is one value for the whole time and weather, so putting it into every segment put a "Moderate"
  floor under all of them. It is applied once, to the overall reading.
- **Free services only:** Caltrans, OSRM, Nominatim, Open-Meteo and OpenStreetMap need no API key.
- **A dark control-room theme** with moving cars that slow down when traffic is congested, glowing badges, and an
  animated counter, all in CSS.

### Bug stories (a good list for questions)

| Bug | What happened | Fix |
|---|---|---|
| Stale selectbox label | Changing a filter crashed with a TypeError | validate the stored value and reset |
| Auto-refresh loop | A Streamlit fragment also runs on every full rerun, causing an endless refresh | elapsed-time guard |
| Everything "Moderate" | the shared historical label was fused into every camera | per-camera live label, fuse once |
| Placeholder as empty road | "Temporarily Unavailable" images read as Low | detect and refuse them |
| Wrong clock | models used the laptop's time zone | California time |
| Runaway challans | 293 in one test | cap, spacing, warning |
| Cropped map | a tall route was cut off in a short wide map | a proper fit-to-view calculation |

### What is distinctive (for "what makes it unique")

Not any single piece, but the combination: a historical model and live computer vision fused with a stated reason;
real, messy live data with real failures found and fixed; measuring the vision part and keeping only what helped;
and going beyond a dashboard to routes, tracking, speed, and a simulated enforcement pipeline, all with the limits
written down.

---

## Added Questions and Answers

### Q12. How is speed calculated?

By tracking each car across frames and converting image position to road distance with a camera model fitted from
the traffic itself, then scaling to a typical traffic speed. It is an estimate, not a measurement.

### Q13. How accurate is the speed?

Honestly, rough. The math is tested on synthetic cars (within 4 percent) but it is not validated against radar, so
I describe it as "faster or slower than the rest of the traffic".

### Q14. Is the challan real?

No. Every challan is simulated and labelled as such. Nothing connects to any authority or registry.

### Q15. Why is the historical model trained on Minneapolis data?

It was the best public hourly dataset with weather and traffic volume that I could use. It does not match California,
so its output is a time-of-day and weather baseline. The fix is California sensor data (PeMS).

### Q16. How did you evaluate the vision part?

I labelled 162 live frames from all 12 districts and measured counting error and label accuracy. Tuning the
confidence cut the error from 7.4 to 4.6 vehicles. I also tried bigger models, sliced inference and a learned
classifier and kept the simple version.

### Q17. What did you train yourself?

The three prediction models (Linear Regression, SVR, LSTM). The detector is pretrained; I tuned and evaluated it.

### Q18. Can it identify the car's make and model?

Mostly no. The median car on these cameras is 19 pixels wide, so the app only reports a make or model when the crop
is big enough and the model is confident, which is rare.

### Q19. How does the app get the speed limit?

From OpenStreetMap, by finding the road the camera watches. It runs in the background with a fallback if the
service is slow.

### Q20. What are the biggest limitations?

The historical model's data mismatch, unvalidated speeds, approximate labels from a small daytime set, and cameras
that are offline or have a sideways view.
