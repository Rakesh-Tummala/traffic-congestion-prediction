# Live demo walkthrough (computer vision + machine learning)

You are presenting by running the real app, not slides. Everything below uses the actual tab and
button names. Total: about 12 minutes if you do everything, 6 if you only do the **must show** items.

## 0. Before you present (do this 20 minutes early)

1. **Pick the right time.** The live parts need daylight and traffic in California, so check the clock
   (California is 12.5 hours behind India until 1 Nov 2026, 13.5 hours after):

   | California time | India time (until 1 Nov) | Verdict |
   |---|---|---|
   | 7-9 AM | 7:30-9:30 PM | Best: commute traffic |
   | 4-6 PM | 4:30-6:30 AM | Best: commute traffic |
   | 10 AM-3 PM | 10:30 PM-3:30 AM | Fine |
   | 8 PM-5 AM | 8:30 AM-5:30 PM | Poor: dark, almost no cars |

   If your slot is in the poor window, the prediction, grid and route tabs still work; the live monitor
   will show "calibrating" because there are too few cars. Have a screen recording of it working ready.
2. **Run the readiness check:** `python demo_check.py`. It tests every online service and lists the
   busiest cameras with working video right now. Note the top one and use it in step 5.
3. **Start the app:** `.venv\\Scripts\\streamlit run app.py`, open http://localhost:8501.
4. **Warm everything up** by visiting every tab once and pressing Fetch live snapshot, Load grid and
   Start monitoring (then Stop). The first run of each loads models and is slow; the second is fast.
5. **Have backups:** screenshots of each tab, and a short screen recording of the Live e-Challan tab.

## 1. Opening (30 seconds, on the header)

Say: "This is a traffic congestion system. It uses **computer vision** to count vehicles on real live
California highway cameras, and **machine learning** to predict what traffic should look like from the
time of day and weather. It combines the two, and everything you'll see runs on real live data."

Point at the moving cars at the top: "They slow down when the latest reading is congested."

## 2. Tab 1: Single camera (about 4 minutes) — must show

**Camera selection**
- Click a dot on the map, or use the District / Filter / Camera boxes. Say: "478 real Caltrans cameras."
- Tick **Watch live video**. Say: "This is the actual live stream, played straight from Caltrans."

**Computer vision**
- Press **Fetch live snapshot**. Point at the boxes: "YOLOv8 detects each vehicle. I tuned its
  confidence on 162 frames I labelled myself, which cut the counting error from 7.4 to 4.6 vehicles."
- Point at the **vehicle counter** and **image density ratio**: "Density is how much of the picture the
  vehicles cover. That becomes the live congestion level."

**Machine learning**
- Open the **model radio** (LSTM / SVR / Linear Regression) and the **All three models** table.
  Say: "Three models trained on 48,000 hours of traffic data, with R-squared of 0.977, 0.96 and 0.948."
- Open **Why this prediction?** "Which inputs are unusual right now."
- Point at **Today's forecast**: "The shaded band is the model's typical error on held-out data, so the
  forecast isn't pretending to be exact."
- Open **When should I leave?** "Suggests a nearby hour with lower predicted congestion."

**Fusion (the key idea)**
- Point at the caption "Live camera reading: ... historical reading: ... final level takes whichever is
  more severe." Say: "A warning system should never say Low when the camera shows gridlock."
- If the **Unusual congestion detected** banner is visible, explain it: the live camera and the
  historical model disagree sharply, which usually means an accident or event.

**Features to tick quickly**
- **Use real weather here** (fills conditions from the camera's location).
- **Auto-refresh** (re-fetches on a timer and builds **Recent history**).
- **Download report (PNG)**.
- Mention the **Upload image** and **Snapshot URL** tabs work for any picture.

**Optional (30 s):** scroll to **Speed check**, press **Check speeds**, then **Identify vehicle color & type**.

## 3. Tab 2: Monitoring grid (1 minute) — must show

Press **Load grid** (6 cameras). Say: "Each camera gets its own live detection and badge." Point at a
"camera unavailable" card: "About 1 in 5 Caltrans cameras serves a 'Temporarily Unavailable' image. My
first version read that as an empty road. I found it by testing and fixed it."

## 4. Tab 3: Route (2 minutes) — must show

Choose **Search places**: type `Burbank Airport` and `Santa Monica Pier`, press **Find places**, pick the
matches, press **Check route** (takes about 1 minute; talk while it runs).
Say: "It finds the real road path, then every camera within 1.5 km of it, checks them all, and colors the
road by each camera's live reading. The overall level is the worst segment."
Show **Click the map** (drop two points) and **Pick cameras** in 15 seconds each.

## 5. Tab 4: Live e-Challan (3 minutes) — the showpiece

1. The camera box is pre-set to **I-5 at Olive** (or pick the camera `demo_check.py` recommended).
   Say: "Live video here, and the speed limit was filled in automatically from OpenStreetMap."
2. Press **Start monitoring**. Say what you see as it runs:
   - "The detector runs on the video about 4 times a second and tracks each car."
   - "For the first 25 seconds it learns the camera's geometry from the traffic itself. There is no
     calibration data for these cameras."
   - Then boxes show **estimated speeds**, and the badge and vehicle counter update live.
3. When a car is clearly over the limit a **simulated challan** appears with the car's picture, time,
   speed, limit and its source. (To force one in a quiet moment, lower the limit and margin first.)
4. Scroll down: **Generate demo challan** (made-up values), the **history** table, **Download history
   (CSV)**, and **Clear history**.
5. **Say this clearly:** "This is simulated. The speeds are estimates, not validated against radar, so
   I built in safeguards: it refuses to issue anything if the camera can't be calibrated, a car must stay
   over the limit for three measurements in a row, and a session issues at most 12 challans."

## 6. Closing (1 minute): under the hood

- "133 automated tests." (Run `python -m pytest -q` in a terminal if you want to show it.)
- Open `cv_eval/results/report.md`: "The evaluation behind the numbers, including the ideas I tried and
  rejected: bigger models, sliced detection, a learned classifier."
- Open the README **Limitations and future work**. Say: "The historical model was trained on Minneapolis
  data, speeds are unvalidated, and the labels are approximate. I'd rather state that than hide it."

## 7. If something breaks

| Problem | What to do |
|---|---|
| Internet or Caltrans down | Show your screenshots and recording; the README and `cv_eval` results work offline |
| One camera says "unavailable" | Pick another; it happens to about 1 in 5 |
| Live monitor says "speeds not reliable" | Say "that is the safeguard working", then switch to the recommended camera |
| Route check is slow | Keep talking; it checks 16 cameras (about 1 minute) |
| Speed limit says "assumed 65 mph" | OpenStreetMap was slow; it updates by itself when it answers |
| Something looks frozen | Reload the page and continue |

## 8. Questions to expect

- **Why take the more severe of the two readings?** To avoid false reassurance.
- **What did you train yourself?** The three prediction models. The detector is pretrained; I tuned and evaluated it.
- **How accurate is the speed?** Honestly, rough: it is calibrated from the traffic itself, not radar.
- **Is the challan real?** No: it is simulated and labelled as such on every card.
- **Why is the historical model trained on Minneapolis data?** It was the best public hourly dataset
  available; the fix is California sensor data (PeMS), listed as future work.
- **Did you use AI to help?** Answer honestly per your college's rules, and be ready to explain any part.
