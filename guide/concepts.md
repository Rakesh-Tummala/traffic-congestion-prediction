# Part C: Concepts and Talking Points

Short explanations you can use when someone asks "what is X?" or "how does Y work?", plus two scripts to read aloud. Each section is written so you can say it in your own words.

## C1. The computer vision concepts used

| Concept | Where it is used | In plain words |
|---|---|---|
| **Object detection (YOLOv8)** | Counting vehicles in every frame | A neural network that finds objects in an image and draws a box around each with a class (car, truck, bus, motorcycle) and a confidence score. |
| **Confidence thresholding** | The 0.08 setting you tuned | Keep a detection only if the model is at least this sure. A lower value finds more cars but risks false ones. |
| **Non-maximum suppression and merging boxes** | Inside YOLO, and in the sliced-detection option | Removes duplicate boxes on the same car. Sliced detection also merges boxes found in overlapping tiles, using intersection over the smaller box. |
| **Multi-scale (sliced) inference** | The tiled option, which was measured and rejected | Run the detector on zoomed windows so tiny, distant cars get more pixels. |
| **Density estimation** | The congestion level | Vehicle-box area divided by image area, turned into Low to Severe. |
| **Image quality validation** | The "camera unavailable" check | Measure pixel statistics (about 86 percent near-white) to spot a placeholder image instead of a real scene. |
| **Video decoding and timestamps** | Live monitor, speed check | Reading frames from a live HLS stream and using the stream's own clock for timing. |
| **Object tracking (data association)** | Live monitor | Matching the same car from frame to frame: predict its next position from its motion, then match the nearest detection. |
| **Camera geometry (pinhole model, horizon, ground plane)** | Speed estimation | Treat the road as flat and use where a car's bottom edge sits in the image to work out how far away it is. |
| **Perspective and scale** | Speed, the distance band | Far cars cover few pixels, so tiny pixel errors become large speed errors. That is why only cars 8 to 75 metres away count. |
| **Motion estimation** | Speed | The slope of a car's position over time. |
| **Zero-shot classification (CLIP)** | Vehicle body type, make and model | A model that matches an image to text prompts like "a sedan", with no training on traffic data. |
| **Color analysis (k-means, HSV)** | Vehicle color | Cluster the car's pixels, ignore shadows and glare, and name the dominant color. |
| **OCR (EasyOCR)** | Plate reading | Find and read text in a cropped image. It usually fails here because the plates are tiny or hidden. |
| **Image preprocessing** | Plate and crop handling | Cropping, upscaling, and drawing boxes and labels on frames (OpenCV). |
| **Evaluation of a vision system** | `cv_eval/` | Hand-labelled frames, counting error (MAE), a confusion matrix, balanced accuracy and cross-validation. |

**What is pretrained and what is yours.** YOLO, CLIP and EasyOCR are pretrained models you use. Your own work is everything around them: the placeholder check, the density-to-label rule, the tracker, the camera-geometry and speed estimation, the evaluation, and the tuning.

## C2. YOLO, CNN, OpenCV and OCR explained

### What is YOLO

YOLO stands for "You Only Look Once". It is a neural network that finds objects in an image. You give it a picture and it returns a list of detections: a **box**, a **class** (car, truck, bus...) and a **confidence score** from 0 to 1 (the `car 0.66` on the boxes in the app).

It is called "You Only Look Once" because older methods scanned an image over and over, window by window, which was slow. YOLO looks at the whole image in one pass and predicts every box at once, which makes it fast enough for live video.

How it works in four steps: the image is resized and passed through a convolutional neural network; the network predicts many candidate boxes, each with a class and a confidence; non-maximum suppression removes duplicate boxes on the same object; boxes below your confidence threshold (0.08) are dropped.

Your version is **YOLOv8s** ("s" means small), pretrained on COCO, a dataset of everyday objects. You did not train it. You tuned it and evaluated it on frames you labelled. It takes about 0.24 seconds per frame on your CPU.

*Say:* "YOLO is a pretrained object detector that finds vehicles in one fast pass over the image. I use it to count cars and measure how crowded the road is, and I tuned its confidence threshold on frames I labelled."

### How YOLO detects the shape of a car

YOLO does not have a hand-written rule like "a car has four wheels and a rectangle body". It learns what cars look like from millions of labelled examples. During training it made guesses, compared them with human-drawn boxes, and adjusted its internal numbers (weights) until the guesses matched. The "shape" of a car is a learned combination of parts and textures, not a stored outline:

```text
Early layers   -> edges and corners, light and dark patches
Middle layers  -> curves, wheels, windows, headlights, textures
Deep layers    -> "this whole arrangement looks like a car"
```

It struggles when a car lacks those cues: very small or far cars, blur, night, glare, or cars hidden behind others. That is why the app undercounts dense, distant traffic.

### What is a CNN

A CNN (convolutional neural network) is a neural network designed for images. It keeps the picture's 2D layout and looks for small local patterns.

1. **Convolution:** a tiny grid of numbers called a **filter** (for example 3 by 3 pixels) slides across the whole image and measures how strongly each spot matches a pattern (an edge, a curve, a texture). The output is a feature map.
2. **Activation (ReLU):** keeps strong responses and sets weak ones to zero.
3. **Pooling or striding:** shrinks the map, keeping the strongest responses.
4. **Stack many layers:** edges become corners and curves, then wheels and windows, then a whole car.

The same filter is used everywhere in the image, so it finds a wheel on the left or the right. The filters are learned during training, not designed by anyone. YOLO and EasyOCR use CNNs, and CLIP's image side is a related vision network. Your LSTM is a different kind of network, built for sequences.

*Say:* "A CNN slides small learned filters across the picture to detect patterns, stacking layers so edges build up into parts and then whole objects."

### What is OpenCV

OpenCV is a toolbox of functions for working with images and video. It does the handling around YOLO, and it does not detect cars itself.

| Job | What OpenCV does | Where |
|---|---|---|
| Decode images | Turns downloaded JPEG bytes into a grid of pixel numbers | `load_frame()` in `src/cv_module.py` |
| Read video | Pulls frames and timestamps from a live stream | Live monitor, speed check |
| Draw on frames | Draws the boxes, labels and speeds | Annotated frames |
| Crop and resize | Cuts out a car and enlarges it 4 times for plate reading | Plates, evidence pictures |
| Convert color | BGR for OpenCV, RGB for display, HSV for color naming | Vehicle color, display |
| Measure pixels | Checks pixel values, such as the "mostly white" test | Camera-unavailable check |

OpenCV treats an image as a NumPy array: height by width by 3 color values per pixel. The flow is: OpenCV loads the frame, YOLO finds vehicles, OpenCV draws the boxes.

*Say:* "OpenCV is the library I use for everything around the detector: loading images, reading live video frames, cropping, color conversion, and drawing the boxes. YOLO does the actual detection."

### What is OCR

OCR stands for Optical Character Recognition: reading text out of an image, turning pixels that look like letters and numbers into text a computer can use. The program finds regions that look like text, a neural network recognizes each character or word, and it outputs text with a confidence score.

Your project uses it to try to read license plates (ANPR), with the **EasyOCR** library in `src/anpr.py`. It crops the picture to the car, enlarges the crop 4 times, reads any text, keeps strings of 4 to 12 letters or digits, and accepts one only if confidence is at least 0.35. If no plate is read, the challan says "NOT READ".

It usually fails here because the cameras are far away and low-resolution, so a plate is a few pixels wide or hidden. In tests, real footage mostly returned "not readable", which is the expected result, not a bug. On a clean test image with clear text it reads correctly.

*Say:* "OCR reads text from images. I use it to try to read plates, but at this camera resolution it rarely succeeds, so the app reports 'not read' honestly instead of guessing."

## C3. How the live feed is captured

The cameras belong to **Caltrans**, California's highway department. Caltrans captures the video; your app only fetches it. There are two kinds of feed.

| | Captured by | Fetched with | Used for |
|---|---|---|---|
| Snapshot (still image, about 1 per minute) | Caltrans camera | `requests` + OpenCV | Congestion readings, grid, route |
| Video to watch | Caltrans camera | hls.js, in your browser | The live video panel |
| Video to analyze | Caltrans camera | OpenCV `VideoCapture` | Tracking, speed, challans |

- **Camera list:** `src/live_cameras.py` downloads Caltrans's public camera list (a JSON file per district). Each entry has the camera's name, coordinates, snapshot URL and video URL.
- **Snapshots:** `load_frame()` downloads the JPEG and decodes it with OpenCV.
- **Video:** most cameras publish an HLS stream (a `playlist.m3u8` file pointing to short video chunks) on `wzmedia.dot.ca.gov`. The hls.js player in your browser plays it directly from Caltrans. The live monitor and speed check read frames from it on the server with OpenCV (FFmpeg underneath), using the stream's own timestamps.

Snapshots refresh only about once a minute, so the congestion reading is near-live. Many video streams are offline at any moment.

## C4. How the density ratio (frame area) is calculated

It is simple arithmetic on the box coordinates YOLO returns, in `analyze_frame` in `src/cv_module.py`.

```text
box area      = (x2 - x1) x (y2 - y1)
density_ratio = sum of all vehicle box areas / (frame width x frame height)      (capped at 1.0)
```

Worked example, a 1280 by 720 frame (921,600 pixels) with three vehicles:

| Vehicle | Box width x height | Area |
|---|---|---|
| Car 1 | 80 x 50 | 4,000 |
| Car 2 | 60 x 40 | 2,400 |
| Truck | 120 x 90 | 10,800 |
| Total | | 17,200 |

Density = 17,200 / 921,600 = 0.0187, about 1.9 percent of the picture, which is under 0.05, so the level is **Low**. The thresholds are: under 0.05 Low, 0.05 to 0.15 Moderate, 0.15 to 0.30 High, 0.30 and above Severe.

Things to know: overlapping boxes are counted twice; a close truck covers more pixels than ten distant cars (perspective); the whole frame is the denominator, including sky and the timestamp bar, which is why the thresholds are small numbers; it is a proxy for congestion, not a direct measurement, which is why it was checked against hand-labelled frames.

*Say:* "I add up the area of every vehicle box (width times height) and divide by the area of the whole image. That fraction is the density, and it maps to Low, Moderate, High or Severe."

## C5. How vehicle type, color, make and model are found

| What | Tool used | Where |
|---|---|---|
| Car, truck, bus or motorcycle | **YOLOv8s**, the same detector | `src/cv_module.py` |
| Body type (sedan, SUV, pickup truck, hatchback, minivan, bus, truck, motorcycle) | **CLIP** | `src/vehicle_attributes.py` |
| Color | **K-means clustering** of the car's pixels (shadows and glare filtered out), then the dominant shade is named | `src/vehicle_attributes.py` |
| Make and model | **CLIP** again, strictly gated | `src/vehicle_attributes.py` |
| Number plate | **EasyOCR** | `src/anpr.py` |

**How CLIP picks the body type.** CLIP is a model trained on huge numbers of images paired with text. The app gives it the car's crop and a list of descriptions such as "a sedan car", "an SUV", "a pickup truck". CLIP scores how well the picture matches each one and the best match wins. This is zero-shot classification: the model was never trained on traffic cameras or on your labels. The app reports a type only when confidence is at least 25 percent.

**Limits.** Body type is usually 30 to 55 percent confident on clear daytime crops and weaker on small or night images. Color can be confused by shadows. Make and model rarely works: the median car on these cameras is only 19 pixels wide, so crops under 70 pixels are refused and a model is reported only above 30 percent confidence.

## C6. The dataset: what is in it and why it was used

The **Metro Interstate Traffic Volume** dataset (UCI Machine Learning Repository) is in `data/Metro_Interstate_Traffic_Volume.csv`. It has **48,204 hourly rows**, from 2 October 2012 to 30 September 2018, for traffic on Interstate 94 near Minneapolis and St. Paul, Minnesota, with the weather at each hour.

| Column | What it is | Example |
|---|---|---|
| `date_time` | the hour | 2014-05-12 20:00 |
| **`traffic_volume`** | **vehicles counted that hour (what the models predict)** | 2,690 |
| `temp` | temperature in Kelvin (the code converts to Celsius) | 286.6 |
| `rain_1h`, `snow_1h` | rain and snow in the last hour, mm | 0.25 |
| `clouds_all` | cloud cover, percent | 68 |
| `weather_main` | the weather type | Clouds, Clear, Rain, Snow, Mist |
| `holiday` | the holiday's name, or "None" | Columbus Day |

Numbers to know: median volume about 3,380 vehicles an hour and a maximum of 7,280; the daily pattern is strong (about 370 at 3 AM, 4,740 at 7 AM, 5,664 at 4 PM, 1,469 at 11 PM); holidays are rare (61 of 48,204 hours); a few bad rows exist (10 readings with a temperature of 0 K).

**Why this dataset:** it is the right shape (hourly traffic volume plus weather and time); it is large and spans years, so the models can learn daily, weekly and seasonal patterns and be tested properly on the newest data; it is free and public, so anyone can reproduce your results; and it lets you show real ML technique (time-based splitting, lag features, comparing three models).

**The honest limit:** it is Minnesota data, not California. For the California cameras the historical model is a time-of-day and weather baseline, not a local prediction. The fix would be California sensor data (PeMS).

## C7. How the three ML models work, are trained and are used

All three models take the same **17 input features** and output one number: the traffic volume for that hour. Features: hour, day of week and month (each as sin and cos), weekend flag, temperature, rain, snow, clouds, holiday flag, weather code, and traffic 1 hour ago, 3 hours ago, 24 hours ago, and the 3-hour average. Features are standardized before training.

| Model | How it makes a prediction | Notes |
|---|---|---|
| **Linear Regression** | A weighted sum of the features: `volume = w1*f1 + ... + w17*f17 + bias`. Training finds the weights that make squared errors smallest. | Instant to train; explainable; only straight-line relationships. |
| **SVR** | Fits a flexible curved surface with an RBF kernel: it blends the values of similar important training hours, weighted by similarity. | `C=100, epsilon=10, gamma=auto`, tuned with RandomizedSearchCV and TimeSeriesSplit. Trained on 8,000 sampled rows because training slows quickly with more rows. |
| **LSTM** | A neural network that reads the previous 24 hours of features and learns patterns over time through "gates" that decide what to remember and forget. | 2 stacked LSTM layers of 64 units, 20 percent dropout, then a small 64 to 32 to 1 output network. |

### How the LSTM is trained (`src/train_dl.py`)

1. **Split by time:** oldest 70 percent train, next 15 percent validation, newest 15 percent test.
2. **Scale** features and the target using only the training part.
3. **Build sequences:** each example is hours 1 to 24, and the answer is the volume at hour 25; then hours 2 to 25 predict hour 26, and so on.
4. **Training loop** (up to 40 rounds called epochs, batches of 128): predict, measure the error with **MSE** loss, use backpropagation to find how much each weight caused the error, and let the **Adam** optimizer nudge every weight a small step (learning rate 0.001).
5. **Validation each round,** with **early stopping** after 5 rounds without improvement; the best weights are kept and reloaded.
6. **Final test** on the newest 15 percent, converted back to real vehicle counts, then MAE, RMSE and R2 are calculated, and the model is saved as `models/lstm.pt`.

Scores vary slightly between runs because the weights start from random numbers, which is why the README says "about 0.977".

*Say:* "I trained the LSTM on sliding 24-hour windows of the features to predict the next hour's traffic, minimizing squared error with the Adam optimizer, and used a validation set with early stopping to keep the best version. I tested it only on the newest 15 percent of the data."

### Where each model is used in the app

The app asks all three for a prediction every time, then uses them in different places.

| Where in the app | Which model | How it is used |
|---|---|---|
| "All three models" table (Single camera) | All three | Shows each model's R2, predicted volume and level side by side. |
| "Model driving the forecast" radio | Your choice (LSTM by default) | Sets the historical congestion label that is fused with the camera reading; draws the 24-hour forecast; powers "When should I leave?". |
| Forecast error band | The chosen model's own error | Shaded band: LR plus or minus 334, SVR plus or minus 254, LSTM plus or minus 203 vehicles an hour. |
| Anomaly warning | The chosen model | Compares its label with the live camera's reading. |
| Monitoring grid and Route | LSTM (fixed) | One historical label per run, used for the overall reading. |

The differences between them are small (R2 0.948 against 0.977), because the lag features already give the simpler models most of the information.

*Say, about 60 seconds:* "Machine learning does the prediction side. I trained three models, Linear Regression, SVR and an LSTM, on 48,000 hours of traffic and weather data to predict how many vehicles to expect at a given time. I used sin and cos for the hour so midnight sits next to 11 PM, and I added traffic from 1, 3 and 24 hours earlier, which gave the biggest gain. I split the data by time, so the models are tested only on the future they never saw. The LSTM scored highest, with an R-squared of about 0.977, and the simpler models were close behind. The app turns the prediction into a congestion level, a 24-hour forecast, and an unusual-congestion warning, and fuses it with what the live camera shows. The weakness is that this dataset is from Minnesota, so for California it is a time-of-day and weather baseline, and the live camera does the real work."

## C7b. The ML part of the project: what the three models represent, and the maths behind them

### What "the ML part" means here

The project has two halves. The **machine learning half** answers: "given the time, the weather and recent traffic, how many vehicles should be on this road?" The **computer vision half** answers: "how many vehicles are on the road right now?" The two are then fused.

The ML half is **supervised regression**: the models are shown many examples where the inputs (time, weather, recent traffic) and the correct answer (the real traffic volume) are both known, and they learn the relationship. Then they predict volumes for hours they have never seen.

```text
Raw CSV (48,204 hours)
   -> data_loader.py: build the 17 features (sin/cos time, weather, lags)
   -> split by time: 70% train / 15% validation / 15% test
   -> train LR, SVR, LSTM on the training part
   -> score each on the unseen test part (MAE, RMSE, R2)
   -> save the models in models/
   -> live: build the 17 features for "now", ask all three models, volume -> label
```

### The three models are a ladder, each adding one idea

| | Linear Regression | SVR | LSTM |
|---|---|---|---|
| **What it assumes** | volume is a weighted sum of the features | volume is a smooth curve of the features | volume depends on the pattern of the last 24 hours |
| **What it can capture** | straight-line effects only | curved effects and interactions | curved effects plus how things change over time |
| **What it needs** | almost nothing | choosing C, epsilon, gamma | many choices: layers, size, learning rate, stopping |
| **Training cost** | instant | slow with many rows (so 8,000 sampled) | the heaviest |
| **Explainable?** | yes, one weight per feature | partly | hard |
| **Score (R2)** | 0.948 | 0.960 | about 0.977 |

Using all three shows what each added idea buys. The gaps are small because the lag features (traffic 1, 3 and 24 hours ago) already give even the simple model most of the information.

### How each prediction is calculated

**Linear Regression.** A straight weighted sum:

```text
volume = w1*hour_sin + w2*hour_cos + ... + w17*rolling_mean_3h + bias
```

Training picks the 17 weights and the bias that make the sum of squared errors on the training data as small as possible (ordinary least squares, which has a direct solution, `w = (X^T X)^-1 X^T y`). A positive weight on `lag_1h` means "more traffic an hour ago, more traffic now". Illustrative only, with made-up numbers: if the standardized `lag_1h` is +1.2 and its weight is 900, that feature alone adds about 1,080 vehicles to the prediction.

**SVR (Support Vector Regression).** Instead of one line, it predicts by comparing the new hour with the most informative training hours (the "support vectors"):

```text
volume = sum over support vectors of ( a_i * K(x_i, x) ) + bias
K(x_i, x) = exp( -gamma * distance(x_i, x)^2 )
```

`K` is the RBF kernel: it equals 1 when two hours are identical and falls toward 0 as they get further apart, so the prediction is a similarity-weighted blend of the training hours that look like this one. Two settings shape the fit: `epsilon = 10` means errors under 10 vehicles cost nothing (so it does not chase noise), and `C = 100` sets how strongly larger errors are penalized. Only the support vectors (hours near or outside the epsilon margin) get a non-zero weight `a_i`.

**LSTM (Long Short-Term Memory).** A neural network that reads the 24 hours one step at a time, keeping a memory. At each hour it uses three small "gates", each a learned weighted sum squashed to a value between 0 and 1:

```text
forget gate   f = sigmoid( W_f . [h_prev, x] + b_f )
input gate    i = sigmoid( W_i . [h_prev, x] + b_i )
candidate     g = tanh(    W_g . [h_prev, x] + b_g )
memory        c = f * c_prev + i * g
output gate   o = sigmoid( W_o . [h_prev, x] + b_o )
output        h = o * tanh(c)
```

Here `x` is this hour's 17 features, `h_prev` is the previous hour's output, and `c_prev` is the memory so far. The **forget gate** decides what to drop from memory, the **input gate** decides how much of the **candidate** (the new information) to store, the **memory** line updates the memory, and the **output gate** decides what to pass on as `h` to the next hour.

After the 24th hour, the last output `h` goes through a small network (64 to 32 to 1) to give the predicted volume. All the `W` and `b` numbers are what training adjusts. Two LSTM layers are stacked, with 64 units each.

### How training finds the numbers

All three learn by reducing the same kind of error, squared prediction error, on the training data: Linear Regression solves for the weights directly; SVR solves an optimization that finds the support vectors; the LSTM improves its weights gradually with backpropagation and the Adam optimizer, checking a validation set after every round and stopping early when it stops improving. In every case, the final score comes from the test set the model never saw.

### From volume to a congestion level

A predicted volume is turned into Low, Moderate, High or Severe by comparing it with the **quartiles of the historical volumes** (the 25th, 50th and 75th percentiles). Below the 25th percentile is Low, between the 25th and 50th is Moderate, between the 50th and 75th is High, and above the 75th is Severe. That label is what gets fused with the live camera's label.

### What counts as "ML" in this project, and what does not

| Part | Type | Trained by you? |
|---|---|---|
| Linear Regression, SVR, LSTM | Supervised regression (the ML half) | Yes |
| YOLOv8 detector | Deep learning (CNN), pretrained | No; tuned and evaluated by you |
| CLIP (vehicle type) | Deep learning (vision-language), pretrained | No |
| EasyOCR (plates) | Deep learning, pretrained | No |
| K-means (vehicle color) | Unsupervised clustering | Fits itself to each picture |
| Fusion rule, tracker, speed model, thresholds | Hand-designed logic and geometry | Written by you |

*Say:* "The machine learning part of my project is the three prediction models. They are supervised regression models that learn from 48,000 hours of traffic and weather to predict how many vehicles to expect. Linear Regression is a weighted sum, SVR is a similarity-based curve, and the LSTM reads the previous 24 hours with memory gates. I trained and tested all three the same way, split by time, and compared them with MAE, RMSE and R-squared. The computer vision half then checks what the camera actually shows, and the two are fused."

*If asked "why not just use the LSTM?":* "Because comparing three shows what each idea adds. The simple linear model is almost as good once the right features are there, and it is explainable. The LSTM gives the best score, so it is the default."

## C8. The decimal numbers: what they mean and how they are calculated

| Where you see it | Example | What it means |
|---|---|---|
| Labels on the boxes in the photo | `car 0.66` | **Confidence**: how sure the detector is (0.66 = 66 percent). The app keeps detections down to 0.08. |
| Image density ratio | `0.0179` | The share of the picture covered by vehicle boxes (about 1.8 percent). |
| Model names | `LSTM (R2=0.977)` | **R2**: how well the model fits held-out data (1.0 is perfect). It belongs to the historical model, not the camera part. |
| Forecast chart band | plus or minus 203 vehicles an hour | The model's typical error (MAE) on held-out data. |
| Input boxes | `45.00`, `65.00` | Numbers in input boxes, shown with two decimals. |
| Live monitor status | `calibration scale x2.26` | How much speeds were multiplied to match typical traffic speed. A big number means speeds are mostly relative. |
| Camera model | `fit quality 0.96` | How well the camera geometry fit the traffic (the app refuses below 0.7). |
| Counting error | `7.43 to 4.55` | Average miss in vehicles per frame (MAE), before and after tuning. |
| Label scores | `0.32 to 0.62` | **Balanced accuracy**: average success per class. It matters because 74 percent of frames are Low. |

### Formulas

```text
Fusion:           final level = the more severe of (historical label, live camera label)
MAE               = average of |predicted - actual|
RMSE              = square root of the average of (predicted - actual)^2
R2                = 1 - (sum of squared errors) / (total variation around the average)
hour_sin, hour_cos = sin(2*pi*hour/24), cos(2*pi*hour/24)
Recall (one class) = frames of that class labelled correctly / all frames of that class
Balanced accuracy  = average of the recall of each class
F1 (one class)     = 2 x precision x recall / (precision + recall)
Distance ahead     = focal_length x camera_height / (y - horizon)
Sideways position  = (x - image_centre) x distance / focal_length
Speed (mph)        = slope of ground position over time (m/s) x 2.237
Scale factor       = typical traffic speed / median tracked speed
Macro-F1          = the average of F1 over the four classes
```

How the ML scores are computed: the models are trained on the oldest 70 percent, tuned on the next 15 percent, and scored on the newest 15 percent, which they have never seen, using MAE, RMSE and R2. A score of 0.977 does not mean "right 97.7 percent of the time"; it measures how closely predictions follow real traffic on unseen hours.

## C9. Why California, and why not India

All live cameras are **Caltrans** (California). Caltrans publishes about 3,400 camera images, live video for most of them, and coordinates, all free with no key or registration. That is what let the app do live detection, routes and video.

I checked India. I found no Indian government API that serves live traffic camera feeds. The government open-data portal (data.gov.in) has traffic-junction CCTV records, but they are one-time summary counts (cameras per city), not a live feed. Websites listing Indian CCTV streams are unofficial, and some warn the streams may be outdated or unauthorized. Live camera access in India usually sits behind city traffic-management control rooms. For training data, I found only small datasets that do not match hourly highway volume plus weather (a Pune intersection count dataset, a Bangalore travel-time dataset, and an IEEE DataPort junction-count set).

```text
data.gov.in:  https://tn.data.gov.in/resource/cctv-cameras-traffic-junctions-2019-2020
Pune counts:  https://data.mendeley.com/datasets/xnf2k6n288
Bangalore:    https://groups.google.com/g/datameet/c/NDT9b0nmVgM/m/l7-Tvkv9CQAJ
IEEE DataPort: https://ieee-dataport.org/comment/7159
```

What works outside California: the Upload image and Snapshot URL tabs work for any traffic photo anywhere; route and place search work worldwide but have no cameras to check; the live video, grid and Live e-Challan tabs are California only. Uploaded images use California time for the historical part.

*Say:* "I used California because Caltrans publishes free, open, machine-readable live cameras. I looked at India and could not find a public live-camera feed; those are usually held by city traffic control rooms. The detection, fusion and routing do not depend on the country, so adding India would mean getting access to a camera source from a city authority. It is listed as future work, and the Upload tab already works on any Indian traffic photo."

Describe it as "I could not find one", not "none exists".

## C10. What the project is built around

**Congestion** is the main feature. **Speed** is an experimental extra.

1. **Congestion prediction** is the heart: machine learning predicts expected traffic from time and weather, computer vision measures what the live camera shows, and the two are fused into one level (Low, Moderate, High, Severe).
2. **Built on that result:** the forecast and "when should I leave", the monitoring grid, and the route map colored by congestion.
3. **Speed and challans** (the Live e-Challan tab) are an additional, experimental part. They show what the same vision pipeline can do, but they are simulated and the speeds are not validated against radar.

Do not call speed a main feature: it is the least reliable part, and the numbers you can defend (counting error, R2 of 0.977) belong to the congestion part.

*Say:* "The core of the project is congestion prediction: machine learning plus computer vision, fused into one level. Everything else builds on it. I also added a speed and challan pipeline as an experimental extension, and I am open that its speeds are estimates."

## C11. What makes this project different

*Say, about 30 seconds:* "Most existing traffic systems do one of two things. Navigation apps like Google Maps use phone and sensor data that only the company has. Many student projects either predict traffic from a historical dataset, or count cars in a picture. Mine combines both on real, free, public camera feeds. It predicts what traffic should be from time and weather, checks what the camera actually shows, and takes the more severe of the two, so it never reassures when the road is jammed. I also measured the vision part on frames I labelled, and I am open about where it falls short."

Three points to back it up:

1. **It fuses two signals, and I can say why.** A weighted average I tried first failed on a gridlocked image, so I changed it.
2. **It works on real, messy live data.** It found real failures (about 1 in 5 cameras serves an "unavailable" image, which the first version read as an empty road) and fixed them.
3. **It measures itself and goes further.** An evaluation on 162 labelled frames, ideas I tried and rejected, and then routes, live tracking, speed estimates and a simulated challan pipeline, with the limits stated.

If asked "Google Maps already does this": "Yes, Google has far better data. This project is not trying to compete with that. It shows how to build a congestion system from only free public data, how to combine ML and computer vision, and how to measure how well it works."

Do not say: that it is more accurate than existing systems, that the speeds are accurate, that it is completely new, or that it can identify car makes. Close with the limit before they ask: the historical model was trained on Minneapolis data, so for California it is a time-of-day baseline.

## C12. Which cameras to use during the presentation

Use cameras in **District 7 (Los Angeles)**, where everything was tested, and run `python demo_check.py` about 20 minutes before you present. It lists the cameras with the most traffic and a working video stream at that moment. Cameras go offline, so treat these as defaults, not guarantees.

| Part of the demo | Camera or route to use | Why |
|---|---|---|
| Single camera | Any I-5 or I-405 camera in District 7 | Busy freeway shots show more detections. The I-110 Avenue 26 default works but is often quiet. |
| Monitoring grid | District 7, filter `I-405` or `US-101`, 6 cameras | Freeway cameras usually have traffic. Leave an "unavailable" card in as a talking point. |
| Route | Search places: **Burbank Airport to Santa Monica Pier** | Tested: 40.7 km, 16 cameras, visually strong. |
| Live e-Challan | **I-5 : (37) Olive, Burbank** (the default) | The camera that calibrated speeds well in tests. |

Backups for the Live e-Challan tab: the I-5 cameras (17) South of I-10, (31) Zoo Drive and (40) North of Burbank Blvd, all District 7. Their streams worked in earlier tests but were not tested in the live monitor at a busy hour.

Avoid District 4 (a large share of its cameras served the unavailable image in my sample), cameras with a sideways view (the monitor will say speeds are not reliable), and quiet or night cameras. Best windows are California 7 to 9 AM or 4 to 6 PM, which is 7:30 to 9:30 PM or 4:30 to 6:30 AM India time.

## C13. Script to read: the computer vision presentation (about 5 minutes)

Lines marked DEMO are cues for what to click.

**Opening.** Good morning. My project is a traffic congestion system, and I will focus on the computer vision part. The question it answers is simple: can a program look at a live highway camera and tell how congested the road is?

**The problem.** Predicting traffic from history alone misses what is happening right now, like an accident. A camera shows the road as it is. So my system uses computer vision to read live cameras, and combines that with a machine learning model that knows what traffic is normally like.

**Detecting vehicles.** DEMO: Single camera, Fetch live snapshot. Here is a real Caltrans camera in Los Angeles. The boxes are from YOLOv8, an object detector. It is a convolutional neural network: early layers find edges, later layers find parts like wheels and windows, and the final layers predict a box and a class for each vehicle, all in one pass. That is why it is fast enough for live video. YOLO is pretrained. I did not train it. What I did was tune it and test it.

**From boxes to congestion.** I add up the area of all the vehicle boxes and divide by the area of the image. That is the density ratio. Under 0.05 is Low, up to 0.15 Moderate, up to 0.30 High, and above that Severe.

**Measuring it honestly.** A model without a measured accuracy is just a claim, so I labelled 162 real frames from all 12 districts, with vehicle counts and congestion levels. That showed two real problems. First, about one in five cameras was serving a "Temporarily Unavailable" image. The detector saw no cars and reported an empty road. Now the app recognizes that image and says the camera is unavailable. DEMO: Monitoring grid, Load grid, point at an unavailable card. Second, at its default setting, the detector missed about 7 vehicles per frame. Lowering the confidence threshold cut the counting error from 7.4 to 4.6. I also tried a bigger model, higher resolution, a learned classifier and sliced detection. None helped enough, so I kept the simpler version.

**Tracking and speed.** DEMO: Live e-Challan, Start monitoring. Now the live video. The app tracks each car from frame to frame. To estimate speed I had a problem: these cameras have no calibration data. So I fit a camera model from the traffic itself. For a camera looking along a flat road, a car's position in the image tells you how far away it is. With that, I convert movement on screen into movement on the road. I tested the math on synthetic cars of known speed, and it is accurate to within 4 percent. But I have not validated it against radar, so I treat the speeds as estimates, and the app refuses to issue anything if it cannot calibrate a camera. Any challan you see is simulated and labelled that way.

**Other vision tools.** I also use CLIP, a vision-language model, to guess the vehicle's body type, and color clustering for its color. OCR tries to read plates, but at this resolution the median car is only 19 pixels wide, so it usually cannot, and the app says so honestly.

**Limits.** Three honest limits. My evaluation set is small, daytime only, and labelled by hand. The speeds are unvalidated. And I did not train the detector. Fine-tuning it on traffic footage is the next step, along with California sensor data for the prediction side.

**Close.** To summarize: I used a pretrained detector, built the measuring and tracking around it, found real failures, fixed them, and kept only what the evidence supported. Thank you. I am happy to take questions.
