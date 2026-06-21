# 📈 Sales Forecasting Dashboard

A multi-page machine learning web app that explores historical sales, forecasts future demand using Prophet, and compares forecasting models — built on the Superstore dataset with Python and Streamlit.

### 🔗 Live Demo
**[https://sales-forecast-dashboard-fmvcgucmpwxnbuniuyh4ng.streamlit.app/](https://sales-forecast-dashboard-fmvcgucmpwxnbuniuyh4ng.streamlit.app/)**

---

## What This Project Does

This app takes years of retail sales data and turns it into something useful: an interactive dashboard where you can explore what happened in the past, and a forecast that predicts what sales might look like in the weeks ahead. Behind the scenes, three different forecasting models were trained and compared to find the most accurate one.

It's a complete, real-world machine learning project — data exploration, feature engineering, model training, an interactive dashboard, and a live deployment.

## The Dashboard Pages

- **Home** — key numbers at a glance (total sales, profit, orders) and the overall sales trend.
- **Overview** — interactive analytics. Filter by region and category and watch every chart update: sales over time, top sub-categories, sales by customer segment, and monthly seasonality.
- **Forecast** — the centerpiece. Pick how many weeks ahead to predict and a region, and the Prophet model retrains live and draws the forecast with a confidence band (the range the true value is likely to fall in).
- **Model Performance** — the honest test scores of all three models, a comparison chart, and an explanation of why the winner won.

## Tech Stack

- **Python 3.11**
- **Streamlit** — the multi-page web app
- **Prophet** — the forecasting model (and the winner)
- **XGBoost** — a competing machine-learning model
- **statsmodels** — seasonal decomposition during analysis
- **pandas / numpy** — data handling
- **Plotly** — interactive charts
- **Anaconda, Git & GitHub, Streamlit Community Cloud** — environment, version control, deployment

## Project Structure

```
sales-forecast-dashboard/
├── data/
│   └── Superstore.csv
├── notebooks/
│   ├── 01_eda.ipynb          # exploring trend & seasonality
│   └── 02_forecasting.ipynb  # feature engineering + model training
├── models/
│   └── model_results.csv     # saved test scores
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Forecast.py
│   └── 3_Model_Performance.py
├── app.py                    # landing page
├── utils.py                  # shared data-loading helpers
├── requirements.txt
└── README.md
```

---

## 📚 Understanding Time Series Forecasting (A Plain-Language Guide)

This section explains the ideas behind the project in simple terms, so you can revisit and deepen your understanding any time.

### What is time series data?

A time series is simply data recorded in time order — one value per time step. Here it's total sales per week, week after week. The key thing is that **the order matters**: week 5 comes after week 4, and you can't shuffle them around without destroying the meaning. This is the biggest difference from ordinary data (like house prices), where each row stood on its own and order was irrelevant.

### The three things hiding inside a time series

Almost any time series is a blend of three ingredients:

- **Trend** — the long-term direction. Are sales generally climbing or falling across months and years? (Ours rises overall, with dips along the way.)
- **Seasonality** — patterns that repeat on a fixed cycle. Sales spike every November for the holidays, year after year. That repeating shape is seasonality.
- **Noise** — the random jitter that follows no pattern. No model can predict noise; the goal is to capture the trend and seasonality and accept that the noise will always be there.

"Seasonal decomposition" is just the act of splitting one messy line into these three separate, clearer pictures.

### The golden rule: never shuffle time

When you predict the future, you only ever have the past to work with — you never get to peek ahead. So when testing a forecasting model, you must respect that. We train the model on the **earliest** weeks and test it on the **most recent** weeks, keeping everything in order. This is called a **chronological split**.

If you shuffled the data randomly (like in a normal ML project), you'd accidentally let the model train on future weeks to "predict" past ones — impossible in real life, and it makes the model's scores look fake-good. Keeping time in order is what makes the evaluation honest.

### How do you teach a model about the past? Features.

A model like XGBoost can't read a calendar date. So we translate the date and the history into numbers it can actually use:

- **Calendar features** — pull the month and week-of-year out of each date, so the model can learn things like "month 11 means high sales."
- **Lag features** — place earlier values next to the current row. `lag_1` is last week's sales; `lag_52` is the same week one year ago. These are literally the model's *memory* of the past.
- **Rolling features** — a moving average of the last few weeks, which captures recent momentum (have sales been drifting up lately?).

### The trap: data leakage

**Data leakage** is when information from the future accidentally sneaks into your features. The model then looks brilliant during testing but fails completely in the real world, because it was secretly peeking at the answer.

Example: if a "rolling average" feature accidentally includes the current week's own sales, the model is using the very thing it's supposed to predict. We prevented this by **shifting** every feature back one step, so each feature only ever uses genuinely past information. Avoiding leakage is the difference between an honest model and one that's fooling itself.

### The three models we compared

- **Seasonal naive (the baseline)** — a deliberately simple guess: "this week will equal the same week last year." It does no learning at all. Its job is to set the bar — if a sophisticated model can't beat this, the sophisticated model is worthless. Always start with a baseline.
- **Prophet** — a model built specifically for time series. You hand it just dates and values, and it works out the trend and seasonality on its own. It shines when data has clear, repeating patterns. **(This was our winner.)**
- **XGBoost** — a powerful general-purpose machine-learning model. It knows nothing about time, but it's excellent at finding patterns in the lag and calendar features we built. It depends entirely on good feature engineering.

These last two represent two different philosophies: Prophet *understands time directly*, while XGBoost *learns from features we hand-craft for it*.

### How we measured success

Each model was scored on the held-out test weeks using three numbers. For all three, **lower is better**:

- **MAE (Mean Absolute Error)** — the average size of the miss, in dollars. "On average, we're off by $X."
- **RMSE (Root Mean Squared Error)** — similar to MAE, but it punishes big misses more harshly.
- **MAPE (Mean Absolute Percentage Error)** — the average miss as a percentage. The most intuitive one: "on average, we're off by X%."

### Why Prophet won

The sales series here is fairly short (a few years) and strongly seasonal — exactly the conditions a model designed around trend and seasonality handles best. XGBoost needs a lot of history for its lag features to really shine, and here it had less to learn from. Importantly, **both real models beat the naive baseline**, which proves the modeling added genuine value over a no-effort guess. And a lesson worth remembering: the more complex model (XGBoost) wasn't automatically the best — always compare.

---

## Results

Tested on the most recent 26 weeks of data (which no model saw during training):

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **Prophet** ✅ | 5,818 | 7,617 | **36.8%** |
| XGBoost | 6,446 | 7,926 | 41.6% |
| Seasonal naive | 7,706 | 9,721 | 45.0% |

Weekly retail sales are spiky and hard to predict, so these error rates are expected at the weekly level. The meaningful result is that Prophet beat the baseline on the toughest stretch of the year (the holiday season).

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/issawick/sales-forecast-dashboard.git
cd sales-forecast-dashboard

# 2. Create and activate the environment
conda create -n sales-forecast python=3.11 -y
conda activate sales-forecast

# 3. Install dependencies for the app
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

To also run the notebooks (data exploration and model training), install the extra packages used there: `pip install xgboost scikit-learn statsmodels jupyter`.

## Possible Improvements

- Add more forecasting models (e.g. SARIMA) to the comparison
- Let users forecast a specific product category, not just a region
- Show forecast accuracy on a rolling basis instead of a single test window
- Add downloadable forecast results

---

*Built as an end-to-end time series machine learning project: data exploration, feature engineering, model comparison, an interactive multi-page dashboard, and live deployment.*