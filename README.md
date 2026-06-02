# Integrating Machine Learning in Decline Rate Analysis

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-FF6F00?style=for-the-badge)
![Petroleum Engineering](https://img.shields.io/badge/Petroleum%20Engineering-DCA-2E8B57?style=for-the-badge)
![Forecasting](https://img.shields.io/badge/Forecasting-Recursive%20Decline%20Rate-8A2BE2?style=for-the-badge)

## Introduction

Reserve estimation is one of the most important tasks in petroleum and reservoir engineering. Decline Curve Analysis (DCA) is widely used alongside volumetric methods, material balance, and numerical simulation because it can provide a fast estimate of reserves and future production performance.

DCA is based on empirical mathematical relationships between production rate and time. The most common DCA models are:

- **Exponential decline**
- **Hyperbolic decline**
- **Harmonic decline**

In the exponential decline model, the decline rate is assumed to remain constant throughout the well's producing life. Because of this assumption, the forecast can sometimes be either too pessimistic or too optimistic, especially when the actual well behavior changes over time.

## Machine Learning in Forecasting

In recent years, machine learning has been increasingly applied to time-series forecasting. The main idea is that a model can learn patterns from historical data and use those learned patterns to forecast future values.

However, direct machine-learning forecasts of production rate can sometimes produce unreliable results, such as:

- A flat forecast
- Repeating sinusoidal-like patterns
- Forecasts that do not preserve the expected decline behavior

This issue can also occur in production forecasting. Instead of directly predicting oil rate, this project uses the **decline rate** as the machine-learning target. The goal is for the model to learn how the decline rate changes throughout the well's producing life, then use recursive forecasting to estimate future decline rates and future oil rates.

## Methodology

The decline rate is calculated using the exponential decline relationship.

For exponential decline:

$$
q_{t+\Delta t} = q_t e^{-D_i \Delta t}
$$

Rearranging the equation gives the decline rate:

$$
D_i = \frac{\ln\left(\frac{q_t}{q_{t+\Delta t}}\right)}{\Delta t}
$$

where:

- $q_t$ is the oil rate at time $t$
- $q_{t+\Delta t}$ is the oil rate at the next time step
- $D_i$ is the decline rate
- $\Delta t$ is the time interval between two production measurements

After the decline rate is calculated, features are built from the historical production data. These features may include lag features, rolling-window statistics, slopes, and other time-based features. The model then learns the relationship between these features and the decline rate.

## Recursive Forecasting Logic

During forecasting, the last observed oil rate is used as the starting point:

$$
q_i = q_{\text{last}}
$$

The trained model predicts the next decline rate:

$$
\hat{D}_{i+1} = f(X_i)
$$

The predicted decline rate is then used to forecast the next oil rate:

$$
\hat{q}_{i+1} = q_i e^{-\hat{D}_{i+1}\Delta t}
$$

The forecasted oil rate is then added back into the sequence and used to build the next set of lag and rolling-window features. This process continues recursively until the selected forecast horizon is reached or another stopping condition is met.

## Core Idea

Instead of asking the model to directly forecast:

$$
q_{t+1}
$$

the model is trained to forecast:

$$
D_{t+1}
$$

Then the predicted decline rate is converted back into oil rate using the exponential decline equation. This approach keeps the forecast closer to decline-curve behavior while still allowing machine learning to capture changes in the decline trend.

## Summary

This workflow combines the physical intuition of Decline Curve Analysis with the flexibility of machine learning:

- DCA provides the exponential decline relationship.
- Machine learning predicts the changing decline rate.
- Recursive forecasting converts predicted decline rates into future oil rates.
- The final forecast is designed to behave more like a decline curve instead of a purely data-driven black-box prediction.
